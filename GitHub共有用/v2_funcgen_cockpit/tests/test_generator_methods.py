"""Exact CST constraints and end-to-end isolation of generator methods."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import geometry, xfoil_runner
from core.database import Database3DManager
from core.generators import funcgen_cst
from core.generators.funcgen_warp import FuncGenWarpGenerator
from core.optimizer import optimize_section_3d


@pytest.mark.parametrize('tc,xt,kle', [(20., 38., .6), (25., 42., 1.), (30., 48., 1.5)])
def test_cst_constraints_and_nose_radius(tc, xt, kle):
    coefficients = np.asarray(funcgen_cst._weights(5, tc/100, xt/100, kle))
    assert coefficients[0]**2/2 == pytest.approx(1.1019*(tc/100)**2*kle, abs=1e-13)
    assert funcgen_cst._basis(5, xt/100) @ coefficients == pytest.approx(tc/200, abs=1e-9)
    assert funcgen_cst._slope_basis(5, xt/100) @ coefficients == pytest.approx(0., abs=1e-9)
    x, y = geometry.generate_funcgen_airfoil(tc, xt, kle, method='cst', num_points=160)
    assert len(x) == len(y) == 160
    assert np.isfinite(y).all()
    assert y[0] == y[-1] == 0.


def test_failures_are_explicit(monkeypatch):
    with pytest.raises(ValueError, match='Unknown generator'):
        geometry.generate_funcgen_airfoil(25., 42., method='typo')
    monkeypatch.setattr(funcgen_cst, 'minimize', lambda *a, **k:
                        SimpleNamespace(success=False, x=np.ones(5), message='test failure'))
    funcgen_cst._weights.cache_clear()
    try:
        with pytest.raises(RuntimeError, match='test failure'):
            geometry.generate_funcgen_airfoil(25., 42., 1.1, method='cst')
    finally:
        funcgen_cst._weights.cache_clear()


def test_warp_entry_points_match():
    x, y = geometry.generate_funcgen_airfoil(25., 42., 1.3)
    xx, yy = FuncGenWarpGenerator().generate(25., 42., 1.3)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(y, yy)
    _, base = geometry.generate_funcgen_airfoil(25., 42., 1.)
    np.testing.assert_array_equal(y[x >= .2], base[x >= .2])


def records(method, cd):
    return [dict(airfoil='same-name', generator_method=method, thickness_pct=25.,
                 max_thickness_pos_pct=42., le_radius_ratio=1., max_camber_pct=0.,
                 reynolds=1000000, alpha=float(a), cl=0., cd=cd,
                 drag_counts=cd*10000, ld=0., cm=0., top_xtr=.5, bot_xtr=.5,
                 converged=True) for a in range(13)]


def test_database_queries_and_optimizer_keep_methods_separate(tmp_path):
    db = Database3DManager(tmp_path / 'methods.csv')
    db.append_records(records('warp', .02) + records('cst', .01))
    assert db.get_polar_interpolator('same-name', 1000000)(2) == pytest.approx(.02)
    assert db.get_polar_interpolator('same-name', 1000000, 'cst')(2) == pytest.approx(.01)
    profile = [dict(x_mm=100., width_mm=320., clearance_mm=15.)]
    for method, cd in [('warp', .02), ('cst', .01)]:
        result = optimize_section_3d(db, profile, generator_method=method, crosswind_ms=0.)
        assert result['generator_method'] == method
        assert result['cd_eff'] == pytest.approx(cd)
    reloaded = Database3DManager(db.csv_path)
    assert reloaded.load_database(tmp_path / 'no-legacy.csv')
    assert set(reloaded.df.generator_method) == {'warp', 'cst'}
    partial = Database3DManager(tmp_path / 'partial.csv')
    partial.append_records(records('warp', .02)[:3] + records('cst', .01)[3:6])
    assert not partial.get_completed_keys()
    assert not partial.get_completed_keys('cst')
    assert not partial.available_airfoils_by_method['warp']
    assert not partial.available_airfoils_by_method['cst']


def test_runner_separate_files_metadata_and_stale_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(xfoil_runner, 'run_single_xfoil_case',
                        lambda path, re, *args, **kwargs: [dict(reynolds=re, cd=.01)])
    names = []
    for method in ['warp', 'cst']:
        name, rows = xfoil_runner.analyze_funcgen_airfoil_all_re(
            25., 42., 1.2, [1000000], airfoils_dir=tmp_path, generator_method=method)
        names.append(name)
        assert rows[0]['generator_method'] == method
        assert (tmp_path / (name+'.dat')).exists()
    assert names[0] != names[1]
    path = tmp_path / (names[0]+'.dat')
    contents = np.loadtxt(path, skiprows=1)
    contents[2, 1] += .01
    np.savetxt(path, contents, header=names[0], comments='')
    with pytest.raises(ValueError, match='coordinates differ'):
        xfoil_runner.analyze_funcgen_airfoil_all_re(
            25., 42., 1.2, [1000000], airfoils_dir=tmp_path)
