"""Regression coverage for lazy curves, boundary Re and completion callbacks."""
from pathlib import Path
import sys
from threading import Event, get_ident

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import database, xfoil_runner
from core.optimizer import _polar_lookup, PRESET_PILOT_STANDARD


def test_lazy_build_and_invalidation(tmp_path, monkeypatch):
    records = []
    for name in ('first', 'second'):
        for re in (1000000, 2000000):
            for alpha in range(6):
                records.append(dict(airfoil=name, thickness_pct=25., max_thickness_pos_pct=42.,
                                    le_radius_ratio=1., max_camber_pct=0., reynolds=re,
                                    alpha=alpha, cl=0., cd=.01, drag_counts=100., ld=0., cm=0.,
                                    top_xtr=.5, bot_xtr=.5, converged=True))
    counts = {'1d': 0, '2d': 0}
    old_1d, old_2d = database.interp1d, database.LinearNDInterpolator
    def build_1d(*args, **kwargs):
        counts['1d'] += 1
        return old_1d(*args, **kwargs)
    def build_2d(*args, **kwargs):
        counts['2d'] += 1
        return old_2d(*args, **kwargs)
    monkeypatch.setattr(database, 'interp1d', build_1d)
    monkeypatch.setattr(database, 'LinearNDInterpolator', build_2d)
    db = database.Database3DManager(tmp_path / 'db.csv')
    db.append_records(records)
    assert not db.interpolators
    assert counts == {'1d': 0, '2d': 0}
    curve = db.get_polar_interpolator('first', 1000000)
    assert curve(2.5) == pytest.approx(.01)
    assert db.interpolators[('first', 1000000)] is curve
    assert db.get_polar_interpolator('first', 1000000) is curve
    assert counts == {'1d': 1, '2d': 0}
    assert db.get_polar_interpolator('first')(1500000, 2.5) == pytest.approx(.01)
    assert counts == {'1d': 1, '2d': 1}
    with pytest.raises(KeyError):
        db.get_polar_interpolator('missing')
    db.append_records(records[:1])
    assert not db.interpolators
    assert db.get_polar_interpolator('first', 1000000) is not curve


def test_reynolds_boundary_clamping():
    frame = pd.DataFrame([dict(reynolds=re, alpha=a, cd=cd)
                          for re, cd in [(1000000, .02), (2000000, .01)]
                          for a in [0., 6., 12.]])
    assert _polar_lookup(frame, 500000, 6)(3) == pytest.approx(.02)
    assert _polar_lookup(frame, 3000000, 6)(3) == pytest.approx(.01)
    assert _polar_lookup(frame, 1500000, 6)(3) == pytest.approx(.015)
    assert len(PRESET_PILOT_STANDARD) == 4
    assert PRESET_PILOT_STANDARD[-1]['width_mm'] == 460.


def test_progress_arrives_in_completion_order(tmp_path, monkeypatch):
    release_first = Event()
    caller = get_ident()
    progress = []
    def run(dat, re, *args, **kwargs):
        if re == 1000000:
            assert release_first.wait(3), 'Callback blocked behind first submitted case'
        return [{'reynolds': re}]
    def callback(done, total):
        assert get_ident() == caller
        progress.append((done, total))
        release_first.set()
    monkeypatch.setattr(xfoil_runner, 'run_single_xfoil_case', run)
    name, result = xfoil_runner.analyze_funcgen_airfoil_all_re(
        25., 42., 1., [1000000, 2000000, 1000000],
        airfoils_dir=tmp_path, progress_callback=callback, max_workers=2)
    assert progress == [(1, 2), (2, 2)]
    assert [row['reynolds'] for row in result] == [1000000, 2000000]
    assert all(row['airfoil'] == name for row in result)
    progress.clear()
    assert xfoil_runner.analyze_funcgen_airfoil_all_re(
        25., 42., 1., [], airfoils_dir=tmp_path, progress_callback=callback)[1] == []
    assert not progress
