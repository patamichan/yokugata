"""Small CLI dry-runs, CAD units and orderly interruption; no grid calculation."""
import json
from pathlib import Path
import sys
from threading import Event

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grid_expander_3d_cli as grid
import cockpit_selection_3d_cli as selection
from core import xfoil_runner


def config():
    # Fixed test fixture: user-editable production ranges must not change this test.
    return dict(tc_min=21., tc_max=28., tc_step=.5,
                xt_min=39., xt_max=46., xt_step=.5, kle_list=[.8, 1., 1.2],
                center_tc=25., center_xt=42., center_kle=1.,
                reynolds_min=500000, reynolds_max=2500000, reynolds_step=50000,
                alpha_min=0., alpha_max=12., alpha_step=.1)


def test_preview_has_no_writes_or_analysis(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config()))
    monkeypatch.setattr(grid, 'analyze_funcgen_airfoil_all_re',
                        lambda *a, **k: pytest.fail('Preview launched analysis'))
    assert grid.main(['--config', str(path), '--csv', str(tmp_path/'missing.csv'), '--dry-run']) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ['config.json']
    assert selection.main(['--dry-run', '--csv', str(tmp_path/'missing.csv')]) == 0


def test_plan_shells_and_invalid_resolution():
    cfg = config()
    points, count = grid.plan_grid(cfg, {(25., 42., 1.)})
    assert count == 675
    assert len(points) == count-1
    assert (25., 42., 1.) not in points
    assert (25., 42., 1.) == grid.plan_grid(cfg)[0][0]
    cfg['tc_step'] = .01
    with pytest.raises(ValueError, match='0.1%'):
        grid.plan_grid(cfg)


def test_cad_units_and_generator_consistency(tmp_path):
    result = dict(airfoil='test', generator_method='cst', thickness_pct=25.,
                  max_thickness_pos_pct=42., le_radius_ratio=1.2, chord_mm=2500.,
                  cd_eff=.01, drag_area_mm=25.)
    folder = selection.export_cad_package(result, [], 14., 1.5, tmp_path)
    scaled = np.loadtxt(folder/'section_scaled_mm.dat', skiprows=1)
    normalized = np.loadtxt(folder/'section_normalized.dat', skiprows=1)
    np.testing.assert_allclose(scaled, normalized*2500, atol=2e-9)
    assert scaled[0, 0] == 2500.
    assert 'cst' in (folder/'選定結果仕様書.txt').read_text(encoding='utf-8')


def test_grid_interruption_saves_completed_records(tmp_path, monkeypatch):
    cfg = config()
    cfg.update(tc_min=25., tc_max=25., xt_min=42., xt_max=42., kle_list=[1.])
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    saved = []
    class FakeDB:
        def __init__(self, **kwargs):
            pass
        def get_completed_keys(self, method):
            return set()
        def append_records(self, rows):
            saved.extend(rows)
            return len(rows)
    def interrupt(*args, **kwargs):
        kwargs['records_callback']([{'completed': True}])
        raise KeyboardInterrupt
    monkeypatch.setattr(grid, 'Database3DManager', FakeDB)
    monkeypatch.setattr(grid, 'analyze_funcgen_airfoil_all_re', interrupt)
    monkeypatch.setattr('builtins.input', lambda prompt: 'START')
    assert grid.main(['--config', str(path), '--csv', str(tmp_path/'db.csv'), '--run']) == 130
    assert saved == [{'completed': True}]


def test_runner_interrupt_preserves_running_case(tmp_path, monkeypatch):
    started = Event()
    def run(path, re, *args, **kwargs):
        started.set()
        return [dict(reynolds=re)]
    monkeypatch.setattr(xfoil_runner, 'run_single_xfoil_case', run)
    def interrupt(futures):
        assert started.wait(3)
        raise KeyboardInterrupt
    monkeypatch.setattr(xfoil_runner, 'as_completed', interrupt)
    collected = []
    with pytest.raises(KeyboardInterrupt):
        xfoil_runner.analyze_funcgen_airfoil_all_re(25., 42., 1., [1000000],
            airfoils_dir=tmp_path, records_callback=collected.extend)
    assert [row['reynolds'] for row in collected] == [1000000]
