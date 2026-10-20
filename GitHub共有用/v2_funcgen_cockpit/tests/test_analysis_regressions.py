"""Exercise cache recovery, legacy imports, and isolated XFOIL failure paths."""
import subprocess
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.database import Database3DManager
from core import xfoil_runner


def rows():
    return [dict(airfoil='legacy', thickness_pct=25., max_thickness_pos_pct=42.,
                 le_radius_ratio=1., max_camber_pct=0., reynolds=1000000,
                 alpha=float(a), cl=0., cd=.01, drag_counts=100., ld=0., cm=0.,
                 top_xtr=.5, bot_xtr=.5, converged=True) for a in range(6)]


def test_cache_import_and_recovery(tmp_path):
    legacy = tmp_path / 'legacy.csv'
    pd.DataFrame(rows()).drop(columns='le_radius_ratio').to_csv(legacy, index=False)
    db = Database3DManager(tmp_path / 'v2.csv', tmp_path / 'cache.pkl')
    assert db.load_database(legacy)
    assert len(db.df) == 6
    assert db.load_database(legacy)
    assert len(db.df) == 6  # Import must not repeat on a cache hit.
    db.cache_path.write_bytes(b'broken pickle')
    assert db.load_database(legacy)
    assert (25., 42., 1.) in db.get_completed_keys()
    assert db.interpolators[('legacy', 1000000)](2.5) == pytest.approx(.01)
    # External CSV modification must invalidate the existing cache.
    updated = pd.DataFrame(rows())
    updated['cd'] = .02
    updated.to_csv(db.csv_path, index=False)
    assert db.load_database(legacy)
    assert db.df.cd.iloc[0] == .02


def test_distinct_alpha_completion_and_append_preserves_disk(tmp_path):
    db = Database3DManager(tmp_path / 'db.csv', tmp_path / 'cache.pkl')
    assert db.append_records([rows()[0]] * 6) == 6
    assert not db.get_completed_keys()
    other = Database3DManager(db.csv_path, db.cache_path)
    assert other.append_records(rows()[1:]) == 5
    assert len(other.df) == 11
    assert (25., 42., 1.) in other.get_completed_keys()


def test_timeout_retains_polar_and_cleans_workspace(tmp_path, monkeypatch):
    dat = tmp_path / 'input.dat'
    dat.write_text('test\n1 0\n0 0\n1 0\n')
    executable = tmp_path / 'xfoil.exe'
    executable.touch()
    workspaces = []
    def timeout_run(args, **kwargs):
        work = Path(kwargs['cwd'])
        workspaces.append(work)
        assert (work / 'airfoil.dat').exists()
        (work / 'output.pol').write_text(
            'alpha CL CD CDp CM Top_Xtr Bot_Xtr\n'
            '0 0 .01 .005 0 .5 .5\n'
            '1 0 ***** .005 0 .5 .5\n'
            '2 0 nan .005 0 .5 .5\n')
        raise subprocess.TimeoutExpired(args, 1)
    monkeypatch.setattr(xfoil_runner.subprocess, 'run', timeout_run)
    with pytest.warns(UserWarning, match='timed out'):
        result = xfoil_runner.run_single_xfoil_case(dat, 1000000, xfoil_exe=executable)
    assert len(result) == 1
    assert result[0]['drag_counts'] == 100
    assert all(not p.exists() for p in workspaces)
