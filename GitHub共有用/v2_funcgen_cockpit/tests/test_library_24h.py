"""Bounded library job persistence and merge contracts; no actual bulk analysis."""
import csv
import json
import time

import pytest
import library_24h as job
from core.database import COLUMNS


def row(name, alpha=0., cd=.01):
    return dict(airfoil=name, generator_method='warp', thickness_pct=24., max_thickness_pos_pct=42.,
        le_radius_ratio=1.2, max_camber_pct=0., reynolds=1200000, alpha=alpha,
        cl=0., cd=cd, drag_counts=cd*10000, ld=0., cm=0., top_xtr=.5, bot_xtr=.5, converged=True)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_streaming_merge_preserves_old_and_replaces_only_same_row(tmp_path):
    original, target = tmp_path/'original.csv', tmp_path/'library.csv'
    name = 'test_warp_c3_v1'
    write_csv(original, [row('legacy'), row(name, 0., .02), row(name, 1., .03)])
    target.write_bytes(original.read_bytes())
    journal = job.open_journal(tmp_path/'journal.sqlite')
    fresh = [row(name, 0., .01), row(name, 2., .04)]
    journal.execute('INSERT INTO cases VALUES (?,?,?,?,?,?)', (name, 'warp',1200000,'partial',json.dumps(fresh),''))
    journal.commit()
    expected, count = job.publish(journal, original, target, job.signature(target))
    assert count == 2
    with target.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert next(r for r in rows if r['airfoil'] == name and float(r['alpha']) == 1.)['cd'] == '0.03'
    before = target.read_bytes()
    expected, _ = job.publish(journal, original, target, expected)
    assert target.read_bytes() == before
    target.write_text('externally changed')
    with pytest.raises(RuntimeError, match='externally'):
        job.publish(journal, original, target, expected)
    assert target.read_text() == 'externally changed'
    journal.close()


def test_coverage_requires_every_requested_alpha_and_version(tmp_path):
    path = tmp_path/'rows.csv'
    name = 'test_warp_c3_v1'
    write_csv(path, [row('old'),row(name),row(name,1.)])
    coverage = job.read_coverage(path, {0.,1.,2.})
    assert not {0.,1.,2.}.issubset(coverage[(name,'warp',1200000)])
    assert not any(k[0] == 'old' for k in coverage)


def test_stop_file_prevents_any_geometry_work(tmp_path):
    (tmp_path/'STOP').touch()
    config = dict(tc_min=24.,tc_max=24.,tc_step=.5,xt_min=42.,xt_max=42.,xt_step=.5,
        kle_list=[1.2],center_tc=24.,center_xt=42.,center_kle=1.2,
        reynolds_min=1200000,reynolds_max=1200000,reynolds_step=300000,
        alpha_min=0.,alpha_max=2.,alpha_step=1.)
    counters = dict(skipped_complete_cases=0,geometry_failures=0)
    assert list(job.task_stream(config, {}, tmp_path, counters, time.monotonic()+60)) == []


def test_short_job_drains_and_publishes(tmp_path, monkeypatch):
    target = tmp_path/'library.csv'
    write_csv(target, [row('legacy')])
    monkeypatch.setattr(job, 'DATABASE_DIR', tmp_path)
    monkeypatch.setattr(job, 'DEFAULT_DATABASE', target)
    monkeypatch.setattr(job, 'stages', lambda cfg: [('one',cfg)])
    monkeypatch.setattr(job, 'compute', lambda task, cfg:
        [dict(row(task['name'], float(a)), generator_method=task['method']) for a in (0,1,2)])
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(tc_min=24.,tc_max=24.,tc_step=.5,xt_min=42.,xt_max=42.,xt_step=.5,
        kle_list=[1.2],center_tc=24.,center_xt=42.,center_kle=1.2,
        reynolds_min=1200000,reynolds_max=1200000,reynolds_step=300000,
        alpha_min=0.,alpha_max=2.,alpha_step=1.)))
    run = tmp_path/'run'
    assert job.main(['--run-dir',str(run),'--hours','.01','--workers','2','--config',str(config)]) == 0
    status = json.loads((run/'status.json').read_text())
    assert status['state'] == 'completed'
    assert status['finished_cases'] == 2
    assert status['converged_rows'] == 6
    assert status['published_rows'] == 6
    assert (run/'database_before.csv').read_text().startswith('airfoil,')
