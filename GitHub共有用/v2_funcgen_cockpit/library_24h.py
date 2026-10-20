"""Bounded, full-CPU XFOIL library builder with durable per-case recovery.

Results are journalled in SQLite, periodically merged atomically into the CSV,
and namespaced by geometry revision. Does not require a live Codex session.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import numpy as np
from core.database import COLUMNS
from core.geometry import generate_funcgen_airfoil, save_airfoil_dat
from core.paths import DEFAULT_DATABASE, DEFAULT_GRID_CONFIG, DATABASE_DIR
from core.xfoil_runner import format_foil_name_3d, run_single_xfoil_case
from grid_expander_3d_cli import plan_grid, analysis_ranges


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def signature(path):
    if not path.exists():
        return None
    s = path.stat()
    return (s.st_size, s.st_mtime_ns)


def case_name(tc, xt, kle, method):
    revision = 'warp_c3_v1' if method == 'warp' else 'cst_eq_v1'
    return format_foil_name_3d(tc, xt, kle)+'_'+revision


def alpha_key(value):
    return round(float(value), 6)


def record_key(row):
    return (row['airfoil'], row.get('generator_method') or 'warp',
            round(float(row['reynolds'])), alpha_key(row['alpha']))


def read_coverage(path, desired_alphas):
    coverage = {}
    if not path.exists():
        return coverage
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            # Only this revision's unambiguous names can be reused.
            if not row['airfoil'].endswith(('_warp_c3_v1', '_cst_eq_v1')):
                continue
            if row.get('converged', '').lower() not in ('true', '1', '1.0'):
                continue
            try:
                key = record_key(row)
                if float(row['cd']) > 0 and key[-1] in desired_alphas:
                    coverage.setdefault(key[:3], set()).add(key[-1])
            except (ValueError, KeyError):
                continue
    return coverage


def open_journal(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS cases (name TEXT, method TEXT, re INTEGER, '
               'state TEXT, rows_json TEXT, error TEXT, PRIMARY KEY(name,method,re))')
    db.commit()
    return db


def publish(journal, original, target, expected_signature):
    """Stream original rows and journal rows; replace only identical record keys."""
    if signature(target) != expected_signature:
        raise RuntimeError('Database changed externally; journal preserved, merge deferred')
    new_keys = set()
    for (payload,) in journal.execute("SELECT rows_json FROM cases WHERE rows_json != '[]'"):
        for row in json.loads(payload):
            new_keys.add(record_key(row))
    if not new_keys:
        return expected_signature, 0
    if shutil.disk_usage(target.parent).free < (original.stat().st_size if original.exists() else 0) + 2_000_000_000:
        raise OSError('Less than required disk space for atomic CSV merge')
    temporary = target.with_name(target.name+'.24h.tmp')
    count = 0
    with temporary.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        if original.exists():
            with original.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    try:
                        replaced = record_key(row) in new_keys
                    except (ValueError, KeyError):
                        replaced = False
                    if not replaced:
                        row.setdefault('generator_method', 'warp')
                        row.setdefault('le_radius_ratio', '1.0')
                        writer.writerow(row)
        for (payload,) in journal.execute("SELECT rows_json FROM cases WHERE rows_json != '[]' ORDER BY name,method,re"):
            for row in json.loads(payload):
                writer.writerow(row)
                count += 1
        output.flush()
        os.fsync(output.fileno())
    if signature(target) != expected_signature:
        raise RuntimeError('Database changed during merge; original CSV left untouched')
    os.replace(temporary, target)
    return signature(target), count


def stages(config):
    yield 'existing_grid', config
    dense = dict(config, tc_step=.1, xt_step=.1,
                 kle_list=[round(.6+i*.1, 2) for i in range(10)])
    yield 'dense_grid', dense


def task_stream(config, coverage, run_dir, counters, deadline):
    re_list, alphas = analysis_ranges(config)
    desired = {alpha_key(a) for a in alphas}
    seen = set()
    foil_root = DATABASE_DIR/'airfoils'
    foil_root.mkdir(parents=True, exist_ok=True)
    for stage, grid in stages(config):
        points, _ = plan_grid(grid)
        for tc, xt, kle in points:
            for method in ('warp', 'cst'):
                if time.monotonic() >= deadline or (run_dir/'STOP').exists():
                    return
                name = case_name(tc, xt, kle, method)
                if name in seen:
                    continue
                seen.add(name)
                missing = [re for re in re_list if not desired.issubset(coverage.get((name, method, re), set()))]
                counters['skipped_complete_cases'] += len(re_list)-len(missing)
                if not missing:
                    continue
                try:
                    x, y = generate_funcgen_airfoil(tc, xt, kle, method=method)
                    _, dense_y = generate_funcgen_airfoil(tc, xt, kle, num_points=1001, method=method)
                    if not np.isfinite(dense_y).all() or np.max(dense_y) > tc/200+1e-5:
                        raise ValueError('Geometry exceeds requested maximum thickness')
                    nose = int(np.argmin(x))
                    if np.min(y[:nose+1]) < -1e-10:
                        raise ValueError('Negative upper half-thickness')
                    dat = foil_root/(name+'.dat')
                    if dat.exists():
                        old = np.loadtxt(dat, skiprows=1)
                        if old.shape != (len(x), 2) or not np.allclose(old, np.column_stack((x,y)), rtol=0, atol=1e-11):
                            raise ValueError('Geometry revision/file mismatch')
                    else:
                        save_airfoil_dat(str(dat), x, y, name)
                    digest = hashlib.sha256(dat.read_bytes()).hexdigest()
                    with (run_dir/'geometry_manifest.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(airfoil=name, method=method, sha256=digest,
                            tc=tc, xt=xt, kle=kle, stage=stage))+'\n')
                except Exception as error:
                    counters['geometry_failures'] += 1
                    print(json.dumps({'geometry_error': name, 'error': str(error)}), flush=True)
                    continue
                for re in missing:
                    yield dict(name=name, method=method, tc=tc, xt=xt, kle=kle, re=re,
                               dat=str(dat), stage=stage, desired=desired)


def compute(task, config):
    rows = run_single_xfoil_case(task['dat'], task['re'], config['alpha_min'],
        config['alpha_max'], config['alpha_step'], timeout_sec=30, panel_count=180, ncrit=9.)
    for row in rows:
        row.update(airfoil=task['name'], generator_method=task['method'], thickness_pct=task['tc'],
            max_thickness_pos_pct=task['xt'], le_radius_ratio=task['kle'], max_camber_pct=0.)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=24.)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--config', type=Path, default=DEFAULT_GRID_CONFIG)
    parser.add_argument('--publish-seconds', type=float, default=1800.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.hours) or not 0 < args.hours <= 24 or not 1 <= args.workers <= 64 or args.publish_seconds <= 0:
        parser.error('hours must be (0,24], workers [1,64], publish-seconds positive')
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir/'journal.sqlite').exists():
        parser.error('Use a fresh run directory; previous journal is preserved')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    re_list, alphas = analysis_ranges(config)
    plan_grid(config)
    start = time.monotonic()
    deadline = start+args.hours*3600
    utc_start = datetime.now(timezone.utc)
    atomic_json(run_dir/'status.json', dict(pid=os.getpid(), state='initializing',
        started_utc=utc_start.isoformat(), workers=args.workers,
        stop_new_work_utc=(utc_start+timedelta(hours=args.hours)).isoformat()))
    atomic_json(run_dir/'config_snapshot.json', config)
    code_dir = run_dir/'code_snapshot'
    code_dir.mkdir(exist_ok=True)
    for relative in ('library_24h.py', 'core/geometry.py', 'core/xfoil_runner.py',
                     'core/generators/funcgen_warp.py', 'core/generators/funcgen_cst.py'):
        source = Path(__file__).resolve().parent/relative
        destination = code_dir/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    lock = (DATABASE_DIR/'library_builder.lock').open('a+b')
    if sys.platform == 'win32':
        import msvcrt
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    original = run_dir/'database_before.csv'
    expected = signature(DEFAULT_DATABASE)
    if expected:
        shutil.copy2(DEFAULT_DATABASE, original)
        if signature(DEFAULT_DATABASE) != expected:
            raise RuntimeError('Database changed while taking backup')
    coverage = read_coverage(original, {alpha_key(a) for a in alphas})
    journal = open_journal(run_dir/'journal.sqlite')
    counters = dict(submitted_cases=0, finished_cases=0, complete_cases=0, partial_cases=0,
        failed_cases=0, converged_rows=0, skipped_complete_cases=0, geometry_failures=0,
        published_rows=0)
    state = dict(pid=os.getpid(), state='running', started_utc=utc_start.isoformat(),
        stop_new_work_utc=(utc_start+timedelta(hours=args.hours)).isoformat(),
        workers=args.workers, config=config, methods=['warp','cst'],
        panels=180, ncrit=9., database=str(DEFAULT_DATABASE), run_dir=str(run_dir))
    last_status = last_publish = time.monotonic()
    last_publish_error = None
    reason = 'grid_exhausted'
    dirty = False
    consecutive_failures = 0
    def status(label):
        atomic_json(run_dir/'status.json', dict(state, state=label, **counters,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.monotonic()-start, 1), last_publish_error=last_publish_error))
    status('running')
    tasks = iter(task_stream(config, coverage, run_dir, counters, deadline))
    exhausted = False
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {}
            while pending or not exhausted:
                stop = (run_dir/'STOP').exists() or time.monotonic() >= deadline
                if shutil.disk_usage(run_dir).free < 5_000_000_000:
                    reason, exhausted = 'low_disk_space', True
                if consecutive_failures >= max(60, args.workers*10):
                    reason, exhausted = 'repeated_case_failures', True
                if stop:
                    reason = 'stop_file' if (run_dir/'STOP').exists() else '24h_deadline'
                    exhausted = True
                while not exhausted and len(pending) < args.workers:
                    task = next(tasks, None)
                    if task is None:
                        exhausted = True
                        if time.monotonic() >= deadline:
                            reason = '24h_deadline'
                        elif (run_dir/'STOP').exists():
                            reason = 'stop_file'
                        break
                    if time.monotonic() >= deadline or (run_dir/'STOP').exists():
                        exhausted = True
                        reason = '24h_deadline' if time.monotonic() >= deadline else 'stop_file'
                        break
                    pending[pool.submit(compute, task, config)] = task
                    counters['submitted_cases'] += 1
                if pending:
                    done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
                    for future in done:
                        task = pending.pop(future)
                        error = ''
                        try:
                            rows = future.result()
                        except Exception as exception:
                            rows, error = [], str(exception)[:2000]
                        complete = task['desired'].issubset({alpha_key(r['alpha']) for r in rows})
                        kind = 'complete' if complete else ('partial' if rows else 'failed')
                        journal.execute('INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,?)',
                            (task['name'], task['method'], task['re'], kind, json.dumps(rows), error))
                        journal.commit()
                        counters['finished_cases'] += 1
                        counters[kind+'_cases'] += 1
                        consecutive_failures = consecutive_failures+1 if kind == 'failed' else 0
                        counters['converged_rows'] += len(rows)
                        dirty = dirty or bool(rows)
                now = time.monotonic()
                if dirty and now-last_publish >= args.publish_seconds and now < deadline:
                    try:
                        expected, counters['published_rows'] = publish(journal, original, DEFAULT_DATABASE, expected)
                        dirty, last_publish_error = False, None
                    except Exception as error:
                        last_publish_error = str(error)
                        print('MERGE_DEFERRED '+str(error), flush=True)
                    last_publish = time.monotonic()
                if now-last_status >= 5:
                    status('draining' if exhausted and pending else 'running')
                    last_status = now
    finally:
        status('saving')
        if dirty:
            try:
                expected, counters['published_rows'] = publish(journal, original, DEFAULT_DATABASE, expected)
                last_publish_error = None
            except Exception as error:
                last_publish_error = str(error)
        state['stop_reason'] = reason
        status('completed' if last_publish_error is None else 'completed_merge_pending')
        journal.close()
        lock.close()
        if sys.platform == 'win32':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print(json.dumps(dict(state, **counters, last_publish_error=last_publish_error)), flush=True)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
