"""Exactly two research geometries at alpha=0,1,2; retain raw XFOIL evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from core.geometry import save_airfoil_dat
from core.paths import RESEARCH_DIR
from tests.benchmarks.benchmark_b_local import base_polynomial, solve_local, L, CHORD, BASE_CHORD

BASE_DIR = Path(__file__).resolve().parents[2]
ALPHAS = (0., 1., 2.)


def numeric_rows(path, minimum):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            values = [float(v.replace('D', 'E')) for v in line.split()]
        except ValueError:
            continue
        if len(values) >= minimum and np.isfinite(values).all():
            rows.append(values)
    return rows


def cp_summary(path):
    data = numeric_rows(path, 2)
    if not data:
        return None
    # XFOIL variants write x,Cp or x,y,Cp. Cp is the last column.
    a = np.array(data)
    nose = int(np.argmin(a[:, 0]))
    answer = {}
    for name, side in [('upper', a[:nose+1]), ('lower', a[nose:])]:
        side = side[np.argsort(side[:, 0])]
        forward = side[(side[:, 0] >= .05) & (side[:, 0] <= .2)]
        if not len(forward):
            continue
        i = int(np.argmin(forward[:, -1]))
        cp = forward[:, -1]
        answer[name] = dict(cp_min_forward=float(cp[i]), x_at_min=float(forward[i, 0]),
                            cp_change_05_to_20=float(cp[-1]-cp[0]))
    return answer


def boundary_summary(path):
    if not path.exists():
        return None
    headers = path.read_text(encoding='utf-8', errors='replace').splitlines()[:3]
    header = next((line.lstrip('#').split() for line in headers if 'Cf' in line), [])
    if 'Cf' not in header or 'x' not in header or 'y' not in header:
        return dict(status='unrecognized columns; inspect raw dump')
    data = numeric_rows(path, len(header))
    if not data:
        return None
    cf, xx, yy = header.index('Cf'), header.index('x'), header.index('y')
    # Wake rows (x>1) do not represent surface separation.
    surface = np.array(data)
    surface = surface[(surface[:, xx] > 0) & (surface[:, xx] < 1)]
    answer = {}
    for name, mask in [('upper', surface[:, yy] >= 0), ('lower', surface[:, yy] < 0)]:
        side = surface[mask]
        separated = side[side[:, cf] < 0]
        answer[name] = dict(negative_cf_points=len(separated),
                            x_negative_cf=separated[:, xx].tolist())
    return answer


def run_case(label, chord, coords, output):
    folder = output/label
    folder.mkdir(parents=True, exist_ok=True)
    dat = folder/'airfoil.dat'
    save_airfoil_dat(str(dat), *coords, name=f'Research_{label}')
    re = 14.*chord/1000/1.51e-5
    commands = ['plop', 'g', '', 'load airfoil.dat', 'ppar', 'n 180', '', '',
                'pane', 'oper', 'iter 200', f'visc {re:.12g}', 'vpar', 'n 9', '',
                'pacc', 'polar.pol', '']
    for alpha in ALPHAS:
        commands += [f'alfa {alpha:g}', f'cpwr cp_{alpha:g}.txt', f'dump bl_{alpha:g}.txt']
    commands += ['pacc', '', 'quit', '']
    (folder/'commands.txt').write_text('\n'.join(commands), encoding='ascii')
    with tempfile.TemporaryDirectory(prefix='cockpit_aero_') as work:
        shutil.copyfile(dat, Path(work)/'airfoil.dat')
        timed_out = False
        try:
            proc = subprocess.run([str(BASE_DIR/'xfoil.exe')], cwd=work,
                input='\n'.join(commands), text=True, capture_output=True, timeout=60,
                creationflags=0x08000000 if sys.platform == 'win32' else 0)
            log, code = proc.stdout+'\n'+proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as error:
            timed_out, code = True, None
            log = error.stdout or b''
            if isinstance(log, bytes):
                log = log.decode('utf-8', errors='replace')
        (folder/'xfoil.log').write_text(log, encoding='utf-8')
        for name in ['polar.pol']+[f'{prefix}_{a:g}.txt' for a in ALPHAS for prefix in ('cp', 'bl')]:
            if (Path(work)/name).exists():
                shutil.copyfile(Path(work)/name, folder/name)
    return summarize_case(label, chord, folder, timed_out, code)


def summarize_case(label, chord, folder, timed_out=False, code=0):
    """Reparse retained raw output without launching XFOIL."""
    re = 14.*chord/1000/1.51e-5
    dat = folder/'airfoil.dat'
    rows = numeric_rows(folder/'polar.pol', 7)
    result = dict(label=label, chord_mm=chord, reynolds=re, ncrit=9., panels=180,
                  dat_sha256=hashlib.sha256(dat.read_bytes()).hexdigest(),
                  timeout=timed_out, returncode=code, alphas=[])
    for alpha in ALPHAS:
        matches = [r for r in rows if abs(r[0]-alpha) < 1e-5 and r[2] > 0]
        if not matches:
            result['alphas'].append(dict(alpha=alpha, converged=False))
            continue
        a, cl, cd, cdp, cm, top, bottom = matches[-1][:7]
        result['alphas'].append(dict(alpha=a, converged=True, cl=cl, cd=cd, cdp=cdp, cm=cm,
             drag_area_mm=chord*cd,
             top_xtr=top, bot_xtr=bottom, cp=cp_summary(folder/f'cp_{alpha:g}.txt'),
             boundary_layer=boundary_summary(folder/f'bl_{alpha:g}.txt')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=RESEARCH_DIR/'results_aero_eval')
    parser.add_argument('--summarize-only', action='store_true', help='Parse saved evidence; no XFOIL execution')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = base_polynomial()
    delta, _ = solve_local(base)
    x = np.sin(np.linspace(0, np.pi/2, 201))**2
    q = np.sqrt(x/L)
    y = base(q)
    edited = y.copy()
    edited[x < L] += delta(q[x < L])
    def coordinates(half):
        half = half.copy()
        half[0] = half[-1] = 0.
        return np.r_[x[::-1], x[1:]], np.r_[half[::-1], -half[1:]]
    if args.summarize_only:
        previous = json.loads((args.output_dir/'summary.json').read_text(encoding='utf-8'))
        results = [summarize_case(c['label'], c['chord_mm'], args.output_dir/c['label'],
                                  c['timeout'], c['returncode']) for c in previous['cases']]
    else:
        results = [run_case('A', BASE_CHORD, coordinates(y), args.output_dir),
                   run_case('B', CHORD, coordinates(edited), args.output_dir)]
    comparisons = []
    for a, b in zip(results[0]['alphas'], results[1]['alphas']):
        if a['converged'] and b['converged']:
            reduction = 100*(1-b['drag_area_mm']/a['drag_area_mm'])
            comparisons.append(dict(alpha=a['alpha'], cd_change_pct=100*(b['cd']/a['cd']-1),
                drag_area_reduction_pct=reduction, meets_3pct_at_this_alpha=reduction >= 3))
        else:
            comparisons.append(dict(alpha=a['alpha'], status='missing convergence; no comparison'))
    report = dict(scope='exactly 2 geometries x 3 alpha, single 180-panel setting',
                  cases=results, comparison=comparisons,
                  limitation='No Gaussian crosswind integration or panel convergence study; no whole-aircraft claim.')
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
