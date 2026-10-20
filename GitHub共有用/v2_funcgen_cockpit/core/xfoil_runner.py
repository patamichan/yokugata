"""Isolated XFOIL polar runs for the three-parameter geometry family."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy as np

from core.geometry import generate_funcgen_airfoil, save_airfoil_dat
from core.paths import DATABASE_DIR, LEGACY_DIR

BASE_DIR = Path(__file__).resolve().parent.parent


def format_foil_name_3d(tc_pct: float, xt_pct: float, kle: float) -> str:
    if not all(math.isfinite(v) for v in (tc_pct, xt_pct, kle)):
        raise ValueError("Airfoil parameters must be finite")
    ratio = format(kle, '.12g')
    if '.' not in ratio and 'e' not in ratio:
        ratio += '.0'
    return f"NACA 00{tc_pct:.12g}_{xt_pct:.12g}_k{ratio}"


def run_single_xfoil_case(dat_path, reynolds, alpha_min=0.0, alpha_max=12.0,
                          alpha_step=0.1, timeout_sec=30, xfoil_exe=None,
                          panel_count=None, ncrit=None):
    values = (reynolds, alpha_min, alpha_max, alpha_step, timeout_sec)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("XFOIL parameters must be finite")
    if reynolds <= 0 or timeout_sec <= 0 or alpha_step == 0:
        raise ValueError("Positive Reynolds number and timeout, and nonzero step required")
    if (alpha_max - alpha_min) * alpha_step < 0:
        raise ValueError("Alpha step must point toward alpha_max")
    if panel_count is not None and (isinstance(panel_count, bool) or not isinstance(panel_count, int) or not 80 <= panel_count <= 300):
        raise ValueError('panel_count must be an integer in [80, 300]')
    if ncrit is not None and (not math.isfinite(ncrit) or ncrit <= 0):
        raise ValueError('ncrit must be finite and positive')
    source = Path(dat_path).resolve(strict=True)
    executable = Path(xfoil_exe).resolve(strict=True) if xfoil_exe else BASE_DIR / 'xfoil.exe'
    if not executable.is_file():
        executable = LEGACY_DIR / 'xfoil.exe'
    if not executable.is_file():
        raise FileNotFoundError("XFOIL executable not found")
    with tempfile.TemporaryDirectory(prefix='cockpit_xfoil_') as work:
        shutil.copyfile(source, Path(work) / 'airfoil.dat')
        commands = ['plop', 'g', '', 'load airfoil.dat']
        if panel_count is not None:
            commands += ['ppar', f'n {panel_count}', '', '']
        commands += ['pane', 'oper', 'iter 100', f'visc {reynolds:.12g}']
        if ncrit is not None:
            commands += ['vpar', f'n {ncrit:.12g}', '']
        commands += ['pacc', 'output.pol', '',
                    f'aseq {alpha_min:.12g} {alpha_max:.12g} {alpha_step:.12g}',
                    'pacc', '', 'quit', '']
        try:
            result = subprocess.run([str(executable)], input='\n'.join(commands),
                                    text=True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, cwd=work,
                                    timeout=timeout_sec,
                                    creationflags=0x08000000 if sys.platform == 'win32' else 0)
            if result.returncode:
                warnings.warn(f'XFOIL exited with status {result.returncode} at Re={reynolds}')
        except subprocess.TimeoutExpired:
            # subprocess.run kills and waits for the child; retain completed polar rows.
            warnings.warn(f'XFOIL timed out at Re={reynolds}; returning partial polar')
        polar = Path(work) / 'output.pol'
        if not polar.exists():
            return []
        records = []
        for line in polar.read_text(encoding='utf-8', errors='replace').splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                alpha, cl, cd, cdp, cm, top, bottom = map(float, parts[:7])
            except ValueError:
                continue
            if not all(math.isfinite(v) for v in (alpha, cl, cd, cm, top, bottom)) or cd <= 0:
                continue
            records.append(dict(reynolds=reynolds, alpha=alpha, cl=cl, cd=cd,
                                drag_counts=cd*10000, ld=cl/cd, cm=cm,
                                top_xtr=top, bot_xtr=bottom, converged=True))
        return records


def analyze_funcgen_airfoil_all_re(tc_pct, xt_pct, kle, re_list, alpha_min=0.0,
                                  alpha_max=12.0, alpha_step=0.1, max_workers=8,
                                  airfoils_dir=None, xfoil_exe=None, progress_callback=None,
                                  generator_method='warp', records_callback=None):
    """Report each completed Re on the calling thread as callback(done, total).

    Duplicate Re values count once. Return rows in input-Re order regardless
    of completion order. Callback exceptions propagate to the caller.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError('progress_callback must be callable or None')
    if records_callback is not None and not callable(records_callback):
        raise TypeError('records_callback must be callable or None')
    name = format_foil_name_3d(tc_pct, xt_pct, kle)
    if generator_method not in ('warp', 'cst'):
        raise ValueError('generator_method must be warp or cst')
    if generator_method != 'warp':
        name += f'_{generator_method}'
    # Validate geometry even when a previously generated file exists.
    x, y = generate_funcgen_airfoil(tc_pct, xt_pct, kle, method=generator_method)
    directory = Path(airfoils_dir) if airfoils_dir else DATABASE_DIR / 'airfoils'
    directory.mkdir(parents=True, exist_ok=True)
    dat = directory / f'{name}.dat'
    if dat.exists():
        previous = np.loadtxt(dat, skiprows=1)
        current = np.column_stack((x, y))
        if previous.shape != current.shape or not np.allclose(previous, current, atol=1e-11, rtol=0):
            raise ValueError(f'Existing airfoil coordinates differ from the current generator: {dat}')
    if not dat.exists():
        fd, temporary = tempfile.mkstemp(dir=directory, suffix='.dat')
        os.close(fd)
        try:
            save_airfoil_dat(temporary, x, y, name)
            os.replace(temporary, dat)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    def run(re):
        return run_single_xfoil_case(dat, re, alpha_min, alpha_max, alpha_step,
                                     xfoil_exe=xfoil_exe)
    records = []
    reynolds_values = list(dict.fromkeys(re_list))
    completed = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run, re): index for index, re in enumerate(reynolds_values)}
        def collect(future):
            rows = future.result()
            for row in rows:
                row.update(airfoil=name, thickness_pct=round(tc_pct, 2),
                           max_thickness_pos_pct=round(xt_pct, 2),
                           le_radius_ratio=round(kle, 3), max_camber_pct=0.0,
                           generator_method=generator_method)
            completed[futures[future]] = rows
            if records_callback is not None:
                records_callback(rows)
            if progress_callback is not None:
                progress_callback(len(completed), len(futures))
        try:
            for future in as_completed(futures):
                collect(future)
        except KeyboardInterrupt:
            # Cancel queued work, then preserve results of already running cases.
            for future in futures:
                future.cancel()
            for future in futures:
                if not future.cancelled() and futures[future] not in completed:
                    collect(future)
            raise
    for index in range(len(reynolds_values)):
        records.extend(completed[index])
    return name, records
