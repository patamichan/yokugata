"""One CST section: local polynomial clearance fit, analytic curvature audit.

Run: python -m tests.benchmarks.benchmark_b_local
No XFOIL, DB modification, training, or production-generator replacement.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.optimize import brentq
from scipy.special import comb
from core.generators.funcgen_cst import _weights
from core.optimizer import PRESET_PILOT_STANDARD, _profile_arrays
from core.paths import RESEARCH_DIR

L = .2
CHORD = 2187.935
BASE_CHORD = 2327.271


def base_polynomial():
    # q=sqrt(x/L) is regular at the nose; the same polynomial can be sampled
    # beyond q=1 to represent the unchanged aft section.
    q = P([0., 1.])
    x = L*q*q
    weights = _weights(5, .24, .42, 1.2)
    shape = sum(a*comb(5, i)*x**i*(1-x)**(5-i) for i, a in enumerate(weights))
    return np.sqrt(L)*q*(1-x)*shape


def curvature(poly, q):
    xp = 2*L*q
    yp, ypp = poly.deriv(1)(q), poly.deriv(2)(q)
    return (xp*ypp-2*L*yp)/(xp*xp+yp*yp)**1.5


def curvature_slope_numerator(poly):
    q = P([0., 1.])
    xp = 2*L*q
    yp, ypp, yppp = poly.deriv(1), poly.deriv(2), poly.deriv(3)
    numerator = xp*ypp-2*L*yp
    denominator = xp*xp+yp*yp
    return (xp*yppp)*denominator-3*numerator*(xp*(2*L)+yp*ypp)


def sign_change_roots(func, grid):
    values = func(grid)
    roots = []
    for i in np.flatnonzero(values[:-1]*values[1:] < 0):
        root = brentq(func, grid[i], grid[i+1], xtol=1e-13)
        if not roots or abs(root-roots[-1]) > 1e-8:
            roots.append(root)
    return roots


def solve_local(base):
    # Cubic h(u); exact linear equality for the foot width. Minimize the
    # squared q-second-derivative of the correction, not aerodynamic curvature.
    q = P([0., 1.])
    carrier = L**1.5*q**3*(1-q*q)**4
    basis = [carrier*q**(2*j) for j in range(4)]
    qs = np.linspace(0., 1., 2001)
    d2 = np.column_stack([b.deriv(2)(qs) for b in basis])
    scales = np.linalg.norm(d2, axis=0)
    matrix = d2/scales
    gram = matrix.T @ matrix / len(qs) + 1e-10*np.eye(4)
    q_foot = np.sqrt((130./CHORD)/L)
    target = 180./CHORD-base(q_foot)
    constraint = np.array([b(q_foot) for b in basis])/scales
    direction = np.linalg.solve(gram, constraint)
    coefficients = target*direction/(constraint @ direction)/scales
    correction = sum(c*b for c, b in zip(coefficients, basis))
    return correction, coefficients


def benchmark(samples=16001):
    start = time.perf_counter()
    base = base_polynomial()
    correction, coefficients = solve_local(base)
    edited = base+correction
    qs = np.linspace(0., 1., samples)
    xn = L*qs**2
    kb, ke = curvature(base, qs), curvature(edited, qs)
    full_x = np.sin(np.linspace(0, np.pi/2, samples))**2
    full_q = np.sqrt(full_x/L)
    yb = base(full_q)
    ye = yb.copy()
    forward = full_x < L
    ye[forward] += correction(full_q[forward])
    def perimeter(y, chord):
        return float(2*chord*np.hypot(np.diff(full_x), np.diff(y)).sum())
    distances, widths = _profile_arrays(PRESET_PILOT_STANDARD, 30.)
    stations = distances/CHORD
    station_q = np.sqrt(stations/L)
    available = base(station_q)
    available[stations < L] += correction(station_q[stations < L])
    margins = 2*CHORD*available-widths
    base_margins_short = 2*CHORD*base(station_q)-widths
    def quality(poly, k):
        inflections = sign_change_roots(lambda q: 2*L*q*poly.deriv(2)(q)-2*L*poly.deriv(1)(q), qs)
        extrema = sign_change_roots(curvature_slope_numerator(poly), qs[1:])
        return dict(curvature_min=float(k.min()), curvature_max=float(k.max()),
                    total_variation=float(np.abs(np.diff(k)).sum()),
                    inflection_x=[float(L*q*q) for q in inflections],
                    curvature_extremum_x=[float(L*q*q) for q in extrema])
    max_delta = int(np.argmax(np.abs(ke-kb)))
    baseline_perimeter = perimeter(yb, BASE_CHORD)
    edited_perimeter = perimeter(ye, CHORD)
    report = dict(scope='one geometry case, no aerodynamic inference', samples=samples,
        tc_pct=24., xt_pct=42., kle=1.2, support_L=L,
        chord_base_mm=BASE_CHORD, chord_edited_mm=CHORD,
        h_degree=3, h_power_coefficients=coefficients.tolist(),
        objective='squared correction second derivative in q=sqrt(x/L), with 1e-10 ridge',
        foot_x_normalized=130./CHORD,
        base_short_chord_full_width_margins_mm=base_margins_short.tolist(),
        edited_full_width_margins_mm=margins.tolist(),
        nose_radius_normalized=float(1.1019*.24**2*1.2),
        join_correction_derivatives_q=[float(correction.deriv(i)(1.)) for i in range(4)],
        minimum_sampled_half_thickness=float(ye.min()),
        maximum_sampled_full_thickness_pct=float(200*ye.max()),
        base_forward_curvature=quality(base, kb), edited_forward_curvature=quality(edited, ke),
        max_normalized_curvature_delta=float(abs(ke[max_delta]-kb[max_delta])),
        max_delta_x_normalized=float(xn[max_delta]),
        max_physical_curvature_delta_per_mm=float(np.max(np.abs(ke/CHORD-kb/BASE_CHORD))),
        base_perimeter_mm=baseline_perimeter, edited_perimeter_mm=edited_perimeter,
        perimeter_reduction_pct=100*(1-edited_perimeter/baseline_perimeter),
        max_half_width_edit_mm=float(CHORD*correction(qs).max()),
        elapsed_seconds=time.perf_counter()-start)
    curves = np.column_stack((xn, base(qs), edited(qs), kb, ke, kb/BASE_CHORD, ke/CHORD))
    return report, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=RESEARCH_DIR/'results_b_local')
    args = parser.parse_args()
    report, curves = benchmark()
    coarse, _ = benchmark(8001)
    report['sampling_check_8001_vs_16001'] = dict(
        perimeter_difference_mm=abs(coarse['edited_perimeter_mm']-report['edited_perimeter_mm']),
        max_curvature_delta_difference=abs(coarse['max_normalized_curvature_delta']-report['max_normalized_curvature_delta']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (args.output_dir/'curvature.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x_normalized','base_half_thickness','edited_half_thickness',
                         'base_curvature_normalized','edited_curvature_normalized',
                         'base_curvature_per_mm','edited_curvature_per_mm'])
        writer.writerows(curves)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
