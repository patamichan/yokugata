"""Audit six geometry cases only; no XFOIL, model training, or DB writes.

Run from the project root: python -m tests.benchmarks.audit_local_vs_global
Surface perimeter is per unit extrusion height, not a 3D wetted-area estimate.
"""
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from core.geometry import generate_funcgen_airfoil
from core.optimizer import calculate_required_chord, PRESET_PILOT_STANDARD, _profile_arrays


def audit():
    tc, xt = 24., 42.
    results = []
    for method in ('warp', 'cst'):
        for kle in (.8, 1., 1.2):
            full = calculate_required_chord(tc, xt, kle, PRESET_PILOT_STANDARD,
                                             generator_method=method)
            remaining = PRESET_PILOT_STANDARD[1:]
            no_feet = calculate_required_chord(tc, xt, kle, remaining, generator_method=method)
            shoulder = calculate_required_chord(tc, xt, kle, remaining[-1:], generator_method=method)
            x, y = generate_funcgen_airfoil(tc, xt, kle, num_points=4001, method=method)
            nose = np.argmin(x)
            surface = PchipInterpolator(np.sqrt(x[:nose+1][::-1]), y[:nose+1][::-1])
            distances, widths = _profile_arrays(remaining, 30.)
            margins = 2*no_feet*surface(np.sqrt(distances/no_feet))-widths
            perimeter = float(np.hypot(np.diff(x), np.diff(y)).sum())
            results.append(dict(method=method, tc_pct=tc, xt_pct=xt, kle=kle,
                chord_full_mm=full, chord_no_feet_mm=no_feet, chord_shoulder_only_mm=shoulder,
                remaining_full_width_margins_mm=margins.tolist(),
                normalized_perimeter=perimeter, perimeter_full_mm=full*perimeter,
                chord_reduction_pct=100*(1-no_feet/full)))
    comparisons = []
    for method in ('warp', 'cst'):
        old = next(r for r in results if r['method'] == method and r['kle'] == 1.)
        new = next(r for r in results if r['method'] == method and r['kle'] == 1.2)
        comparisons.append(dict(method=method,
            chord_reduction_pct=100*(1-new['chord_full_mm']/old['chord_full_mm']),
            perimeter_reduction_pct=100*(1-new['perimeter_full_mm']/old['perimeter_full_mm'])))
    return dict(scope='six geometry cases; fixed tc=24%, xt=42%; no aerodynamic claims',
                cases=results, kle_1_to_1p2=comparisons)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
