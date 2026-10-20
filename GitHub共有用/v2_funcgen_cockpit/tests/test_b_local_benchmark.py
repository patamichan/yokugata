"""Numerical contract checks for the single-case research benchmark."""
import numpy as np
import pytest
from tests.benchmarks.benchmark_b_local import (
    base_polynomial, solve_local, curvature, benchmark, L)


def test_local_clearance_join_and_nose_contract():
    base = base_polynomial()
    delta, coefficients = solve_local(base)
    report, curves = benchmark(4001)
    assert abs(report['edited_full_width_margins_mm'][0]) < 1e-8
    assert min(report['edited_full_width_margins_mm']) >= -1e-8
    assert report['maximum_sampled_full_thickness_pct'] <= 24.+1e-7
    assert np.isfinite(coefficients).all()
    assert delta(0.) == 0.
    assert delta.deriv(1)(0.) == 0.
    assert curvature(base+delta, np.array([0.]))[0] == pytest.approx(
        -1/report['nose_radius_normalized'], abs=1e-10)
    assert max(abs(delta.deriv(i)(1.)) for i in range(4)) < 1e-9
    assert not report['edited_forward_curvature']['inflection_x']
    assert len(report['edited_forward_curvature']['curvature_extremum_x']) == 2
    assert report['perimeter_reduction_pct'] > 0
    # Report curves use matching x, but each actual chord for dimensional curvature.
    np.testing.assert_allclose(curves[:, 6], curves[:, 4]/report['chord_edited_mm'])
