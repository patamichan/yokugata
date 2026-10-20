"""Envelope feasibility and section-level selection at physical Reynolds numbers."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.geometry import generate_funcgen_airfoil
from core.optimizer import (calculate_required_chord, calculate_expected_cd,
                            optimize_section_3d)


PROFILE = [dict(x_mm=100., width_mm=320., clearance_mm=15.),
           dict(x_mm=800., width_mm=450., clearance_mm=20.)]


def test_chord_encloses_every_point_and_is_minimal():
    chord = calculate_required_chord(25., 42., 1., PROFILE, tolerance_mm=.1)
    x, y = generate_funcgen_airfoil(25., 42., 1., num_points=2001)
    nose = np.argmin(x)
    curve = PchipInterpolator(np.sqrt(x[:nose+1][::-1]), y[:nose+1][::-1])
    distances = np.array([p['x_mm'] + 30 for p in PROFILE])
    required = np.array([p['width_mm'] + 2*p['clearance_mm'] for p in PROFILE])
    def widths(c):
        return 2*c*curve(np.sqrt(distances/c))
    assert np.all(widths(chord) >= required)
    assert np.any(widths(chord-.11) < required)
    with pytest.raises(ValueError, match='closed nose'):
        calculate_required_chord(25., 42., 1., [dict(x_mm=0., width_mm=1.)], nose_margin_mm=0.)


def test_literal_gaussian_mass_and_missing_polar_rejection():
    assert calculate_expected_cd(lambda a: .01) == pytest.approx(
        .01 * math.erf(2.807034 / math.sqrt(2)), abs=1e-10)
    with pytest.raises(ValueError, match='exceeds'):
        calculate_expected_cd(lambda a: .01, crosswind_ms=10.)
    with pytest.raises(ValueError, match='finite positive'):
        calculate_expected_cd(lambda a: float('nan'))


def make_database():
    rows = []
    for name, kle, cd in [('base', 1., .02), ('blunt', 1.3, .01)]:
        for re in (500000, 10000000):
            for alpha in (0., 3., 6., 9., 12.):
                rows.append(dict(airfoil=name, thickness_pct=25., max_thickness_pos_pct=42.,
                                 le_radius_ratio=kle, reynolds=re, alpha=alpha, cd=cd))
    return SimpleNamespace(df=pd.DataFrame(rows),
                           get_completed_keys=lambda: {(25., 42., 1.), (25., 42., 1.3)})


def test_full_section_selection_and_unsupported_constraints():
    db = make_database()
    result = optimize_section_3d(db, PROFILE)
    assert result['airfoil'] == 'blunt'
    assert result['drag_area_mm'] == pytest.approx(result['chord_mm']*result['cd_eff'])
    assert result['cd_eff'] == pytest.approx(.00995, abs=1e-7)
    assert optimize_section_3d(db, PROFILE, max_chord_mm=10.) == {}
    db.df['reynolds'] += 100000000
    result = optimize_section_3d(db, PROFILE)
    assert result['airfoil'] == 'blunt'
    assert result['reynolds_clamped']
    assert result['reynolds_evaluated'] == db.df.reynolds.min()
