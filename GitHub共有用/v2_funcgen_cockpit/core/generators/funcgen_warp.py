"""
FuncGenFoil Airfoil Generator with C2 Monotone Coordinate Warping
Original implementation by Codex based on Task 01 specification.
Uses circular cosine projection and compact window leading-edge scaling.
"""

from typing import Tuple
import numpy as np
from scipy.interpolate import BPoly
from scipy.optimize import brentq
from core.generators.base import BaseAirfoilGenerator


def _naca(x):
    return 0.2969 * np.sqrt(x) + x * (-0.1260 + x * (-0.3516 + x * (0.2843 - 0.1036 * x)))


_PEAK = brentq(lambda x: 0.14845 / np.sqrt(x) - 0.1260 - 0.7032*x + 0.8529*x*x - 0.4144*x*x*x, 0.2, 0.4)
_HEIGHT = float(_naca(_PEAK))
_NOSE_COEFFICIENT = np.sqrt(2.0 * 1.1019)


def cosine_projection_x(alpha: np.ndarray) -> np.ndarray:
    """Return circular cosine projection: x(alpha) = sin^2(pi*(alpha - 0.5))"""
    alpha = np.asarray(alpha, dtype=float)
    return np.sin(np.pi * (alpha - 0.5)) ** 2


class FuncGenWarpGenerator(BaseAirfoilGenerator):
    """Airfoil generator using C2 coordinate warping and compact polynomial window."""

    def generate(
        self,
        tc_pct: float,
        xt_pct: float,
        kle: float = 1.0,
        num_points: int = 161
    ) -> Tuple[np.ndarray, np.ndarray]:
        parameters = np.asarray([tc_pct, xt_pct, kle], dtype=float)
        if not np.all(np.isfinite(parameters)):
            raise ValueError("Design parameters must be finite")
        if not 20.0 <= tc_pct <= 30.0 or not 38.0 <= xt_pct <= 48.0:
            raise ValueError("tc_pct must be in [20, 30] and xt_pct in [38, 48]")
        if not 0.6 <= kle <= 1.5:
            raise ValueError("kle must be in [0.6, 1.5]")
        if isinstance(num_points, (bool, np.bool_)) or not isinstance(num_points, (int, np.integer)) or num_points < 5:
            raise ValueError("num_points must be an integer of at least 5")

        tc, xt = tc_pct / 100.0, xt_pct / 100.0
        n_upper = num_points // 2 + 1
        upper_x = cosine_projection_x(np.linspace(0.0, 0.5, n_upper))
        if num_points % 2:
            lower_x = upper_x[-2::-1].copy()
        else:
            lower_x = cosine_projection_x(np.linspace(0.5, 1.0, num_points - n_upper + 1))[1:]
        x = np.concatenate((upper_x, lower_x))
        x[0] = x[-1] = 1.0
        x[n_upper - 1] = 0.0

        scale = 0.5 * tc / _HEIGHT
        nose_slope = (_NOSE_COEFFICIENT * tc / (scale * 0.2969)) ** 2
        warp = BPoly.from_derivatives([0.0, xt, 1.0],
                                     [[0.0, nose_slope, 0.0], [_PEAK, 1.0, 0.0], [1.0, 1.0, 0.0]])
        xp = np.clip(warp(x), 0.0, 1.0)
        half = scale * _naca(xp)

        forward = x < 0.2
        u = x[forward] / 0.2
        window = (1.0 - u)**4
        half[forward] += (np.sqrt(kle) - 1.0) * _NOSE_COEFFICIENT * tc * np.sqrt(x[forward]) * window

        y = half.copy()
        y[n_upper:] *= -1.0
        y[[0, n_upper - 1, num_points - 1]] = 0.0
        return x, y
