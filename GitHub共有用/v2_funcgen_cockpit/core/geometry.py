"""Symmetric, closed airfoils with independent thickness, high point and nose radius.

Coordinates use a unit chord and Selig ordering. Only the forward 20 percent
of the chord is changed by the compact leading-edge deformation.
"""
from typing import Tuple

import numpy as np
from scipy.interpolate import BPoly, CubicSpline
from scipy.optimize import brentq, minimize_scalar


def _naca(x):
    # Factoring the polynomial enforces trailing-edge closure without cancellation.
    return 0.2969 * np.sqrt(x) + x * (-0.1260 + x * (-0.3516 + x * (0.2843 - 0.1036 * x)))


_PEAK = brentq(lambda x: 0.14845 / np.sqrt(x) - 0.1260 - 0.7032*x
               + 0.8529*x*x - 0.4144*x*x*x, 0.2, 0.4)
_HEIGHT = float(_naca(_PEAK))
_NOSE_COEFFICIENT = np.sqrt(2.0 * 1.1019)


def cosine_projection_x(alpha: np.ndarray) -> np.ndarray:
    """Return the circular cosine projection, evaluated stably near the nose."""
    alpha = np.asarray(alpha, dtype=float)
    return np.sin(np.pi * (alpha - 0.5)) ** 2


def generate_funcgen_airfoil(tc_pct: float, xt_pct: float, kle: float = 1.0,
                             num_points: int = 161, method: str = "warp") -> Tuple[np.ndarray, np.ndarray]:
    """Generate unit-chord coordinates from upper TE through LE to lower TE.

    Supports multiple underlying generators via method parameter:
    - 'warp': Original C2 monotone coordinate warp (FuncGenWarpGenerator)
    - 'cst': Equality-constrained Class-Shape Transformation (FuncGenCSTGenerator)
    """
    if method != "warp":
        from core.generators import get_generator
        gen = get_generator(method)
        return gen.generate(tc_pct, xt_pct, kle, num_points)

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


def _coordinates(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or x.size < 3:
        raise ValueError("x and y must be matching one-dimensional arrays with at least three points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Coordinates must be finite")
    return x, y


def save_airfoil_dat(filepath: str, x: np.ndarray, y: np.ndarray,
                     name: str = "FuncGenAirfoil"):
    """Write a named, two-column Selig file with sufficient precision for reuse."""
    x, y = _coordinates(x, y)
    if not isinstance(name, str) or not name.strip() or '\n' in name or '\r' in name:
        raise ValueError("name must be a nonempty single line")
    np.savetxt(filepath, np.column_stack((x, y)), fmt="%.12f", header=name, comments="", encoding="utf-8")


def calculate_le_radius(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares fit of x^2+y^2=2*r*x over 0<x<=0.02.

    This is an effective finite-window circle radius, rather than an
    extrapolation of the exact limiting curvature at x=0.
    """
    x, y = _coordinates(x, y)
    mask = (x > 0.0) & (x <= 0.02)
    if not np.any(mask):
        raise ValueError("At least one nonzero nose point with x <= 0.02 is required")
    xn, yn = x[mask], y[mask]
    # Scale both coordinates to avoid squaring very small dimensional values.
    s = max(float(np.max(xn)), float(np.max(np.abs(yn))))
    a, b = xn / s, yn / s
    return float(s * np.dot(a, a*a + b*b) / (2.0 * np.dot(a, a)))


def calculate_max_thickness(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Measure full thickness and its position (percent) using surface splines."""
    x, y = _coordinates(x, y)
    nose = int(np.argmin(x))
    if nose == 0 or nose == len(x)-1:
        raise ValueError("Expected Selig ordering with an interior leading edge")
    xu, yu = x[:nose+1][::-1], y[:nose+1][::-1]
    xl, yl = x[nose:], y[nose:]
    if np.any(np.diff(xu) <= 0) or np.any(np.diff(xl) <= 0):
        raise ValueError("Each surface must have strictly monotone x coordinates")
    upper, lower = CubicSpline(xu, yu), CubicSpline(xl, yl)
    thickness = lambda z: upper(z) - lower(z)
    # Search every spline interval: avoids assuming a unimodal input profile.
    knots = np.unique(np.concatenate((xu, xl)))
    candidates = [(float(thickness(z)), float(z)) for z in knots]
    for left, right in zip(knots[:-1], knots[1:]):
        result = minimize_scalar(lambda z: -float(thickness(z)), bounds=(left, right), method="bounded",
                                 options={"xatol": 1e-13})
        candidates.append((-float(result.fun), float(result.x)))
    height, location = max(candidates)
    return 100.0 * height, 100.0 * location


