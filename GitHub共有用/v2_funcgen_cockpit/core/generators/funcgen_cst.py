"""CST geometry with a fixed nose coefficient and exact peak constraints.

Fairness penalizes second differences, not aerodynamic drag or inflection count.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb
from core.generators.base import BaseAirfoilGenerator


def bernstein_poly(i, n, x):
    return comb(n, i) * x**i * (1.0-x)**(n-i)


def _basis(n, x):
    x = np.asarray(x)
    return np.stack([np.sqrt(x)*(1-x)*bernstein_poly(i, n, x) for i in range(n+1)], axis=-1)


def _slope_basis(n, x):
    c = np.sqrt(x)*(1-x)
    dc = (1-3*x)/(2*np.sqrt(x))
    result = []
    for i in range(n+1):
        left = bernstein_poly(i-1, n-1, x) if i > 0 else 0.
        right = bernstein_poly(i, n-1, x) if i < n else 0.
        result.append(dc*bernstein_poly(i, n, x) + c*n*(left-right))
    return np.array(result)


@lru_cache(maxsize=256)
def _weights(n, tc, xt, kle):
    a0 = np.sqrt(2*1.1019*tc*tc*kle)
    peak = np.stack([_basis(n, xt), _slope_basis(n, xt)])
    matrix = peak[:, 1:]
    target = np.array([tc/2, 0.]) - peak[:, 0]*a0
    samples = _basis(n, np.linspace(.01, .99, 100))
    differences = np.diff(samples, n=2, axis=0)
    d0, d = differences[:, 0]*a0, differences[:, 1:]
    scale = 1e6
    def objective(free):
        residual = d0 + d @ free
        return scale * np.dot(residual, residual)
    def gradient(free):
        return 2*scale * d.T @ (d0 + d @ free)
    initial = np.full(n, tc*.8)
    initial += matrix.T @ np.linalg.solve(matrix @ matrix.T, target-matrix @ initial)
    result = minimize(objective, initial, jac=gradient, method='SLSQP',
                      constraints=[{'type': 'eq', 'fun': lambda a: matrix @ a-target,
                                    'jac': lambda a: matrix}],
                      options={'maxiter': 500, 'ftol': 1e-12})
    if (not result.success or not np.isfinite(result.x).all()
            or np.max(np.abs(matrix @ result.x-target)) > 1e-9):
        raise RuntimeError(f'CST optimization failed to converge: {result.message}')
    return tuple(np.concatenate(([a0], result.x)))


class FuncGenCSTGenerator(BaseAirfoilGenerator):
    def __init__(self, n_order=5):
        if isinstance(n_order, bool) or not isinstance(n_order, (int, np.integer)) or n_order < 2:
            raise ValueError('CST order must be an integer of at least 2')
        self.n_order = int(n_order)

    def generate(self, tc_pct, xt_pct, kle=1.0, num_points=161):
        if not np.isfinite([tc_pct, xt_pct, kle]).all():
            raise ValueError('Design parameters must be finite')
        if not 20 <= tc_pct <= 30 or not 38 <= xt_pct <= 48 or not .6 <= kle <= 1.5:
            raise ValueError('Design parameters outside supported range')
        if (isinstance(num_points, (bool, np.bool_)) or
                not isinstance(num_points, (int, np.integer)) or num_points < 5):
            raise ValueError('num_points must be an integer of at least 5')
        weights = np.asarray(_weights(self.n_order, tc_pct/100, xt_pct/100, kle))
        n_upper = num_points//2+1
        xu = np.sin(np.pi*(np.linspace(0., .5, n_upper)-.5))**2
        xu[0], xu[-1] = 1., 0.
        yu = _basis(self.n_order, xu) @ weights
        if num_points % 2:
            xl, yl = xu[-2::-1].copy(), -yu[-2::-1].copy()
        else:
            xl = np.sin(np.pi*(np.linspace(.5, 1., num_points-n_upper+1)[1:]-.5))**2
            xl[-1] = 1.
            yl = -_basis(self.n_order, xl) @ weights
        x, y = np.concatenate((xu, xl)), np.concatenate((yu, yl))
        y[[0, n_upper-1, num_points-1]] = 0.
        return x, y
