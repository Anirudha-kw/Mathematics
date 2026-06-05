"""Numerical differentiation helpers.

These utilities provide high-accuracy finite-difference derivatives of
vector-valued functions ``f: R -> R^n`` and ``f: R^2 -> R^n``.  They are the
numerical backbone for the curve and surface modules so that a user only has to
supply a parametrisation, not its analytic derivatives.

A central, 5-point stencil is used wherever possible which gives an error of
order ``O(h^4)`` for first derivatives, a good compromise between truncation
error (small ``h`` is better) and round-off error (large ``h`` is better).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ArrayLike = np.ndarray

# A reasonable default step.  For double precision the optimal step for a
# central difference scales like eps**(1/3) ~ 6e-6; we round up a little to stay
# clear of catastrophic cancellation for typical functions.
DEFAULT_STEP = 1e-5


def derivative(f: Callable[[float], ArrayLike], t: float, h: float = DEFAULT_STEP) -> np.ndarray:
    """First derivative of ``f`` at ``t`` via a 5-point central stencil.

    Parameters
    ----------
    f : callable
        Function returning a scalar or 1-D ``numpy`` array.
    t : float
        Point at which to differentiate.
    h : float, optional
        Step size.

    Returns
    -------
    numpy.ndarray
        Approximation of ``f'(t)`` with error ``O(h^4)``.
    """
    f1 = np.asarray(f(t - 2 * h), dtype=float)
    f2 = np.asarray(f(t - h), dtype=float)
    f3 = np.asarray(f(t + h), dtype=float)
    f4 = np.asarray(f(t + 2 * h), dtype=float)
    return (f1 - 8 * f2 + 8 * f3 - f4) / (12 * h)


def second_derivative(f: Callable[[float], ArrayLike], t: float, h: float = DEFAULT_STEP) -> np.ndarray:
    """Second derivative of ``f`` at ``t`` via a 5-point central stencil."""
    f0 = np.asarray(f(t - 2 * h), dtype=float)
    f1 = np.asarray(f(t - h), dtype=float)
    f2 = np.asarray(f(t), dtype=float)
    f3 = np.asarray(f(t + h), dtype=float)
    f4 = np.asarray(f(t + 2 * h), dtype=float)
    return (-f0 + 16 * f1 - 30 * f2 + 16 * f3 - f4) / (12 * h * h)


def third_derivative(f: Callable[[float], ArrayLike], t: float, h: float = DEFAULT_STEP) -> np.ndarray:
    """Third derivative of ``f`` at ``t`` via a central stencil."""
    f0 = np.asarray(f(t - 2 * h), dtype=float)
    f1 = np.asarray(f(t - h), dtype=float)
    f3 = np.asarray(f(t + h), dtype=float)
    f4 = np.asarray(f(t + 2 * h), dtype=float)
    return (-f0 + 2 * f1 - 2 * f3 + f4) / (2 * h ** 3)


def partial(
    f: Callable[[float, float], ArrayLike],
    u: float,
    v: float,
    axis: int,
    h: float = DEFAULT_STEP,
) -> np.ndarray:
    """First partial derivative of ``f(u, v)`` along ``axis`` (0 = u, 1 = v)."""
    if axis == 0:
        g = lambda x: f(x, v)  # noqa: E731 - small local lambda is clearest here
        return derivative(g, u, h)
    if axis == 1:
        g = lambda y: f(u, y)  # noqa: E731
        return derivative(g, v, h)
    raise ValueError("axis must be 0 (u) or 1 (v)")


def second_partial(
    f: Callable[[float, float], ArrayLike],
    u: float,
    v: float,
    axis_a: int,
    axis_b: int,
    h: float = DEFAULT_STEP,
) -> np.ndarray:
    """Second partial derivative of ``f(u, v)`` along ``axis_a`` then ``axis_b``.

    Pure second derivatives (``axis_a == axis_b``) use a 1-D stencil; mixed
    derivatives use the standard 4-point central scheme.
    """
    if axis_a == axis_b:
        if axis_a == 0:
            return second_derivative(lambda x: f(x, v), u, h)
        return second_derivative(lambda y: f(u, y), v, h)

    # Mixed partial d^2f / du dv.
    fpp = np.asarray(f(u + h, v + h), dtype=float)
    fpm = np.asarray(f(u + h, v - h), dtype=float)
    fmp = np.asarray(f(u - h, v + h), dtype=float)
    fmm = np.asarray(f(u - h, v - h), dtype=float)
    return (fpp - fpm - fmp + fmm) / (4 * h * h)
