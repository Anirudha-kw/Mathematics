"""Differential geometry of parametric curves in 3-D.

The central object is :class:`Curve`, which wraps a parametrisation
``gamma: R -> R^3`` and exposes the classical local theory of curves: the
Frenet-Serret frame (tangent ``T``, normal ``N``, binormal ``B``), curvature,
torsion and arc length.

These quantities are the bread and butter of robot path planning: curvature
bounds translate directly into steering / turning-radius constraints, and the
Frenet frame is the natural moving coordinate system for trajectory tracking.

Example
-------
>>> import numpy as np
>>> from differential_geometry.curves import Curve
>>> helix = Curve(lambda t: np.array([np.cos(t), np.sin(t), 0.5 * t]))
>>> round(helix.curvature(0.0), 6)
0.8
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import numdiff


class Curve:
    """A smooth parametric curve ``gamma(t)`` in ``R^3``.

    Parameters
    ----------
    gamma : callable
        Maps a scalar parameter ``t`` to a point in ``R^3`` (a length-3
        array-like).  2-D curves can be supplied by padding the third
        coordinate with zeros.
    h : float, optional
        Finite-difference step for first/second derivatives.
    h_high : float, optional
        Larger step used for the third derivative (jerk).  Higher-order
        stencils divide by ``h**3``, so a step as small as ``h`` would be
        swamped by floating-point round-off; ``~1e-3`` is near optimal.
    """

    def __init__(
        self,
        gamma: Callable[[float], np.ndarray],
        h: float = numdiff.DEFAULT_STEP,
        h_high: float = 1e-3,
    ):
        self.gamma = gamma
        self.h = h
        self.h_high = h_high

    # -- raw derivatives ---------------------------------------------------
    def point(self, t: float) -> np.ndarray:
        """Position ``gamma(t)``."""
        return np.asarray(self.gamma(t), dtype=float)

    def velocity(self, t: float) -> np.ndarray:
        """First derivative ``gamma'(t)`` (the velocity vector)."""
        return numdiff.derivative(self.gamma, t, self.h)

    def acceleration(self, t: float) -> np.ndarray:
        """Second derivative ``gamma''(t)``."""
        return numdiff.second_derivative(self.gamma, t, self.h)

    def jerk(self, t: float) -> np.ndarray:
        """Third derivative ``gamma'''(t)`` (needed for torsion)."""
        return numdiff.third_derivative(self.gamma, t, self.h_high)

    def speed(self, t: float) -> float:
        """Speed ``|gamma'(t)|``."""
        return float(np.linalg.norm(self.velocity(t)))

    # -- Frenet-Serret frame ----------------------------------------------
    def tangent(self, t: float) -> np.ndarray:
        """Unit tangent ``T = gamma' / |gamma'|``."""
        v = self.velocity(t)
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError(f"Singular point: zero velocity at t={t}")
        return v / n

    def binormal(self, t: float) -> np.ndarray:
        """Unit binormal ``B = (gamma' x gamma'') / |gamma' x gamma''|``."""
        v = self.velocity(t)
        a = self.acceleration(t)
        cross = np.cross(v, a)
        n = np.linalg.norm(cross)
        if n == 0:
            raise ValueError(
                f"Binormal undefined at t={t} (curve is locally straight)"
            )
        return cross / n

    def normal(self, t: float) -> np.ndarray:
        """Principal unit normal ``N = B x T``."""
        return np.cross(self.binormal(t), self.tangent(t))

    def frenet_frame(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the orthonormal triad ``(T, N, B)`` at ``t``."""
        T = self.tangent(t)
        B = self.binormal(t)
        N = np.cross(B, T)
        return T, N, B

    # -- curvature & torsion ----------------------------------------------
    def curvature(self, t: float) -> float:
        r"""Curvature ``kappa = |gamma' x gamma''| / |gamma'|^3``.

        Curvature measures how sharply the curve bends; its reciprocal is the
        radius of the osculating circle.
        """
        v = self.velocity(t)
        a = self.acceleration(t)
        speed = np.linalg.norm(v)
        if speed == 0:
            raise ValueError(f"Singular point: zero velocity at t={t}")
        return float(np.linalg.norm(np.cross(v, a)) / speed ** 3)

    def torsion(self, t: float) -> float:
        r"""Torsion ``tau = (gamma' x gamma'') . gamma''' / |gamma' x gamma''|^2``.

        Torsion measures how much the curve twists out of its osculating
        plane.  A planar curve has zero torsion everywhere.
        """
        v = self.velocity(t)
        a = self.acceleration(t)
        j = self.jerk(t)
        cross = np.cross(v, a)
        denom = float(np.dot(cross, cross))
        if denom == 0:
            raise ValueError(
                f"Torsion undefined at t={t} (curve is locally straight)"
            )
        return float(np.dot(cross, j) / denom)

    def radius_of_curvature(self, t: float) -> float:
        """Radius of the osculating circle, ``1 / kappa``."""
        k = self.curvature(t)
        if k == 0:
            return np.inf
        return 1.0 / k

    # -- integral quantities ----------------------------------------------
    def arc_length(self, t0: float, t1: float, n: int = 1000) -> float:
        """Arc length over ``[t0, t1]`` via composite Simpson's rule.

        Computes ``integral_{t0}^{t1} |gamma'(t)| dt``.  ``n`` (made even
        internally) is the number of sub-intervals.
        """
        if n % 2 == 1:
            n += 1
        ts = np.linspace(t0, t1, n + 1)
        speeds = np.array([self.speed(t) for t in ts])
        # Simpson weights: 1,4,2,4,...,4,1
        weights = np.ones(n + 1)
        weights[1:-1:2] = 4
        weights[2:-1:2] = 2
        dt = (t1 - t0) / n
        return float(dt / 3 * np.dot(weights, speeds))
