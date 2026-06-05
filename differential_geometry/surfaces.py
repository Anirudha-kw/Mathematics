"""Differential geometry of parametric surfaces in 3-D.

:class:`Surface` wraps a parametrisation ``r(u, v) -> R^3`` and computes the
classical local invariants:

* the **first fundamental form** ``I = [[E, F], [F, G]]`` (the induced metric,
  governing lengths, angles and areas on the surface),
* the **second fundamental form** ``II = [[L, M], [M, N]]`` (how the surface
  curves inside the ambient space),
* the **shape operator** ``S = I^{-1} II`` and from it the principal,
  Gaussian and mean curvatures.

Gaussian curvature is intrinsic (Gauss's *Theorema Egregium*): it can be felt
by an inhabitant of the surface without reference to the embedding.

Example
-------
>>> import numpy as np
>>> from differential_geometry.surfaces import Surface
>>> sphere = Surface(lambda u, v: np.array([
...     np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)]))
>>> round(sphere.gaussian_curvature(1.0, 0.5), 4)  # unit sphere -> K = 1
1.0
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import numdiff


class Surface:
    """A smooth parametric surface ``r(u, v)`` in ``R^3``.

    Parameters
    ----------
    r : callable
        Maps ``(u, v)`` to a point in ``R^3``.
    h : float, optional
        Finite-difference step used for all numerical derivatives.
    """

    def __init__(self, r: Callable[[float, float], np.ndarray], h: float = numdiff.DEFAULT_STEP):
        self.r = r
        self.h = h

    # -- point and tangent vectors ----------------------------------------
    def point(self, u: float, v: float) -> np.ndarray:
        """Position ``r(u, v)``."""
        return np.asarray(self.r(u, v), dtype=float)

    def r_u(self, u: float, v: float) -> np.ndarray:
        """Partial derivative ``dr/du`` (a tangent vector)."""
        return numdiff.partial(self.r, u, v, axis=0, h=self.h)

    def r_v(self, u: float, v: float) -> np.ndarray:
        """Partial derivative ``dr/dv`` (a tangent vector)."""
        return numdiff.partial(self.r, u, v, axis=1, h=self.h)

    def r_uu(self, u: float, v: float) -> np.ndarray:
        return numdiff.second_partial(self.r, u, v, 0, 0, self.h)

    def r_uv(self, u: float, v: float) -> np.ndarray:
        return numdiff.second_partial(self.r, u, v, 0, 1, self.h)

    def r_vv(self, u: float, v: float) -> np.ndarray:
        return numdiff.second_partial(self.r, u, v, 1, 1, self.h)

    def normal(self, u: float, v: float) -> np.ndarray:
        """Unit surface normal ``(r_u x r_v) / |r_u x r_v|``."""
        n = np.cross(self.r_u(u, v), self.r_v(u, v))
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError(f"Degenerate (non-regular) point at (u, v)=({u}, {v})")
        return n / norm

    # -- fundamental forms -------------------------------------------------
    def first_fundamental_form(self, u: float, v: float) -> np.ndarray:
        """First fundamental form (metric) ``[[E, F], [F, G]]``."""
        ru = self.r_u(u, v)
        rv = self.r_v(u, v)
        E = float(np.dot(ru, ru))
        F = float(np.dot(ru, rv))
        G = float(np.dot(rv, rv))
        return np.array([[E, F], [F, G]])

    def second_fundamental_form(self, u: float, v: float) -> np.ndarray:
        """Second fundamental form ``[[L, M], [M, N]]``."""
        n = self.normal(u, v)
        L = float(np.dot(self.r_uu(u, v), n))
        M = float(np.dot(self.r_uv(u, v), n))
        N = float(np.dot(self.r_vv(u, v), n))
        return np.array([[L, M], [M, N]])

    def shape_operator(self, u: float, v: float) -> np.ndarray:
        """Shape (Weingarten) operator ``S = I^{-1} II``.

        Its eigenvalues are the principal curvatures and its eigenvectors the
        principal directions (expressed in the ``(r_u, r_v)`` basis).
        """
        I = self.first_fundamental_form(u, v)
        II = self.second_fundamental_form(u, v)
        return np.linalg.solve(I, II)

    # -- curvatures --------------------------------------------------------
    def gaussian_curvature(self, u: float, v: float) -> float:
        r"""Gaussian curvature ``K = det(II) / det(I) = (LN - M^2)/(EG - F^2)``."""
        I = self.first_fundamental_form(u, v)
        II = self.second_fundamental_form(u, v)
        return float(np.linalg.det(II) / np.linalg.det(I))

    def mean_curvature(self, u: float, v: float) -> float:
        r"""Mean curvature ``H = trace(S) / 2 = (EN - 2FM + GL)/(2(EG - F^2))``."""
        E, F = self.first_fundamental_form(u, v)[0]
        _, G = self.first_fundamental_form(u, v)[1]
        L, M = self.second_fundamental_form(u, v)[0]
        _, N = self.second_fundamental_form(u, v)[1]
        return float((E * N - 2 * F * M + G * L) / (2 * (E * G - F * F)))

    def principal_curvatures(self, u: float, v: float) -> tuple[float, float]:
        """Principal curvatures ``(k1, k2)`` with ``k1 >= k2``.

        These are the eigenvalues of the shape operator and equal
        ``H +/- sqrt(H^2 - K)``.
        """
        H = self.mean_curvature(u, v)
        K = self.gaussian_curvature(u, v)
        disc = max(H * H - K, 0.0)  # clamp tiny negative round-off
        root = np.sqrt(disc)
        return H + root, H - root

    def principal_directions(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Principal directions as ambient ``R^3`` vectors.

        Returns the two unit tangent vectors along which normal curvature is
        extremal, ordered to match :meth:`principal_curvatures`.
        """
        S = self.shape_operator(u, v)
        eigvals, eigvecs = np.linalg.eig(S)
        # Order by eigenvalue descending to match principal_curvatures (k1>=k2).
        order = np.argsort(eigvals)[::-1]
        ru = self.r_u(u, v)
        rv = self.r_v(u, v)
        dirs = []
        for idx in order:
            a, b = eigvecs[:, idx].real
            d = a * ru + b * rv
            dirs.append(d / np.linalg.norm(d))
        return dirs[0], dirs[1]

    # -- intrinsic quantities ---------------------------------------------
    def area(self, u0: float, u1: float, v0: float, v1: float, n: int = 100) -> float:
        r"""Surface area over ``[u0, u1] x [v0, v1]``.

        Integrates the area element ``sqrt(EG - F^2) du dv`` with the midpoint
        rule on an ``n x n`` grid.
        """
        us = np.linspace(u0, u1, n + 1)
        vs = np.linspace(v0, v1, n + 1)
        umid = 0.5 * (us[:-1] + us[1:])
        vmid = 0.5 * (vs[:-1] + vs[1:])
        du = (u1 - u0) / n
        dv = (v1 - v0) / n
        total = 0.0
        for u in umid:
            for v in vmid:
                I = self.first_fundamental_form(u, v)
                total += np.sqrt(max(np.linalg.det(I), 0.0))
        return float(total * du * dv)
