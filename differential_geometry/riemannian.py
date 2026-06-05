"""Riemannian metrics, Christoffel symbols and geodesics.

This module works intrinsically with an ``n``-dimensional manifold described by
a metric tensor field ``g(x)`` given in some coordinate chart.  From the metric
alone it derives:

* the inverse metric ``g^{ij}``,
* the **Christoffel symbols of the second kind** ``Gamma^k_{ij}`` (the
  Levi-Civita connection),
* **geodesics**, the straightest-possible curves, by integrating the geodesic
  equation ``x''^k + Gamma^k_{ij} x'^i x'^j = 0``.

Geodesics are the manifold generalisation of straight lines and are exactly the
shortest paths used in motion planning on curved configuration spaces.

Example
-------
Polar coordinates of the flat plane, ``ds^2 = dr^2 + r^2 dtheta^2``:

>>> import numpy as np
>>> g = lambda x: np.array([[1.0, 0.0], [0.0, x[0] ** 2]])
>>> M = RiemannianManifold(g)
>>> # Christoffel symbol Gamma^r_{theta theta} = -r
>>> round(M.christoffel(np.array([2.0, 0.0]))[0, 1, 1], 4)
-2.0
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULT_STEP = 1e-5


class RiemannianManifold:
    """An ``n``-dimensional manifold equipped with a metric tensor field.

    Parameters
    ----------
    metric : callable
        Maps a coordinate vector ``x`` (shape ``(n,)``) to a symmetric positive
        definite ``(n, n)`` metric matrix ``g(x)``.
    h : float, optional
        Finite-difference step used to differentiate the metric.
    """

    def __init__(self, metric: Callable[[np.ndarray], np.ndarray], h: float = DEFAULT_STEP):
        self.metric = metric
        self.h = h

    @property
    def dim(self) -> int:
        # Infer dimension by probing the metric at the origin.
        return self.metric(np.zeros(self._probe_dim())).shape[0]

    def _probe_dim(self) -> int:
        # Try increasing dimensions until the metric accepts the input.
        for n in range(1, 11):
            try:
                g = np.asarray(self.metric(np.zeros(n)), dtype=float)
                if g.shape == (n, n):
                    return n
            except Exception:  # noqa: BLE001 - probing, any failure means wrong dim
                continue
        raise ValueError("Could not infer manifold dimension (tried 1..10)")

    def g(self, x: np.ndarray) -> np.ndarray:
        """Metric matrix ``g_{ij}(x)``."""
        return np.asarray(self.metric(np.asarray(x, dtype=float)), dtype=float)

    def g_inv(self, x: np.ndarray) -> np.ndarray:
        """Inverse metric ``g^{ij}(x)``."""
        return np.linalg.inv(self.g(x))

    def metric_derivative(self, x: np.ndarray) -> np.ndarray:
        """Partial derivatives of the metric, ``dg[k, i, j] = d g_{ij} / dx^k``."""
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        dg = np.zeros((n, n, n))
        for k in range(n):
            step = np.zeros(n)
            step[k] = self.h
            dg[k] = (self.g(x + step) - self.g(x - step)) / (2 * self.h)
        return dg

    def christoffel(self, x: np.ndarray) -> np.ndarray:
        r"""Christoffel symbols of the second kind ``Gamma^k_{ij}``.

        Computed from the metric via

        ``Gamma^k_{ij} = 1/2 g^{kl} (d_i g_{lj} + d_j g_{li} - d_l g_{ij})``.

        Returns an array indexed ``[k, i, j]`` and symmetric in ``i, j``.
        """
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        ginv = self.g_inv(x)
        dg = self.metric_derivative(x)  # dg[k, i, j] = d_k g_{ij}
        gamma = np.zeros((n, n, n))
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for l in range(n):
                        # d_i g_{lj} + d_j g_{li} - d_l g_{ij}
                        s += ginv[k, l] * (dg[i, l, j] + dg[j, l, i] - dg[l, i, j])
                    gamma[k, i, j] = 0.5 * s
        return gamma

    def geodesic_acceleration(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""Right-hand side of the geodesic equation.

        Returns ``a^k = -Gamma^k_{ij} v^i v^j``, i.e. the coordinate
        acceleration required for ``x(t)`` to be a geodesic.
        """
        gamma = self.christoffel(x)
        v = np.asarray(v, dtype=float)
        return -np.einsum("kij,i,j->k", gamma, v, v)

    def geodesic(
        self,
        x0: np.ndarray,
        v0: np.ndarray,
        t_span: tuple[float, float],
        n_steps: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate a geodesic from initial position/velocity.

        Uses a classical 4th-order Runge-Kutta scheme on the first-order system
        ``(x, v)' = (v, -Gamma(x) v v)``.

        Parameters
        ----------
        x0, v0 : array_like
            Initial position and velocity (shape ``(n,)``).
        t_span : (float, float)
            Start and end parameter values.
        n_steps : int
            Number of integration steps.

        Returns
        -------
        ts : numpy.ndarray
            Parameter samples, shape ``(n_steps + 1,)``.
        xs : numpy.ndarray
            Geodesic positions, shape ``(n_steps + 1, n)``.
        """
        x0 = np.asarray(x0, dtype=float)
        v0 = np.asarray(v0, dtype=float)
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        n = x0.shape[0]

        def f(state: np.ndarray) -> np.ndarray:
            x = state[:n]
            v = state[n:]
            return np.concatenate([v, self.geodesic_acceleration(x, v)])

        ts = np.linspace(t0, t1, n_steps + 1)
        xs = np.zeros((n_steps + 1, n))
        state = np.concatenate([x0, v0])
        xs[0] = x0
        for i in range(n_steps):
            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            xs[i + 1] = state[:n]
        return ts, xs
