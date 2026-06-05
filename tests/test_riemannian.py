"""Tests for differential_geometry.riemannian against known analytic results."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differential_geometry import RiemannianManifold  # noqa: E402

TOL = 1e-4


def polar_plane():
    # Flat plane in polar coordinates: ds^2 = dr^2 + r^2 dtheta^2.
    return RiemannianManifold(lambda x: np.array([[1.0, 0.0], [0.0, x[0] ** 2]]))


def test_polar_christoffel():
    # Non-zero symbols: Gamma^r_{tt} = -r, Gamma^t_{rt} = Gamma^t_{tr} = 1/r.
    M = polar_plane()
    r = 2.0
    gamma = M.christoffel(np.array([r, 0.0]))
    assert abs(gamma[0, 1, 1] - (-r)) < TOL          # Gamma^r_{theta theta}
    assert abs(gamma[1, 0, 1] - 1.0 / r) < TOL       # Gamma^theta_{r theta}
    assert abs(gamma[1, 1, 0] - 1.0 / r) < TOL       # symmetry
    assert abs(gamma[0, 0, 0]) < TOL                 # Gamma^r_{rr} = 0


def test_euclidean_christoffel_vanishes():
    # Cartesian flat metric -> all Christoffel symbols zero.
    M = RiemannianManifold(lambda x: np.eye(2))
    gamma = M.christoffel(np.array([0.3, -1.2]))
    assert np.allclose(gamma, 0.0, atol=TOL)


def test_geodesic_straight_line_in_cartesian():
    # In Cartesian coordinates a geodesic is a straight line x(t) = x0 + t v0.
    M = RiemannianManifold(lambda x: np.eye(2))
    x0 = np.array([0.0, 0.0])
    v0 = np.array([1.0, 2.0])
    ts, xs = M.geodesic(x0, v0, (0.0, 1.0), n_steps=50)
    expected = x0[None, :] + ts[:, None] * v0[None, :]
    assert np.allclose(xs, expected, atol=1e-6)


def test_geodesic_in_polar_is_straight_line():
    # A radial line theta = const is a geodesic; integrate from (r=1, theta=0)
    # with purely radial velocity and check theta stays constant, r grows
    # linearly.
    M = polar_plane()
    x0 = np.array([1.0, 0.0])
    v0 = np.array([1.0, 0.0])
    ts, xs = M.geodesic(x0, v0, (0.0, 2.0), n_steps=200)
    assert np.allclose(xs[:, 1], 0.0, atol=1e-6)            # theta const
    assert np.allclose(xs[:, 0], 1.0 + ts, atol=1e-4)      # r = 1 + t


def test_polar_geodesic_matches_cartesian_line():
    # A geodesic in the polar-coordinate plane must trace a Euclidean straight
    # line. Start off-radial and compare against the analytic line.
    M = polar_plane()
    r0, th0 = 1.0, 0.0
    # Pick a velocity; convert to Cartesian to know the true path.
    rdot, thdot = 0.5, 1.0
    x0 = np.array([r0, th0])
    v0 = np.array([rdot, thdot])
    ts, xs = M.geodesic(x0, v0, (0.0, 0.6), n_steps=400)

    # Cartesian start point and velocity.
    px0 = r0 * np.cos(th0)
    py0 = r0 * np.sin(th0)
    pvx = rdot * np.cos(th0) - r0 * thdot * np.sin(th0)
    pvy = rdot * np.sin(th0) + r0 * thdot * np.cos(th0)

    px = xs[:, 0] * np.cos(xs[:, 1])
    py = xs[:, 0] * np.sin(xs[:, 1])
    assert np.allclose(px, px0 + pvx * ts, atol=1e-3)
    assert np.allclose(py, py0 + pvy * ts, atol=1e-3)


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
