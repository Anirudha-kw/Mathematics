"""Tests for differential_geometry.surfaces against known analytic results."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differential_geometry import Surface  # noqa: E402

TOL = 1e-4


def sphere(R=1.0):
    # Standard sphere parametrisation, u = polar angle, v = azimuth.
    return Surface(
        lambda u, v: np.array(
            [R * np.sin(u) * np.cos(v), R * np.sin(u) * np.sin(v), R * np.cos(u)]
        )
    )


def plane():
    return Surface(lambda u, v: np.array([u, v, 0.0]))


def cylinder(R=1.0):
    return Surface(lambda u, v: np.array([R * np.cos(u), R * np.sin(u), v]))


def test_sphere_gaussian_curvature():
    # Unit sphere: K = 1 everywhere.
    s = sphere(1.0)
    for u in (0.5, 1.0, 2.0):
        for v in (0.0, 1.0, 2.5):
            assert abs(s.gaussian_curvature(u, v) - 1.0) < TOL


def test_sphere_radius_curvature():
    # Sphere of radius R: K = 1/R^2, |H| = 1/R.
    R = 2.0
    s = sphere(R)
    assert abs(s.gaussian_curvature(1.0, 0.7) - 1.0 / R ** 2) < TOL
    assert abs(abs(s.mean_curvature(1.0, 0.7)) - 1.0 / R) < TOL


def test_plane_is_flat():
    p = plane()
    assert abs(p.gaussian_curvature(0.3, -0.4)) < TOL
    assert abs(p.mean_curvature(0.3, -0.4)) < TOL


def test_cylinder_curvatures():
    # Cylinder radius R: K = 0, principal curvatures {0, 1/R}.
    R = 1.5
    c = cylinder(R)
    assert abs(c.gaussian_curvature(0.6, 2.0)) < TOL
    k1, k2 = c.principal_curvatures(0.6, 2.0)
    pc = sorted([abs(k1), abs(k2)])
    assert abs(pc[0]) < TOL
    assert abs(pc[1] - 1.0 / R) < TOL


def test_first_fundamental_form_sphere():
    # For the unit sphere: E = 1, F = 0, G = sin^2(u).
    s = sphere(1.0)
    u, v = 1.1, 0.3
    I = s.first_fundamental_form(u, v)
    assert abs(I[0, 0] - 1.0) < TOL
    assert abs(I[0, 1]) < TOL
    assert abs(I[1, 1] - np.sin(u) ** 2) < TOL


def test_unit_normal():
    s = sphere(1.0)
    u, v = 1.0, 0.5
    n = s.normal(u, v)
    assert abs(np.linalg.norm(n) - 1.0) < TOL
    # On the unit sphere the outward normal equals the position.
    assert np.allclose(np.abs(n), np.abs(s.point(u, v)), atol=TOL)


def test_sphere_area():
    # Area of unit sphere is 4*pi.
    s = sphere(1.0)
    A = s.area(0.0, np.pi, 0.0, 2 * np.pi, n=120)
    assert abs(A - 4 * np.pi) < 1e-2


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
