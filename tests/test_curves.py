"""Tests for differential_geometry.curves against known analytic results."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differential_geometry import Curve  # noqa: E402

TOL = 1e-5


def test_unit_circle_curvature():
    # Unit circle in the xy-plane: curvature 1, torsion 0 everywhere.
    circle = Curve(lambda t: np.array([np.cos(t), np.sin(t), 0.0]))
    for t in np.linspace(0, 2 * np.pi, 7):
        assert abs(circle.curvature(t) - 1.0) < TOL


def test_circle_radius_curvature():
    # Circle of radius R has curvature 1/R.
    R = 3.0
    circle = Curve(lambda t: np.array([R * np.cos(t), R * np.sin(t), 0.0]))
    assert abs(circle.curvature(1.3) - 1.0 / R) < TOL
    assert abs(circle.radius_of_curvature(1.3) - R) < 1e-4


def test_planar_curve_zero_torsion():
    # Any curve in a plane has zero torsion. Use an ellipse.
    ellipse = Curve(lambda t: np.array([2 * np.cos(t), np.sin(t), 0.0]))
    assert abs(ellipse.torsion(0.7)) < 1e-4


def test_helix_curvature_and_torsion():
    # Helix (a cos t, a sin t, b t):
    #   curvature = a / (a^2 + b^2), torsion = b / (a^2 + b^2).
    a, b = 2.0, 1.0
    helix = Curve(lambda t: np.array([a * np.cos(t), a * np.sin(t), b * t]))
    expected_k = a / (a ** 2 + b ** 2)
    expected_tau = b / (a ** 2 + b ** 2)
    assert abs(helix.curvature(0.4) - expected_k) < TOL
    assert abs(helix.torsion(0.4) - expected_tau) < 1e-4


def test_frenet_frame_orthonormal():
    helix = Curve(lambda t: np.array([np.cos(t), np.sin(t), 0.3 * t]))
    T, N, B = helix.frenet_frame(0.9)
    # Orthonormality.
    for u in (T, N, B):
        assert abs(np.linalg.norm(u) - 1.0) < TOL
    assert abs(np.dot(T, N)) < TOL
    assert abs(np.dot(T, B)) < TOL
    assert abs(np.dot(N, B)) < TOL
    # Right-handed: T x N = B.
    assert np.allclose(np.cross(T, N), B, atol=1e-4)


def test_arc_length_of_circle():
    # Full unit circle has length 2*pi.
    circle = Curve(lambda t: np.array([np.cos(t), np.sin(t), 0.0]))
    L = circle.arc_length(0.0, 2 * np.pi)
    assert abs(L - 2 * np.pi) < 1e-6


def test_arc_length_of_helix():
    a, b = 1.0, 1.0
    helix = Curve(lambda t: np.array([a * np.cos(t), a * np.sin(t), b * t]))
    # Speed is constant sqrt(a^2 + b^2), so length over [0, T] is T*sqrt(2).
    L = helix.arc_length(0.0, 3.0)
    assert abs(L - 3.0 * np.sqrt(2.0)) < 1e-6


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
