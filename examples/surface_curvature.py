"""Example: curvature of classic surfaces (sphere, cylinder, torus).

Run with:  python examples/surface_curvature.py

Demonstrates Gaussian and mean curvature and how they distinguish the local
shape of a surface: elliptic (K > 0), parabolic (K = 0) and hyperbolic
(K < 0) points.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differential_geometry import Surface


def main() -> None:
    # --- Unit sphere: every point is elliptic with K = 1, H = -1. ----------
    sphere = Surface(
        lambda u, v: np.array(
            [np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)]
        )
    )
    print("Unit sphere (radius 1)")
    K = sphere.gaussian_curvature(1.0, 0.5)
    H = sphere.mean_curvature(1.0, 0.5)
    print(f"  Gaussian K = {K:.4f}  (exact 1.0)")
    print(f"  Mean     H = {H:.4f}  (exact -1.0)\n")

    # --- Cylinder: K = 0 (developable), one zero principal curvature. -------
    R = 1.5
    cylinder = Surface(lambda u, v: np.array([R * np.cos(u), R * np.sin(u), v]))
    print(f"Cylinder (radius {R})")
    print(f"  Gaussian K = {cylinder.gaussian_curvature(0.6, 2.0):.6f}  (exact 0)")
    k1, k2 = cylinder.principal_curvatures(0.6, 2.0)
    print(f"  Principal curvatures = {k1:.4f}, {k2:.4f}  (exact 1/R={1/R:.4f}, 0)\n")

    # --- Torus: has elliptic, parabolic and hyperbolic regions. ------------
    Rb, rt = 3.0, 1.0  # tube centre radius and tube radius
    torus = Surface(
        lambda u, v: np.array(
            [
                (Rb + rt * np.cos(v)) * np.cos(u),
                (Rb + rt * np.cos(v)) * np.sin(u),
                rt * np.sin(v),
            ]
        )
    )
    print(f"Torus (R={Rb}, r={rt}) - Gaussian curvature by location:")
    cases = {
        "outer equator (v=0)  ": 0.0,        # K > 0  elliptic
        "top      (v=pi/2)    ": np.pi / 2,  # K = 0  parabolic
        "inner equator (v=pi) ": np.pi,      # K < 0  hyperbolic
    }
    for label, v in cases.items():
        Kt = torus.gaussian_curvature(0.0, v)
        sign = "elliptic" if Kt > 1e-6 else "hyperbolic" if Kt < -1e-6 else "parabolic"
        print(f"  {label} K = {Kt:+.4f}  -> {sign}")


if __name__ == "__main__":
    main()
