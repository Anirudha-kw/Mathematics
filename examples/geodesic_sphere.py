"""Example: geodesics on the unit sphere via its intrinsic metric.

Run with:  python examples/geodesic_sphere.py

The sphere's metric in (theta, phi) = (polar, azimuth) coordinates is

    ds^2 = dtheta^2 + sin^2(theta) dphi^2.

Its geodesics are great circles.  Starting on the equator and heading in a
purely azimuthal direction, the geodesic must stay on the equator
(theta = pi/2); we integrate the geodesic equation and confirm this.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differential_geometry import RiemannianManifold


def main() -> None:
    # Metric of the unit sphere in (theta, phi) coordinates.
    sphere = RiemannianManifold(
        lambda x: np.array([[1.0, 0.0], [0.0, np.sin(x[0]) ** 2]])
    )

    print("Unit sphere metric  ds^2 = dtheta^2 + sin^2(theta) dphi^2\n")

    # Christoffel symbols at a generic point.
    theta0 = np.pi / 3
    gamma = sphere.christoffel(np.array([theta0, 0.0]))
    print(f"Christoffel symbols at theta = {theta0:.4f}:")
    print(f"  Gamma^theta_phiphi = {gamma[0, 1, 1]:+.4f} "
          f"(exact -sin*cos = {-np.sin(theta0)*np.cos(theta0):+.4f})")
    print(f"  Gamma^phi_thetaphi = {gamma[1, 0, 1]:+.4f} "
          f"(exact cot(theta) = {1/np.tan(theta0):+.4f})\n")

    # Geodesic starting on the equator heading east -> stays on the equator.
    x0 = np.array([np.pi / 2, 0.0])   # on equator
    v0 = np.array([0.0, 1.0])         # purely azimuthal
    ts, xs = sphere.geodesic(x0, v0, (0.0, 2 * np.pi), n_steps=400)
    max_dev = np.max(np.abs(xs[:, 0] - np.pi / 2))
    print("Equatorial geodesic (great circle):")
    print(f"  max deviation of theta from pi/2 = {max_dev:.2e}  (should be ~0)")
    print(f"  phi advanced from {xs[0, 1]:.4f} to {xs[-1, 1]:.4f} "
          f"(~2*pi = {2*np.pi:.4f})\n")

    # A tilted great circle: start on equator but head north-east.
    x0 = np.array([np.pi / 2, 0.0])
    v0 = np.array([0.6, 1.0])
    ts, xs = sphere.geodesic(x0, v0, (0.0, np.pi), n_steps=400)
    # Convert to 3-D to verify it lies on the unit sphere.
    theta, phi = xs[:, 0], xs[:, 1]
    pts = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=1,
    )
    radii = np.linalg.norm(pts, axis=1)
    print("Tilted great circle:")
    print(f"  embedded radius stays {radii.min():.6f}..{radii.max():.6f} "
          f"(should be 1, confirming it lies on the sphere)")


if __name__ == "__main__":
    main()
