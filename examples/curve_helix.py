"""Example: Frenet-Serret analysis of a circular helix.

Run with:  python examples/curve_helix.py

A helix has *constant* curvature and torsion, which makes it the canonical
worked example for the local theory of space curves.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differential_geometry import Curve


def main() -> None:
    a, b = 2.0, 1.0  # radius and pitch parameter
    helix = Curve(lambda t: np.array([a * np.cos(t), a * np.sin(t), b * t]))

    print("Circular helix  gamma(t) = (a cos t, a sin t, b t)")
    print(f"  a = {a}, b = {b}\n")

    # Analytic values for reference.
    k_exact = a / (a ** 2 + b ** 2)
    tau_exact = b / (a ** 2 + b ** 2)
    print(f"Analytic curvature  kappa = a/(a^2+b^2) = {k_exact:.6f}")
    print(f"Analytic torsion    tau   = b/(a^2+b^2) = {tau_exact:.6f}\n")

    print(f"{'t':>6} {'kappa':>10} {'tau':>10} {'speed':>10}")
    for t in np.linspace(0, 2 * np.pi, 6):
        print(
            f"{t:6.3f} {helix.curvature(t):10.6f} "
            f"{helix.torsion(t):10.6f} {helix.speed(t):10.6f}"
        )

    print("\nFrenet frame at t = 1.0:")
    T, N, B = helix.frenet_frame(1.0)
    print(f"  T = {np.round(T, 4)}")
    print(f"  N = {np.round(N, 4)}")
    print(f"  B = {np.round(B, 4)}")

    length = helix.arc_length(0.0, 2 * np.pi)
    print(f"\nArc length over one turn = {length:.6f}")
    print(f"Exact (2*pi*sqrt(a^2+b^2)) = {2 * np.pi * np.sqrt(a**2 + b**2):.6f}")


if __name__ == "__main__":
    main()
