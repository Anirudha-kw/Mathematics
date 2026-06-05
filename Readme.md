# Mathematics for Robotics

A small, dependency-light numerical **differential geometry** toolkit in pure
Python + NumPy.  You supply a parametrisation (or a metric) and the library
computes the classical invariants for you using finite-difference derivatives —
no analytic derivatives required.

## Features

| Module | Class | What it computes |
|--------|-------|------------------|
| `curves.py` | `Curve` | Frenet–Serret frame (T, N, B), curvature, torsion, speed, arc length |
| `surfaces.py` | `Surface` | First/second fundamental forms, shape operator, Gaussian/mean/principal curvatures, area |
| `riemannian.py` | `RiemannianManifold` | Inverse metric, Christoffel symbols, geodesics (RK4) |
| `numdiff.py` | — | High-accuracy central finite-difference derivatives |

These are the core tools behind robot path planning (curvature ↔ turning-radius
limits), trajectory tracking (the Frenet frame), and motion planning on curved
configuration spaces (geodesics).

## Install

```bash
pip install -r requirements.txt   # just numpy
```

## Quick start

```python
import numpy as np
from differential_geometry import Curve, Surface, RiemannianManifold

# --- A space curve (helix) --------------------------------------------
helix = Curve(lambda t: np.array([np.cos(t), np.sin(t), 0.3 * t]))
helix.curvature(0.0)     # how sharply it bends
helix.torsion(0.0)       # how it twists out of plane
T, N, B = helix.frenet_frame(0.0)

# --- A surface (unit sphere) ------------------------------------------
sphere = Surface(lambda u, v: np.array([
    np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)]))
sphere.gaussian_curvature(1.0, 0.5)   # -> 1.0
sphere.mean_curvature(1.0, 0.5)       # -> -1.0

# --- A Riemannian manifold (sphere metric) and a geodesic -------------
g = lambda x: np.array([[1.0, 0.0], [0.0, np.sin(x[0]) ** 2]])
M = RiemannianManifold(g)
ts, path = M.geodesic(x0=np.array([np.pi/2, 0.0]),
                      v0=np.array([0.0, 1.0]),
                      t_span=(0.0, 2 * np.pi))   # a great circle
```

## Examples

Runnable demos live in `examples/`:

```bash
python examples/curve_helix.py        # constant curvature & torsion of a helix
python examples/surface_curvature.py  # sphere / cylinder / torus curvatures
python examples/geodesic_sphere.py    # great circles as geodesics
```

## Tests

Each test checks the numerics against known analytic results (unit circle,
helix, sphere, cylinder, flat plane in polar coordinates, …):

```bash
python tests/test_curves.py
python tests/test_surfaces.py
python tests/test_riemannian.py
# or, if you have pytest:
pytest tests/
```

## Notes on accuracy

* First/second derivatives use a 5-point central stencil (`O(h^4)`).
* Torsion needs a third derivative, which is far more sensitive to round-off,
  so `Curve` uses a separate larger step `h_high` (default `1e-3`) for it.
* All step sizes are constructor arguments if you need to tune accuracy.
