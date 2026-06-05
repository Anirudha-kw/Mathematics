"""A small, dependency-light differential geometry toolkit.

The package provides numerical implementations of the classical local theory of

* curves        -> :class:`differential_geometry.curves.Curve`
* surfaces      -> :class:`differential_geometry.surfaces.Surface`
* Riemannian
  manifolds     -> :class:`differential_geometry.riemannian.RiemannianManifold`

Everything is built on top of ``numpy`` finite-difference derivatives
(:mod:`differential_geometry.numdiff`) so the user only ever supplies a
parametrisation or a metric, never analytic derivatives.

Quick start
-----------
>>> import numpy as np
>>> from differential_geometry import Curve, Surface, RiemannianManifold
>>> circle = Curve(lambda t: np.array([np.cos(t), np.sin(t), 0.0]))
>>> round(circle.curvature(0.0), 6)  # unit circle has curvature 1
1.0
"""

from .curves import Curve
from .surfaces import Surface
from .riemannian import RiemannianManifold

__all__ = ["Curve", "Surface", "RiemannianManifold"]
__version__ = "0.1.0"
