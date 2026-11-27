"""Geometry fitting for scans.

Use :func:`tomojax.align` with ``fit`` to select global calibration, per-view
translations and per-view rotations. On unposed, unrolled, well-sampled axial
parallel half-turns, ``fit="center"`` measures the rotation axis directly from
the projections without a working image; ``method="joint"``, joint solver
options, multi-parameter fits and other geometries use one joint solver.
``method="opposite-views"`` estimates a center from near-opposite view pairs
without an image. :class:`AlignConfig` holds the settings of the joint solver.
"""

from tomojax.alignment._config import AlignConfig

__all__ = ["AlignConfig"]
