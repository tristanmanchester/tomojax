"""Geometry fitting for scans.

Use :func:`tomojax.align` with ``fit`` to select global calibration, per-view
translations and per-view rotations. The implementation uses one joint solver.
``method="opposite-views"`` estimates a center from near-opposite views without an image.
:class:`AlignConfig` holds every setting.
"""

from tomojax.alignment._config import AlignConfig

__all__ = ["AlignConfig"]
