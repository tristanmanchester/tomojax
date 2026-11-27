"""Pair-only opposing-view center estimation for acquisition setup.

General alignment and full-sinogram axial center calibration use
:func:`tomojax.align` (``tj.align(scan, fit="center")``). The helpers below
implement the pair-only estimator used by ``method="opposite-views"``, which
estimates an axial detector center from near-opposite projections without
reconstructing.
"""

from tomojax.alignment._geometry.detector_center import (
    CenterEstimate,
    CenterPair,
    estimate_center,
    estimate_center_report,
)

__all__ = ["CenterEstimate", "CenterPair", "estimate_center", "estimate_center_report"]
