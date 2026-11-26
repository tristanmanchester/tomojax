"""Projection-domain center preview for acquisition setup.

All iterative alignment uses :func:`tomojax.align`; the preview below estimates
an axial detector center from opposing projections without reconstructing.
"""

from tomojax.alignment._geometry.detector_center import (
    CenterEstimate,
    CenterPair,
    estimate_center,
    estimate_center_report,
)

__all__ = ["CenterEstimate", "CenterPair", "estimate_center", "estimate_center_report"]
