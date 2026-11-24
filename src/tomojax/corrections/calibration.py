"""Metadata checks for raw-frame workflow preflight."""

from typing import Any

from ._engine import exposure_scales
from ._fields import check_stack
from ._validation import positive_finite


def check_calibration_layout(frames: Any) -> None:
    """Check calibration dimensions and exposure metadata before reading any counts."""
    for name in ("flats", "darks"):
        if (stack := getattr(frames, name)) is not None:
            check_stack(stack, frames.counts.shape[1:], name)
    if frames.white_level is not None:
        positive_finite(frames.white_level, "white_level")
    exposure_scales(frames.exposure_times, frames.flat_exposure_times, frames.views, frames.flats)
