"""Provisional geometry-calibration value types.

Production code should import calibration-derived geometry helpers through
``tomojax.geometry``. The package root intentionally exposes only retained
schema/value types; implementation helpers stay in calibration submodules until
the v2 plan either promotes or deletes this support package.
"""

from tomojax.calibration.api import (
    CalibrationState,
    CalibrationVariable,
    DetectorPixelScale,
    DetectorPixelValue,
)

__all__ = [
    "CalibrationState",
    "CalibrationVariable",
    "DetectorPixelScale",
    "DetectorPixelValue",
]
