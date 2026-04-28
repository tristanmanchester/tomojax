"""Provisional geometry-calibration foundation types.

The package root intentionally exposes only stable value/schema types. Helper
functions live in owner submodules such as ``tomojax.calibration.detector_grid``
or ``tomojax.calibration.manifest``. Estimation workflows remain executable
setup-alignment stages in ``tomojax.align``.
"""

from .state import CalibrationState, CalibrationVariable
from .units import DetectorPixelScale, DetectorPixelValue

__all__ = [
    "CalibrationState",
    "CalibrationVariable",
    "DetectorPixelScale",
    "DetectorPixelValue",
]
