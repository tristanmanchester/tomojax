"""Private calibration metadata helpers for the public geometry facade."""

from tomojax.geometry._calibration.gauge import GaugeValidationError, validate_calibration_gauges
from tomojax.geometry._calibration.manifest import (
    CalibratedGeometryMetadataPatch,
    GeometryCalibrationPatch,
    build_calibrated_geometry_metadata_patch,
)
from tomojax.geometry._calibration.state import CalibrationState, CalibrationVariable

__all__ = [
    "CalibratedGeometryMetadataPatch",
    "CalibrationState",
    "CalibrationVariable",
    "GaugeValidationError",
    "GeometryCalibrationPatch",
    "build_calibrated_geometry_metadata_patch",
    "validate_calibration_gauges",
]
