"""Private calibration metadata helpers for the public geometry facade."""

from tomojax.geometry._calibration.conventions import ConventionAudit, ConventionEvidence
from tomojax.geometry._calibration.gauge import GaugeValidationError, validate_calibration_gauges
from tomojax.geometry._calibration.manifest import (
    CalibratedGeometryMetadataPatch,
    GeometryCalibrationPatch,
    build_calibrated_geometry_metadata_patch,
)
from tomojax.geometry._calibration.objectives import CandidateScore, MetricSpec, ObjectiveCard
from tomojax.geometry._calibration.state import CalibrationState, CalibrationVariable
from tomojax.geometry._calibration.units import DetectorPixelScale

__all__ = [
    "CalibratedGeometryMetadataPatch",
    "CalibrationState",
    "CalibrationVariable",
    "CandidateScore",
    "ConventionAudit",
    "ConventionEvidence",
    "DetectorPixelScale",
    "GaugeValidationError",
    "GeometryCalibrationPatch",
    "MetricSpec",
    "ObjectiveCard",
    "build_calibrated_geometry_metadata_patch",
    "validate_calibration_gauges",
]
