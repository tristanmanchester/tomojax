"""Internal geometry-calibration foundations.

This package intentionally contains schema, unit, gauge, convention, and detector-grid
helpers only. Estimation workflows live in later phases.
"""

from .conventions import ConventionAudit, ConventionEvidence
from .detector_grid import detector_grid_from_center_offset, offset_detector_grid
from .detector_grid import zero_center_detector_grid
from .gauge import GaugeConflict, GaugeValidationError, validate_calibration_gauges
from .manifest import CALIBRATION_MANIFEST_SCHEMA_VERSION, build_calibration_manifest
from .objectives import CandidateScore, MetricSpec, ObjectiveCard
from .state import CalibrationState, CalibrationVariable
from .units import DetectorPixelScale, DetectorPixelValue

__all__ = [
    "CALIBRATION_MANIFEST_SCHEMA_VERSION",
    "CalibrationState",
    "CalibrationVariable",
    "CandidateScore",
    "ConventionAudit",
    "ConventionEvidence",
    "DetectorPixelScale",
    "DetectorPixelValue",
    "GaugeConflict",
    "GaugeValidationError",
    "MetricSpec",
    "ObjectiveCard",
    "build_calibration_manifest",
    "detector_grid_from_center_offset",
    "offset_detector_grid",
    "validate_calibration_gauges",
    "zero_center_detector_grid",
]
