"""Internal geometry-calibration foundations.

This package intentionally contains schema, unit, gauge, convention, and detector-grid
helpers only. Estimation workflows live in later phases.
"""

from .conventions import ConventionAudit, ConventionEvidence
from .center import (
    DETECTOR_CENTER_DOFS,
    DetectorCenterCalibrationConfig,
    DetectorCenterCalibrationResult,
    DetectorCenterIteration,
    calibrate_detector_center,
    detector_with_center_offset,
)
from .axis import (
    AxisDirectionCalibrationConfig,
    AxisDirectionCalibrationResult,
    AxisDirectionIteration,
    calibrate_axis_direction,
)
from .axis_geometry import AXIS_DIRECTION_DOFS
from .detector_grid import (
    detector_grid_from_calibration,
    detector_grid_from_center_offset,
    detector_grid_from_detector_roll,
    detector_grid_from_geometry_inputs,
    offset_detector_grid,
    transform_detector_grid,
)
from .detector_grid import zero_center_detector_grid
from .gauge import GaugeConflict, GaugeValidationError, validate_calibration_gauges
from .manifest import CALIBRATION_MANIFEST_SCHEMA_VERSION, build_calibration_manifest
from .objectives import CandidateScore, MetricSpec, ObjectiveCard
from .roll import (
    DETECTOR_ROLL_DOFS,
    DetectorRollCalibrationConfig,
    DetectorRollCalibrationResult,
    DetectorRollIteration,
    calibrate_detector_roll,
)
from .state import CalibrationState, CalibrationVariable
from .units import DetectorPixelScale, DetectorPixelValue

__all__ = [
    "CALIBRATION_MANIFEST_SCHEMA_VERSION",
    "AXIS_DIRECTION_DOFS",
    "AxisDirectionCalibrationConfig",
    "AxisDirectionCalibrationResult",
    "AxisDirectionIteration",
    "CalibrationState",
    "CalibrationVariable",
    "CandidateScore",
    "ConventionAudit",
    "ConventionEvidence",
    "DETECTOR_CENTER_DOFS",
    "DETECTOR_ROLL_DOFS",
    "DetectorCenterCalibrationConfig",
    "DetectorCenterCalibrationResult",
    "DetectorCenterIteration",
    "DetectorPixelScale",
    "DetectorPixelValue",
    "DetectorRollCalibrationConfig",
    "DetectorRollCalibrationResult",
    "DetectorRollIteration",
    "GaugeConflict",
    "GaugeValidationError",
    "MetricSpec",
    "ObjectiveCard",
    "build_calibration_manifest",
    "calibrate_axis_direction",
    "calibrate_detector_center",
    "calibrate_detector_roll",
    "detector_grid_from_calibration",
    "detector_grid_from_center_offset",
    "detector_grid_from_detector_roll",
    "detector_grid_from_geometry_inputs",
    "detector_with_center_offset",
    "offset_detector_grid",
    "transform_detector_grid",
    "validate_calibration_gauges",
    "zero_center_detector_grid",
]
