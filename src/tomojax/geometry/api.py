"""Public API for geometry metadata, axes, and field-of-view helpers."""

from tomojax.core.geometry import (
    Detector,
    Geometry,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
    grid_volume_origin,
    normalize_axis_unit,
)
from tomojax.core.geometry.views import stack_view_poses
from tomojax.geometry._axes import (
    CORE_X_AXIS,
    CORE_Y_AXIS,
    CORE_Z_AXIS,
    DISK_VOLUME_AXES,
    INTERNAL_VOLUME_AXES,
    VOLUME_AXES_ATTR,
    axes_to_perm,
    infer_disk_axes,
    transpose_volume,
)
from tomojax.geometry._axis_geometry import (
    axis_pose_stack,
    axis_unit_from_rotations,
    nominal_axis_unit_from_inputs,
)
from tomojax.geometry._calibration import (
    CalibratedGeometryMetadataPatch,
    CalibrationState,
    CalibrationVariable,
    CandidateScore,
    MetricSpec,
    ObjectiveCard,
    build_calibrated_geometry_metadata_patch,
    validate_calibration_gauges,
)
from tomojax.geometry._detector_grid import (
    detector_grid_from_calibration,
    detector_grid_from_geometry_inputs,
)
from tomojax.geometry._fov import (
    RoiInfo,
    compute_roi,
    cylindrical_mask_xy,
    grid_from_detector_fov,
    grid_from_detector_fov_cube,
    grid_from_detector_fov_slices,
)
from tomojax.geometry._serialization import read_geometry_json, read_pose_params_csv
from tomojax.geometry._state import (
    AcquisitionParameters,
    GeometryState,
    PoseParameters,
    SetupParameters,
)

__all__ = [
    "CORE_X_AXIS",
    "CORE_Y_AXIS",
    "CORE_Z_AXIS",
    "DISK_VOLUME_AXES",
    "INTERNAL_VOLUME_AXES",
    "VOLUME_AXES_ATTR",
    "AcquisitionParameters",
    "CalibratedGeometryMetadataPatch",
    "CalibrationState",
    "CalibrationVariable",
    "CandidateScore",
    "Detector",
    "Geometry",
    "GeometryState",
    "Grid",
    "LaminographyGeometry",
    "MetricSpec",
    "ObjectiveCard",
    "ParallelGeometry",
    "PoseParameters",
    "RoiInfo",
    "RotationAxisGeometry",
    "SetupParameters",
    "axes_to_perm",
    "axis_pose_stack",
    "axis_unit_from_rotations",
    "build_calibrated_geometry_metadata_patch",
    "compute_roi",
    "cylindrical_mask_xy",
    "detector_grid_from_calibration",
    "detector_grid_from_geometry_inputs",
    "grid_from_detector_fov",
    "grid_from_detector_fov_cube",
    "grid_from_detector_fov_slices",
    "grid_volume_origin",
    "infer_disk_axes",
    "nominal_axis_unit_from_inputs",
    "normalize_axis_unit",
    "read_geometry_json",
    "read_pose_params_csv",
    "stack_view_poses",
    "transpose_volume",
    "validate_calibration_gauges",
]
