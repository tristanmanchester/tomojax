"""Small production facade for geometry.

The package root keeps stable geometry objects and commonly used product
helpers. The broader developer-oriented surface remains available from
``tomojax.geometry.api``. Geometry objects import without JAX; the remaining
helpers load ``tomojax.geometry.api`` on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tomojax.core.geometry import (
    ConeBeam,
    ConeGeometry,
    Detector,
    Geometry,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
    beam_of,
    grid_volume_origin,
)

if TYPE_CHECKING:
    from tomojax.geometry.api import (
        CORE_X_AXIS,
        CORE_Y_AXIS,
        CORE_Z_AXIS,
        DISK_VOLUME_AXES,
        INTERNAL_VOLUME_AXES,
        VOLUME_AXES_ATTR,
        CalibrationState,
        CalibrationVariable,
        GeometryState,
        axes_to_perm,
        axis_pose_stack,
        axis_unit_from_rotations,
        build_calibrated_geometry_metadata_patch,
        build_calibration_manifest,
        compute_roi,
        cylindrical_mask_xy,
        detector_grid_from_calibration,
        detector_grid_from_geometry_inputs,
        grid_from_detector_fov,
        grid_from_detector_fov_cube,
        grid_from_detector_fov_slices,
        infer_disk_axes,
        nominal_axis_unit_from_inputs,
        read_geometry_json,
        read_pose_params_csv,
        stack_view_poses,
        transpose_volume,
        validate_calibration_gauges,
    )

_LAZY = frozenset(
    {
        "CORE_X_AXIS",
        "CORE_Y_AXIS",
        "CORE_Z_AXIS",
        "DISK_VOLUME_AXES",
        "INTERNAL_VOLUME_AXES",
        "VOLUME_AXES_ATTR",
        "CalibrationState",
        "CalibrationVariable",
        "GeometryState",
        "axes_to_perm",
        "axis_pose_stack",
        "axis_unit_from_rotations",
        "build_calibrated_geometry_metadata_patch",
        "build_calibration_manifest",
        "compute_roi",
        "cylindrical_mask_xy",
        "detector_grid_from_calibration",
        "detector_grid_from_geometry_inputs",
        "grid_from_detector_fov",
        "grid_from_detector_fov_cube",
        "grid_from_detector_fov_slices",
        "infer_disk_axes",
        "nominal_axis_unit_from_inputs",
        "read_geometry_json",
        "read_pose_params_csv",
        "stack_view_poses",
        "transpose_volume",
        "validate_calibration_gauges",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from tomojax.geometry import api

    value = getattr(api, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "CORE_X_AXIS",
    "CORE_Y_AXIS",
    "CORE_Z_AXIS",
    "DISK_VOLUME_AXES",
    "INTERNAL_VOLUME_AXES",
    "VOLUME_AXES_ATTR",
    "CalibrationState",
    "CalibrationVariable",
    "ConeBeam",
    "ConeGeometry",
    "Detector",
    "Geometry",
    "GeometryState",
    "Grid",
    "LaminographyGeometry",
    "ParallelGeometry",
    "RotationAxisGeometry",
    "axes_to_perm",
    "axis_pose_stack",
    "axis_unit_from_rotations",
    "beam_of",
    "build_calibrated_geometry_metadata_patch",
    "build_calibration_manifest",
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
    "read_geometry_json",
    "read_pose_params_csv",
    "stack_view_poses",
    "transpose_volume",
    "validate_calibration_gauges",
]
