"""Public API for geometry metadata, axes, and field-of-view helpers."""

from tomojax.geometry._axes import (
    DISK_VOLUME_AXES,
    INTERNAL_VOLUME_AXES,
    VOLUME_AXES_ATTR,
    axes_to_perm,
    infer_disk_axes,
    is_shape_xyz,
    is_shape_zyx,
    transpose_volume,
)
from tomojax.geometry._fov import (
    RoiInfo,
    compute_roi,
    cylindrical_mask_xy,
    grid_from_detector_fov,
    grid_from_detector_fov_cube,
    grid_from_detector_fov_slices,
)
from tomojax.geometry._gauges import (
    CanonicalizedGeometry,
    GaugeReport,
    GaugeTransfer,
    canonicalize_geometry_gauges,
)
from tomojax.geometry._serialization import (
    GEOMETRY_STATE_SCHEMA_VERSION,
    geometry_state_from_dict,
    geometry_state_to_dict,
    read_geometry_json,
    read_pose_params_csv,
    write_geometry_json,
    write_pose_decomposition_csv,
    write_pose_params_csv,
)
from tomojax.geometry._state import (
    AcquisitionParameters,
    GeometryState,
    PoseParameters,
    ScalarParameter,
    SetupParameters,
)

__all__ = [
    "DISK_VOLUME_AXES",
    "GEOMETRY_STATE_SCHEMA_VERSION",
    "INTERNAL_VOLUME_AXES",
    "VOLUME_AXES_ATTR",
    "AcquisitionParameters",
    "CanonicalizedGeometry",
    "GaugeReport",
    "GaugeTransfer",
    "GeometryState",
    "PoseParameters",
    "RoiInfo",
    "ScalarParameter",
    "SetupParameters",
    "axes_to_perm",
    "canonicalize_geometry_gauges",
    "compute_roi",
    "cylindrical_mask_xy",
    "geometry_state_from_dict",
    "geometry_state_to_dict",
    "grid_from_detector_fov",
    "grid_from_detector_fov_cube",
    "grid_from_detector_fov_slices",
    "infer_disk_axes",
    "is_shape_xyz",
    "is_shape_zyx",
    "read_geometry_json",
    "read_pose_params_csv",
    "transpose_volume",
    "write_geometry_json",
    "write_pose_decomposition_csv",
    "write_pose_params_csv",
]
