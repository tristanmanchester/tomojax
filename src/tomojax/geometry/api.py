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
from tomojax.geometry._state import GeometryState, PoseParameters, ScalarParameter, SetupParameters

__all__ = [
    "DISK_VOLUME_AXES",
    "INTERNAL_VOLUME_AXES",
    "VOLUME_AXES_ATTR",
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
    "grid_from_detector_fov",
    "grid_from_detector_fov_cube",
    "grid_from_detector_fov_slices",
    "infer_disk_axes",
    "is_shape_xyz",
    "is_shape_zyx",
    "transpose_volume",
]
