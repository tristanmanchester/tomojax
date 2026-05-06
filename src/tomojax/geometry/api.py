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

__all__ = [
    "DISK_VOLUME_AXES",
    "INTERNAL_VOLUME_AXES",
    "VOLUME_AXES_ATTR",
    "RoiInfo",
    "axes_to_perm",
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
