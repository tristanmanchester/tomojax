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
    ConeSegments,
    Detector,
    Geometry,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
    ScanGeometry,
    beam_of,
    grid_volume_origin,
)

if TYPE_CHECKING:
    from tomojax.geometry.api import (
        compute_roi,
        cylindrical_mask_xy,
        grid_from_detector_fov,
        grid_from_detector_fov_cube,
        grid_from_detector_fov_slices,
        stack_view_poses,
    )

_LAZY = frozenset(
    {
        "compute_roi",
        "cylindrical_mask_xy",
        "grid_from_detector_fov",
        "grid_from_detector_fov_cube",
        "grid_from_detector_fov_slices",
        "stack_view_poses",
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
    "ConeBeam",
    "ConeGeometry",
    "ConeSegments",
    "Detector",
    "Geometry",
    "Grid",
    "LaminographyGeometry",
    "ParallelGeometry",
    "RotationAxisGeometry",
    "ScanGeometry",
    "beam_of",
    "compute_roi",
    "cylindrical_mask_xy",
    "grid_from_detector_fov",
    "grid_from_detector_fov_cube",
    "grid_from_detector_fov_slices",
    "grid_volume_origin",
    "stack_view_poses",
]
