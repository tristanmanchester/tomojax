"""Scan geometry: grids, detectors, beams and field-of-view helpers.

The geometry classes import without JAX; the field-of-view helpers load it on
first use. Expert helpers for calibration state and axis order are in
``tomojax.geometry.api``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from tomojax.core.geometry.views import stack_view_poses


def __getattr__(name: str) -> object:
    if name != "stack_view_poses":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from tomojax.core.geometry.views import stack_view_poses

    globals()[name] = stack_view_poses
    return stack_view_poses


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
    "grid_volume_origin",
    "stack_view_poses",
]
