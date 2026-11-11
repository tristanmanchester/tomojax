"""Scan geometry: grids, detectors, beams and field-of-view helpers.

The geometry classes import without JAX; the field-of-view helpers load it on
first use. Expert helpers for calibration state and axis order are in
``tomojax.geometry.api``.
"""

from __future__ import annotations

from importlib import import_module
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
    from tomojax.core.geometry.views import stack_view_poses

_LAZY = {
    "stack_view_poses": "tomojax.core.geometry.views",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
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
    "grid_volume_origin",
    "stack_view_poses",
]
