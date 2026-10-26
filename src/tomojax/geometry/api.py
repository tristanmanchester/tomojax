"""Expert geometry helpers: calibration state and detector grids.

The geometry classes and field-of-view helpers users call are exported by
:mod:`tomojax.geometry`; this module holds what alignment, IO and the solvers
build on.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from tomojax.core.geometry.cone import is_cone_beam, view_frames
from tomojax.geometry._calibration import (
    CalibrationState,
    CalibrationVariable,
    build_calibrated_geometry_metadata_patch,
    validate_calibration_gauges,
)
from tomojax.geometry._serialization import nominal_axis_unit_from_inputs
from tomojax.geometry._state import GeometryState
from tomojax.geometry.detector_grid import (
    detector_grid_from_calibration,
    detector_grid_from_geometry_inputs,
)

if TYPE_CHECKING:
    from tomojax.geometry._axis_geometry import axis_pose_stack, axis_unit_from_rotations


def __getattr__(name: str) -> Any:
    if name not in {"axis_pose_stack", "axis_unit_from_rotations"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("tomojax.geometry._axis_geometry"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "CalibrationState",
    "CalibrationVariable",
    "GeometryState",
    "axis_pose_stack",
    "axis_unit_from_rotations",
    "build_calibrated_geometry_metadata_patch",
    "detector_grid_from_calibration",
    "detector_grid_from_geometry_inputs",
    "is_cone_beam",
    "nominal_axis_unit_from_inputs",
    "validate_calibration_gauges",
    "view_frames",
]
