"""Joint solver and ray geometry used by the public scan workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomojax.alignment._geometry.center_focus import angular_focus
from tomojax.alignment._geometry.center_registration import paired_center
from tomojax.alignment._geometry.detector_center import CenterSelection, opposite_samples
from tomojax.alignment._joint._fit import FitSchedule, fit_poses, fit_schedule
from tomojax.alignment._joint._parameters import parse_fit
from tomojax.core.geometry.cone import cone_parts
from tomojax.geometry import ParallelGeometry

if TYPE_CHECKING:
    import jax

    from tomojax.geometry import ScanGeometry


def initial_center_change(
    geometry: ScanGeometry, projections: np.ndarray | jax.Array, positions: np.ndarray | None
) -> float:
    """Measure the starting correction in native pixels using opposing ray constraints."""
    if type(geometry) is not ParallelGeometry or geometry.detector_roll:
        return 0.0
    focus = angular_focus(geometry)
    selection = opposite_samples(geometry, focus=focus)
    if focus is not None:
        center, _ = selection.focused(projections, positions)
    else:
        center = paired_center(selection, projections, positions)
    return center - geometry.detector.center[0] / geometry.detector.du


def cone_frames(geometry: ScanGeometry) -> np.ndarray | None:
    """Lab detector center, u, v and source for each view; None for parallel rays."""
    parts = cone_parts(geometry)
    if parts is None:
        return None
    views = len(geometry.angles)
    return np.concatenate(
        [
            np.broadcast_to(
                np.stack((*beam.detector_frame(detector), beam.source())),
                (views if count is None else count, 4, 3),
            )
            for count, beam, detector in parts
        ]
    )


__all__ = [
    "CenterSelection",
    "FitSchedule",
    "angular_focus",
    "cone_frames",
    "fit_poses",
    "fit_schedule",
    "initial_center_change",
    "opposite_samples",
    "parse_fit",
]
