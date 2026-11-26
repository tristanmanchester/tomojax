"""Joint solver and ray geometry used by the public scan workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomojax.alignment._geometry.detector_center import CenterSelection, opposite_samples
from tomojax.alignment._joint._fit import FitSchedule, fit_poses, fit_schedule
from tomojax.alignment._joint._parameters import parse_fit
from tomojax.core.geometry.cone import cone_parts

if TYPE_CHECKING:
    from tomojax.geometry import ScanGeometry


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
    "cone_frames",
    "fit_poses",
    "fit_schedule",
    "opposite_samples",
    "parse_fit",
]
