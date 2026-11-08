"""Joint solver and ray geometry used by the public scan workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomojax.alignment._joint import fit_poses
from tomojax.core.geometry.cone import cone_parts

if TYPE_CHECKING:
    from tomojax.geometry import ScanGeometry


def cone_frames(geometry: ScanGeometry) -> np.ndarray | None:
    """Lab detector centre, u, v and source for each view; None for parallel rays."""
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


__all__ = ["cone_frames", "fit_poses"]
