"""Parallel-beam CT geometry implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .base import Detector, Grid, PoseMatrix, RayPair, _parallel_detector_rays
from .transforms import rotz

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ParallelGeometry:
    """Parallel-beam CT geometry with axis-aligned detector.

    - World frame: detector lies in (x,z) plane; rays along +y.
    - ``angles`` are the view angles, in degrees. View ``i`` rotates the object
      around +z by ``phi = angles[i]``; pose_for_view returns
      T_world_from_obj = Rz(phi). At phi=0, T = I.

    Parameters
    ----------
    grid : Grid
        Reconstruction voxel grid.
    detector : Detector
        Detector dimensions, pitches and physical centre.
    angles : sequence of float
        View rotation angles in degrees, in acquisition order.
    """

    grid: Grid
    detector: Detector
    angles: Sequence[float]

    def pose_for_view(self, i: int) -> PoseMatrix:
        """Return the world-from-object pose for one view.

        Parameters
        ----------
        i : int
            Zero-based view index.

        Returns
        -------
        tuple of tuple of float
            Row-major 4 x 4 homogeneous world-from-object transform.

        Raises
        ------
        IndexError
            If the view index is outside the angle sequence.
        """
        phi = np.deg2rad(float(self.angles[i]))
        T = rotz(phi)
        return tuple(map(tuple, T))  # 4x4

    def rays_for_view(self, i: int) -> RayPair:
        """Return detector rays for one view.

        Parameters
        ----------
        i : int
            Zero-based view index.

        Returns
        -------
        tuple of callable
            Origin and unit-direction callbacks taking detector pixel indices (u, v).
        """
        del i
        return _parallel_detector_rays(self.grid, self.detector)
