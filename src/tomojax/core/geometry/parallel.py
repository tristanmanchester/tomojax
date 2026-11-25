"""Parallel-beam CT geometry implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Angles, Detector, Grid, PoseMatrix, RayPair, parallel_detector_rays
from .transforms import rotz


@dataclass
class ParallelGeometry:
    """Parallel-beam CT geometry.

    - World frame: detector lies in (x,z) plane; rays along +y.
    - ``angles`` are the view angles, in degrees. View ``i`` rotates the object
      around +z by ``phi = angles[i]``; pose_matrix returns
      T_world_from_obj = Rz(phi). At phi=0, T = I.

    Parameters
    ----------
    grid : Grid
        Reconstruction voxel grid.
    detector : Detector
        Detector dimensions, pitches and physical centre.
    angles : sequence of float
        View rotation angles in degrees, in acquisition order.
    detector_roll : float, optional
        Roll about the detector normal, in degrees; positive turns u towards v.
    """

    grid: Grid
    detector: Detector
    angles: Angles
    detector_roll: float = 0.0

    def pose_matrix(self, i: int) -> PoseMatrix:
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
        return parallel_detector_rays(self.grid, self.detector, self.detector_roll)
