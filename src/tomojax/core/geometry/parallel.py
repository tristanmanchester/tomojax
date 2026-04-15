from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import Detector, Grid, PoseMatrix, RayPair, _parallel_detector_rays
from .transforms import rotz


@dataclass
class ParallelGeometry:
    """Parallel-beam CT geometry with axis-aligned detector.

    - World frame: detector lies in (x,z) plane; rays along +y.
    - Object pose per view is rotation around +z by angle phi (radians).
      pose_for_view returns T_world_from_obj = Rz(phi). At theta=0, T = I.
    """

    grid: Grid
    detector: Detector
    thetas_deg: Sequence[float]

    def pose_for_view(self, i: int) -> PoseMatrix:
        phi = float(np.deg2rad(self.thetas_deg[i]))
        T = rotz(phi)
        return tuple(map(tuple, T))  # 4x4

    def rays_for_view(self, i: int) -> RayPair:
        return _parallel_detector_rays(self.grid, self.detector)
