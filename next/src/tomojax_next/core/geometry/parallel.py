from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Sequence

import numpy as np

from .base import Grid, Detector, Geometry
from .transforms import rotz


@dataclass
class ParallelGeometry:
    """Parallel-beam CT geometry with axis-aligned detector.

    - World frame: detector lies in (x,z) plane; rays along +y.
    - Object pose per view is rotation around +z by angle phi (radians).
    """

    grid: Grid
    detector: Detector
    thetas_deg: Sequence[float]

    def pose_for_view(self, i: int) -> Tuple[Tuple[float, ...], ...]:
        phi = float(np.deg2rad(self.thetas_deg[i]))
        T = rotz(phi)
        return tuple(map(tuple, T))  # 4x4

    def rays_for_view(self, i: int):
        nu, nv = int(self.detector.nu), int(self.detector.nv)
        du, dv = float(self.detector.du), float(self.detector.dv)
        cx, cz = float(self.detector.det_center[0]), float(self.detector.det_center[1])

        y0 = (
            float(self.grid.vol_origin[1])
            if self.grid.vol_origin is not None
            else -((self.grid.ny / 2.0) - 0.5) * float(self.grid.vy)
        )

        def origin_fn(u: int, v: int) -> Tuple[float, float, float]:
            x = (u - (nu / 2.0 - 0.5)) * du + cx
            z = (v - (nv / 2.0 - 0.5)) * dv + cz
            return float(x), float(y0), float(z)

        def dir_fn(u: int, v: int) -> Tuple[float, float, float]:
            return (0.0, 1.0, 0.0)

        return origin_fn, dir_fn

