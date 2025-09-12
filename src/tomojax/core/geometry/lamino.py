from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Sequence

import numpy as np

from .base import Grid, Detector, Geometry
from .transforms import rot_axis_angle


@dataclass
class LaminographyGeometry:
    """Laminography geometry with tilted rotation axis (parallel-ray model).

    - Beam direction is fixed along +y (as in ParallelGeometry).
    - Rotation axis is tilted by `tilt_deg` around the chosen axis (x or z),
      forming a unit vector `axis_unit` in world coordinates. Each view rotates
      the object around this axis by angle `thetas_deg[i]`.
    """

    grid: Grid
    detector: Detector
    thetas_deg: Sequence[float]
    tilt_deg: float = 30.0
    tilt_about: str = "x"  # or "z"

    def _axis_unit(self) -> np.ndarray:
        tilt = np.deg2rad(self.tilt_deg)
        if self.tilt_about == "x":
            # start from +z, tilt towards +x by tilt degrees
            ax = np.array([np.sin(tilt), 0.0, np.cos(tilt)], dtype=np.float64)
        elif self.tilt_about == "z":
            # start from +y, tilt towards +z by tilt degrees (alternative)
            ax = np.array([0.0, np.sin(tilt), np.cos(tilt)], dtype=np.float64)
        else:
            raise ValueError("tilt_about must be 'x' or 'z'")
        return ax / (np.linalg.norm(ax) + 1e-12)

    def pose_for_view(self, i: int):
        theta = float(np.deg2rad(self.thetas_deg[i]))
        axis = self._axis_unit()
        T = rot_axis_angle(axis, theta)
        return tuple(map(tuple, T))  # 4x4

    def rays_for_view(self, i: int):
        # Same detector plane and beam direction as parallel CT for now
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

