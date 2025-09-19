from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Sequence

import numpy as np

from .base import Grid, Detector, Geometry
from .transforms import rot_axis_angle


def laminography_tilt_matrix(tilt_deg: float, tilt_about: str) -> np.ndarray:
    tilt = float(np.deg2rad(tilt_deg))
    if abs(tilt) < 1e-12:
        return np.eye(3, dtype=np.float64)
    if tilt_about == "x":
        c, s = np.cos(tilt), np.sin(tilt)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=np.float64)
    if tilt_about == "z":
        c, s = np.cos(tilt), np.sin(tilt)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    raise ValueError("tilt_about must be 'x' or 'z'")


def laminography_axis_unit(tilt_deg: float, tilt_about: str) -> np.ndarray:
    R_tilt = laminography_tilt_matrix(tilt_deg, tilt_about)
    ax = R_tilt @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return ax / (np.linalg.norm(ax) + 1e-12)

def _align_u_to_v(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Small utility: rotation matrix mapping unit vector u to unit vector v."""
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.dot(u, v))
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-12:
        # 180-degree: pick any axis orthogonal to u
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, u)) > 0.9:
            tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        k = tmp - np.dot(tmp, u) * u
        k = k / (np.linalg.norm(k) + 1e-12)
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
        return np.eye(3) + 2.0 * (K @ K)
    k = np.cross(u, v)
    s = float(np.linalg.norm(k))
    k = k / (s + 1e-12)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3) + K + ((1.0 - c) / (s * s + 1e-12)) * (K @ K)

@dataclass
class LaminographyGeometry:
    """Laminography geometry with tilted rotation axis (parallel-ray model).

    - Beam direction is fixed along +y (as in ParallelGeometry).
    - Rotation axis is tilted away from +z by `tilt_deg` within a chosen plane.
      With `tilt_about='x'` the axis leans towards +y (beam-aligned laminography);
      with `tilt_about='z'` the axis leans towards +x. Each view rotates the
      object around this axis by angle `thetas_deg[i]`.
    """

    grid: Grid
    detector: Detector
    thetas_deg: Sequence[float]
    tilt_deg: float = 30.0
    tilt_about: str = "x"  # or "z"

    def _tilt_matrix(self) -> np.ndarray:
        return laminography_tilt_matrix(self.tilt_deg, self.tilt_about)

    def _axis_unit(self) -> np.ndarray:
        return laminography_axis_unit(self.tilt_deg, self.tilt_about)

    def pose_for_view(self, i: int):
        """Object->world pose in a sample frame: rotate around object +y.

        We align the object's +y axis to the laminography axis once (S), then
        rotate around +y by view angle theta: R = S @ R_y(theta).
        """
        theta = float(np.deg2rad(self.thetas_deg[i]))
        axis = self._axis_unit()
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        S = _align_u_to_v(ey, axis)
        Ry = rot_axis_angle(ey, theta)[:3, :3]
        R = S @ Ry
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
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
