from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import Detector, Grid, PoseMatrix, RayPair, _parallel_detector_rays
from .lamino import _align_u_to_v
from .transforms import rot_axis_angle


def normalize_axis_unit(axis_unit_lab: Sequence[float]) -> tuple[float, float, float]:
    """Return a validated unit axis vector in lab/world coordinates."""
    axis = np.asarray(axis_unit_lab, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError(f"axis_unit_lab must have shape (3,), got {axis.shape}")
    if not np.isfinite(axis).all():
        raise ValueError("axis_unit_lab must contain finite values")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("axis_unit_lab must be non-zero")
    return tuple(float(v) for v in axis / norm)


@dataclass
class RotationAxisGeometry:
    """Parallel-ray geometry with an arbitrary lab-frame rotation axis direction.

    The detector and beam model are unchanged from the standard parallel-ray model:
    rays remain fixed in the world frame along +y and detector coordinates live in
    the world x/z detector plane. Only the per-view object pose changes.
    """

    grid: Grid
    detector: Detector
    thetas_deg: Sequence[float]
    axis_unit_lab: Sequence[float]

    def __post_init__(self) -> None:
        self.axis_unit_lab = normalize_axis_unit(self.axis_unit_lab)

    def _axis_unit(self) -> np.ndarray:
        return np.asarray(normalize_axis_unit(self.axis_unit_lab), dtype=np.float64)

    def pose_for_view(self, i: int) -> PoseMatrix:
        theta = float(np.deg2rad(self.thetas_deg[i]))
        axis = self._axis_unit()
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        S = _align_u_to_v(ez, axis)
        Rz = rot_axis_angle(ez, theta)[:3, :3]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = S @ Rz
        return tuple(map(tuple, T))

    def rays_for_view(self, i: int) -> RayPair:
        return _parallel_detector_rays(self.grid, self.detector)
