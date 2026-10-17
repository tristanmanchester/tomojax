"""Rotation-axis geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .base import Detector, Grid, PoseMatrix, RayPair, _parallel_detector_rays
from .transforms import align_u_to_v, rot_axis_angle

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    the world x/z detector plane. Only the per-view object pose changes: view
    ``i`` rotates the object about the axis by ``angles[i]``, in degrees.

    Parameters
    ----------
    grid : Grid
        Reconstruction voxel grid.
    detector : Detector
        Detector dimensions, pitches and physical centre.
    angles : sequence of float
        View rotation angles in degrees, in acquisition order.
    axis_unit_lab : sequence of float
        Nonzero finite lab-frame rotation-axis vector, normalized on construction.

    Raises
    ------
    ValueError
        If the rotation-axis vector is not finite, nonzero and three-dimensional.
    """

    grid: Grid
    detector: Detector
    angles: Sequence[float]
    axis_unit_lab: Sequence[float]

    def __post_init__(self) -> None:
        self.axis_unit_lab = normalize_axis_unit(self.axis_unit_lab)

    def _axis_unit(self) -> np.ndarray:
        return np.asarray(normalize_axis_unit(self.axis_unit_lab), dtype=np.float64)

    def pose_for_view(self, i: int) -> PoseMatrix:
        """Return the world-from-object pose for one view.

        Degree conversion uses host precision for lists and saved FP32 arrays alike.

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
        angle_deg = float(self.angles[i])
        theta = float(np.deg2rad(angle_deg))
        axis = self._axis_unit()
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        S = align_u_to_v(ez, axis)
        Rz = rot_axis_angle(ez, theta)[:3, :3]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = S @ Rz
        return tuple(map(tuple, T))

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
