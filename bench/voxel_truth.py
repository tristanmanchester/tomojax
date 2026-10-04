"""Independent FP64 line integrals of a zero-extended trilinear voxel field.

Split rays at every voxel-centre plane. Within each resulting cell the field
along a ray is cubic, so two-point Gauss-Legendre integration is exact apart
from floating-point arithmetic. No TomoJAX projector or CUDA runtime is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import map_coordinates

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from tomojax.core.geometry.base import Detector, Grid


def voxel_line_integrals(
    volume: ArrayLike,
    bases: ArrayLike,
    direction: ArrayLike,
    *,
    spacing: ArrayLike,
    origin: ArrayLike,
) -> np.ndarray:
    """Integrate complete rays ``base + t * direction`` in physical units."""
    volume = np.asarray(volume, dtype=np.float64)
    bases = np.asarray(bases, dtype=np.float64).reshape(-1, 3)
    direction = np.asarray(direction, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if volume.ndim != 3 or min(volume.shape) < 1:
        raise ValueError("volume must be a nonempty 3D field")
    if direction.shape != (3,) or not np.isfinite(direction).all() or not np.linalg.norm(direction):
        raise ValueError("direction must be a finite nonzero vector")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing must contain three finite positive lengths")
    if origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(bases).all():
        raise ValueError("ray bases and volume origin must be finite 3D coordinates")
    lower = origin - spacing
    upper = origin + np.asarray(volume.shape) * spacing
    near, far = np.full(len(bases), -np.inf), np.full(len(bases), np.inf)
    valid = np.ones(len(bases), dtype=bool)
    knots = []
    for axis in range(3):
        if abs(direction[axis]) < 1e-14 * np.linalg.norm(direction):
            valid &= (bases[:, axis] >= lower[axis]) & (bases[:, axis] <= upper[axis])
            continue
        planes = origin[axis] + np.arange(-1, volume.shape[axis] + 1) * spacing[axis]
        crossing = (planes[None, :] - bases[:, axis, None]) / direction[axis]
        knots.append(crossing)
        near = np.maximum(near, np.minimum(crossing[:, 0], crossing[:, -1]))
        far = np.minimum(far, np.maximum(crossing[:, 0], crossing[:, -1]))
    valid &= far > near
    near, far = np.where(valid, near, 0), np.where(valid, far, 0)
    parameters = np.sort(np.clip(np.concatenate(knots, axis=1), near[:, None], far[:, None]))
    half_width = np.diff(parameters, axis=1) / 2
    midpoint = (parameters[:, 1:] + parameters[:, :-1]) / 2
    total = np.zeros(len(bases), dtype=np.float64)
    for node in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
        position = bases[:, None, :] + (midpoint + node * half_width)[..., None] * direction
        indices = ((position - origin) / spacing).reshape(-1, 3).T
        value = map_coordinates(
            volume, indices, order=1, mode="grid-constant", cval=0, prefilter=False
        )
        total += np.sum(value.reshape(midpoint.shape) * half_width, axis=1)
    return total * np.linalg.norm(direction)


def project_voxel_truth(
    volume: np.ndarray, poses: np.ndarray, grid: Grid, detector: Detector
) -> np.ndarray:
    """Independent detector sampling in the library's lab XZ / beam-Y convention."""
    from tomojax.core.geometry.base import grid_volume_origin

    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.det_center[0]
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv + detector.det_center[1]
    uu, vv = np.meshgrid(u, v)
    world = np.stack([uu.ravel(), np.zeros(uu.size), vv.ravel()], axis=-1)
    output = []
    for pose in np.asarray(poses, dtype=np.float64):
        inverse = pose[:3, :3].T
        bases = (world - pose[:3, 3]) @ inverse.T
        output.append(
            voxel_line_integrals(
                volume,
                bases,
                inverse[:, 1],
                spacing=(grid.vx, grid.vy, grid.vz),
                origin=grid_volume_origin(grid),
            ).reshape(detector.nv, detector.nu)
        )
    return np.asarray(output, dtype=np.float32)
