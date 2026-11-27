"""Detector visibility of the working image, in the geometry used for the fit."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from tomojax.core.geometry.base import grid_volume_origin

if TYPE_CHECKING:
    from ._operators import Operators

type _Array = np.ndarray[tuple[int, ...], np.dtype[np.float32]]


def _camera(ops: Operators, ids: np.ndarray) -> tuple[_Array, _Array, _Array]:
    """Numerators of the centered detector coordinates, in pixels, and their common divisor."""
    poses = np.asarray(ops.nominal, np.float32)[ids, :3].copy()
    d = ops.detector
    if ops.frames is None:
        roll = math.radians(ops.detector_roll)
        u = np.array([math.cos(roll), 0.0, math.sin(roll)], np.float32)
        v = np.array([-math.sin(roll), 0.0, math.cos(roll)], np.float32)
        # Cropping changes the working center in the detector's own axes.
        # Roll still turns about the original detector pivot.
        pu, pv = ops.detector_pivot
        poses[:, :, 3] -= np.array([pu, 0, pv], np.float32)
        horizontal, vertical = u @ poses, v @ poses
        horizontal[:, 3] += pu - d.center[0]
        vertical[:, 3] += pv - d.center[1]
        divisor = np.zeros_like(horizontal)
        divisor[:, 3] = 1
    else:
        frames = np.asarray(ops.frames, np.float32)[ids]
        center, u, v, source = (frames[:, j] for j in range(4))
        normal = np.cross(u, v)
        poses[:, :, 3] -= source
        divisor = (normal[:, None, :] @ poses)[:, 0]
        arm = center - source
        distance = np.sum(arm * normal, axis=1, keepdims=True, dtype=np.float32)

        def coordinate(axis: _Array) -> _Array:
            offset = np.sum(arm * axis, axis=1, keepdims=True, dtype=np.float32)
            return np.asarray(
                distance * (axis[:, None, :] @ poses)[:, 0] - offset * divisor, np.float32
            )

        horizontal, vertical = coordinate(u), coordinate(v)
    return horizontal / np.float32(d.du), vertical / np.float32(d.dv), divisor


def visibility_gradient(ops: Operators) -> _Array:
    """RMS change in detector footprint across the three forward voxel differences.

    A view weights continuation across its detector edge. Equally visible neighbors
    share the same support, so that view gives their difference no boundary penalty.
    """
    g, d = ops.grid, ops.detector
    x0, y0, z0 = grid_volume_origin(g)
    ids = np.linspace(0, ops.n - 1, math.ceil(math.sqrt(ops.n))).round().astype(int)
    x = np.float32(x0) + np.float32(g.vx) * np.arange(g.nx, dtype=np.float32)[None, :]
    y = np.float32(y0) + np.float32(g.vy) * np.arange(g.ny, dtype=np.float32)[:, None]
    z = np.float32(z0) + np.float32(g.vz) * np.arange(g.nz, dtype=np.float32)[:, None, None]
    energy = np.zeros(ops.shape, np.float32)
    horizontal, vertical, divisor = _camera(ops, ids)
    for i in range(len(ids)):
        u, v, den = horizontal[i, :], vertical[i, :], divisor[i, :]
        planes = [x * p.item(0) + y * p.item(1) + p.item(3) for p in (u, v, den)]
        for start in range(0, g.nz, 8):
            stop = min(start + 8, g.nz)
            slab = slice(start, min(stop + 1, g.nz))  # one more plane for the z difference
            scale = planes[2] + z[slab] * den.item(2)
            col = (planes[0] + z[slab] * u.item(2)) / scale
            row = (planes[1] + z[slab] * v.item(2)) / scale
            hit = np.clip((d.nu + 1) / 2 - np.abs(col), 0, 1)
            hit *= np.clip((d.nv + 1) / 2 - np.abs(row), 0, 1)
            length = stop - start
            out = energy[start:stop]
            out[:, :, :-1] += np.diff(hit[:length], axis=2) ** 2
            out[:, :-1] += np.diff(hit[:length], axis=1) ** 2
            dz = np.diff(hit, axis=0)
            out[: len(dz)] += dz**2
    return np.sqrt(energy / len(ids))
