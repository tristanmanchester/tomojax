"""Exact zero-extended trilinear line integrals, with a differentiable JAX reference.

Split each ray at all voxel-centre planes. Within each interval density is a
cubic polynomial in physical distance, so two-point Gauss-Legendre integration
is exact up to floating-point rounding. A bounded traversal avoids allocating
a ray-by-plane intersection table.
"""

from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp

from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.projector import get_detector_grid_device


def _interpolate(volume, points, *, gradient=False):
    indices = jnp.floor(points).astype(jnp.int32)
    fraction = points - indices
    value = jnp.zeros(points.shape[1], jnp.float32)
    derivative = jnp.zeros_like(points)
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                offsets = jnp.array([dx, dy, dz])[:, None]
                loc = indices + offsets
                weights = jnp.where(offsets == 1, fraction, 1 - fraction)
                weight = jnp.prod(weights, axis=0)
                valid = jnp.all((loc >= 0) & (loc < jnp.array(volume.shape)[:, None]), axis=0)
                sample = volume[
                    jnp.clip(loc[0], 0, volume.shape[0] - 1),
                    jnp.clip(loc[1], 0, volume.shape[1] - 1),
                    jnp.clip(loc[2], 0, volume.shape[2] - 1),
                ]
                value = value + jnp.where(valid, weight * sample, 0)
                if gradient:
                    signs = 2 * offsets - 1
                    dw = signs * jnp.stack(
                        [weights[1] * weights[2], weights[0] * weights[2], weights[0] * weights[1]]
                    )
                    derivative = derivative + jnp.where(valid, dw * sample, 0)
    return (value, derivative) if gradient else value


@partial(jax.jit, static_argnames=("grid", "detector", "derivatives"))
def integrate(pose, grid, detector, volume, det_grid, *, derivatives=False):
    """Return one projection and optionally its ray-base/direction derivatives."""
    u, v = det_grid
    world = jnp.stack([u, jnp.zeros_like(u), v])
    inverse_rotation = pose[:3, :3].T
    voxel = jnp.array([grid.vx, grid.vy, grid.vz])[:, None]
    origin = jnp.array(grid_volume_origin(grid))[:, None]
    base = (
        jnp.matmul(inverse_rotation, world - pose[:3, 3, None], precision=jax.lax.Precision.HIGHEST)
        - origin
    ) / voxel
    direction = inverse_rotation[:, 1, None] / voxel
    moving = jnp.abs(direction) > 1e-12
    safe_direction = jnp.where(moving, direction, 1)
    lower = (-1 - base) / safe_direction
    upper = (jnp.array(volume.shape)[:, None] - base) / safe_direction
    inside = (base >= -1) & (base <= jnp.array(volume.shape)[:, None])
    first = jnp.where(moving, jnp.minimum(lower, upper), jnp.where(inside, -jnp.inf, jnp.inf))
    last = jnp.where(moving, jnp.maximum(lower, upper), jnp.where(inside, jnp.inf, -jnp.inf))
    entry, t_exit = jnp.max(first, axis=0), jnp.min(last, axis=0)
    valid = jnp.isfinite(entry) & jnp.isfinite(t_exit) & (t_exit > entry)
    entry, t_exit = jnp.where(valid, entry, 0), jnp.where(valid, t_exit, 0)
    start = base + direction * entry
    next_plane = jnp.where(direction > 0, jnp.floor(start) + 1, jnp.ceil(start) - 1)
    crossing = jnp.where(moving, (next_plane - base) / safe_direction, jnp.inf)
    period = jnp.where(moving, 1 / jnp.abs(safe_direction), jnp.inf)
    fractions = (0.5 - math.sqrt(3) / 6, 0.5 + math.sqrt(3) / 6)

    def step(carry, _):
        current, crossing, accumulated, grad_base, grad_direction = carry
        end = jnp.minimum(t_exit, jnp.min(crossing, axis=0))
        length = jnp.maximum(end - current, 0)
        for fraction in fractions:
            time = current + fraction * length
            points = base + direction * time
            if derivatives:
                density, gradient = _interpolate(volume, points, gradient=True)
                grad_base = grad_base + 0.5 * length * gradient
                grad_direction = grad_direction + 0.5 * length * time * gradient
            else:
                density = _interpolate(volume, points)
            accumulated = accumulated + 0.5 * length * density
        crossing = jnp.where(crossing <= end, crossing + period, crossing)
        return (end, crossing, accumulated, grad_base, grad_direction), None

    (_, _, result, grad_base, grad_direction), _ = jax.lax.scan(
        jax.checkpoint(step),
        (entry, crossing, jnp.zeros_like(entry), jnp.zeros_like(base), jnp.zeros_like(base)),
        None,
        length=sum(volume.shape) + 9,
    )
    image = result.reshape((detector.nv, detector.nu))
    return (image, grad_base, grad_direction) if derivatives else image


def forward_project_view_exact_T(pose, grid, detector, volume, *, det_grid=None):
    """Project one physical view using the exact trilinear-basis integral."""
    if det_grid is None:
        det_grid = get_detector_grid_device(detector)
    return integrate(pose, grid, detector, volume, det_grid)
