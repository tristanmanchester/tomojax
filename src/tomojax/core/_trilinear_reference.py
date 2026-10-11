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


def _integrate_cell(volume, base, direction, current, length, derivatives):
    """Integrate both Gauss nodes using one load per shared cell corner."""
    times = [current + f * length for f in (0.5 - math.sqrt(3) / 6, 0.5 + math.sqrt(3) / 6)]
    midpoint = base + direction * (current + 0.5 * length)
    index = jnp.floor(midpoint).astype(jnp.int32)
    fractions = [base + direction * time - index for time in times]
    # Rounded nodes can cross a cell face on tiny intervals. Strict comparisons
    # retain the right-cell derivative at exact voxel centres; jnp.clip would
    # halve that derivative at its endpoints.
    fractions = [jnp.where(f < 0, 0, jnp.where(f > 1, 1, f)) for f in fractions]
    value = jnp.zeros_like(current)
    gb = jnp.zeros_like(base)
    gd = jnp.zeros_like(base)
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                offsets = jnp.array([dx, dy, dz], jnp.int32)[:, None]
                loc = index + offsets
                weights = [
                    jnp.where(offsets == 1, fraction, 1 - fraction) for fraction in fractions
                ]
                valid = jnp.all(
                    (loc >= 0) & (loc < jnp.array(volume.shape, jnp.int32)[:, None]), axis=0
                )
                sample = jnp.where(
                    valid,
                    volume[
                        jnp.clip(loc[0], 0, volume.shape[0] - 1),
                        jnp.clip(loc[1], 0, volume.shape[1] - 1),
                        jnp.clip(loc[2], 0, volume.shape[2] - 1),
                    ],
                    0,
                )
                value += (
                    0.5
                    * length
                    * (jnp.prod(weights[0], axis=0) + jnp.prod(weights[1], axis=0))
                    * sample
                )
                if derivatives:
                    signs = 2 * offsets - 1
                    dw = [
                        signs * jnp.stack([w[1] * w[2], w[0] * w[2], w[0] * w[1]]) for w in weights
                    ]
                    gb += 0.5 * length * (dw[0] + dw[1]) * sample
                    gd += 0.5 * length * (dw[0] * times[0] + dw[1] * times[1]) * sample
    return value, gb, gd


def _ray_coordinates(pose, grid, det_grid):
    u, v = (jnp.asarray(a, jnp.float32) for a in det_grid)
    world = jnp.stack([u, jnp.zeros_like(u), v])
    inverse_rotation = pose[:3, :3].T
    voxel = jnp.array([grid.vx, grid.vy, grid.vz], jnp.float32)[:, None]
    origin = jnp.array(grid_volume_origin(grid), jnp.float32)[:, None]
    base = (
        jnp.matmul(inverse_rotation, world - pose[:3, 3, None], precision=jax.lax.Precision.HIGHEST)
        - origin
    ) / voxel
    direction = inverse_rotation[:, 1, None] / voxel
    return base, direction


@partial(jax.jit, static_argnames=("grid", "detector", "derivatives"))
def integrate(pose, grid, detector, volume, det_grid, *, derivatives=False):
    """Return one projection and optionally its ray-base/direction derivatives."""
    base, direction = _ray_coordinates(pose, grid, det_grid)
    moving = jnp.abs(direction) > 1e-12
    safe_direction = jnp.where(moving, direction, 1)
    lower = (-1 - base) / safe_direction
    upper = (jnp.array(volume.shape, jnp.int32)[:, None] - base) / safe_direction
    inside = (base >= -1) & (base <= jnp.array(volume.shape, jnp.int32)[:, None])
    first = jnp.where(moving, jnp.minimum(lower, upper), jnp.where(inside, -jnp.inf, jnp.inf))
    last = jnp.where(moving, jnp.maximum(lower, upper), jnp.where(inside, jnp.inf, -jnp.inf))
    entry, t_exit = jnp.max(first, axis=0), jnp.min(last, axis=0)
    valid = jnp.isfinite(entry) & jnp.isfinite(t_exit) & (t_exit > entry)
    entry, t_exit = jnp.where(valid, entry, 0), jnp.where(valid, t_exit, 0)
    start = base + direction * entry
    next_plane = jnp.where(direction > 0, jnp.floor(start) + 1, jnp.ceil(start) - 1)
    crossing = jnp.where(moving, (next_plane - base) / safe_direction, jnp.inf)
    period = jnp.where(moving, 1 / jnp.abs(safe_direction), jnp.inf)

    def advance(carry):
        current, crossing, accumulated, grad_base, grad_direction = carry
        end = jnp.minimum(t_exit, jnp.min(crossing, axis=0))
        length = jnp.maximum(end - current, 0)
        density, gb, gd = _integrate_cell(volume, base, direction, current, length, derivatives)
        accumulated += density
        grad_base += gb
        grad_direction += gd
        crossing = jnp.where(crossing <= end, crossing + period, crossing)
        return end, crossing, accumulated, grad_base, grad_direction

    def chunk(carry, length):
        # Test once per group, not per interval: on CUDA, tiny conditional
        # launches can cost more than the interpolation they avoid. Both loops
        # remain statically bounded to support JVP, VJP and linear_transpose.
        return jax.lax.cond(
            jnp.any(carry[0] < t_exit),
            lambda state: jax.lax.fori_loop(0, length, lambda _, state: advance(state), state),
            lambda state: state,
            carry,
        )

    initial = (entry, crossing, jnp.zeros_like(entry), jnp.zeros_like(base), jnp.zeros_like(base))
    groups, remainder = divmod(sum(volume.shape) + 9, 16)
    state, _ = jax.lax.scan(
        jax.checkpoint(lambda carry, _: (chunk(carry, 16), None)),
        initial,
        None,
        length=groups,
    )
    if remainder:
        state = jax.checkpoint(lambda carry: chunk(carry, remainder))(state)
    _, _, result, grad_base, grad_direction = state
    image = result.reshape((detector.nv, detector.nu))
    return (image, grad_base, grad_direction) if derivatives else image


def project_pose_jacobian(pose, directions, grid, detector, volume, *, det_grid=None):
    """Integrate spatial derivatives once, then apply each pose tangent.

    The zero-extended trilinear field is continuous and vanishes at the support
    boundary. Thus moving cell/support endpoints cancel from the first
    derivative, leaving the integral of the field's spatial gradient.
    """
    if det_grid is None:
        det_grid = get_detector_grid_device(detector)
    image, grad_base, grad_direction = integrate(
        pose, grid, detector, volume, det_grid, derivatives=True
    )
    dbase, ddirection = jax.vmap(
        lambda tangent: jax.jvp(lambda t: _ray_coordinates(t, grid, det_grid), (pose,), (tangent,))[
            1
        ]
    )(directions)
    jacobian = jnp.sum(dbase * grad_base + ddirection * grad_direction, axis=1)
    return image, jacobian.reshape((directions.shape[0], detector.nv, detector.nu))


def forward_project_view_exact_T(pose, grid, detector, volume, *, det_grid=None):
    """Project one physical view using the exact trilinear-basis integral."""
    if det_grid is None:
        det_grid = get_detector_grid_device(detector)
    return integrate(pose, grid, detector, volume, det_grid)
