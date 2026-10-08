"""Exact trilinear integration, matched atomic adjoint and pose normals on CUDA.

Each ray carries only its next three voxel-plane intersections. Pose derivatives
integrate the spatial gradient and its first distance moment alongside density;
continuity cancels internal interval-boundary terms. No ray-by-sample tape or
full projection Jacobian is retained by the fused normal-equation kernel.
"""

from __future__ import annotations

from functools import partial
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.pallas._pallas_sampling import _trilinear_atomic_add


def coefficients(poses, grid, detector):
    """Map integer detector coordinates and physical ray distance to voxel indices."""
    voxel = jnp.array([grid.vx, grid.vy, grid.vz])
    origin = jnp.array(grid_volume_origin(grid))
    u0 = detector.center[0] - (detector.nu - 1) * detector.du / 2
    v0 = detector.center[1] - (detector.nv - 1) * detector.dv / 2
    base = jnp.einsum(
        "vji,vj->vi",
        poses[:, :3, :3],
        jnp.array([u0, 0, v0]) - poses[:, :3, 3],
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.stack(
        [
            (base - origin) / voxel,
            poses[:, 0, :3] * detector.du / voxel,
            poses[:, 2, :3] * detector.dv / voxel,
            poses[:, 1, :3] / voxel,
        ],
        axis=1,
    )


def _kernel(
    cf_ref,
    volume_ref,
    image_ref,
    _init_ref,
    out_ref,
    *,
    shape,
    nu,
    nv,
    nviews,
    block,
    mode,
    interpret,
):
    view = pl.program_id(0)
    ray = pl.program_id(1) * block + jnp.arange(block, dtype=jnp.int32)
    pixel_u, pixel_v = ray % nu, ray // nu
    valid_pixel = ray < nu * nv
    q, d = [], []
    for axis in range(3):

        def cf(row, axis=axis):
            return plt.load(cf_ref.at[view, jnp.int32(row), jnp.int32(axis)])

        q.append(cf(0) + cf(1) * pixel_u + cf(2) * pixel_v)
        d.append(cf(3))
    first, last, safe_d = [], [], []
    for axis in range(3):
        moving = jnp.abs(d[axis]) > 1e-12
        safe = jnp.where(moving, d[axis], 1)
        safe_d.append(safe)
        a, b = (-1 - q[axis]) / safe, (shape[axis] - q[axis]) / safe
        inside = (q[axis] >= -1) & (q[axis] <= shape[axis])
        first.append(jnp.where(moving, jnp.minimum(a, b), jnp.where(inside, -jnp.inf, jnp.inf)))
        last.append(jnp.where(moving, jnp.maximum(a, b), jnp.where(inside, jnp.inf, -jnp.inf)))
    entry = jnp.maximum(jnp.maximum(first[0], first[1]), first[2])
    t_exit = jnp.minimum(jnp.minimum(last[0], last[1]), last[2])
    valid = valid_pixel & jnp.isfinite(entry) & jnp.isfinite(t_exit) & (t_exit > entry)
    entry, t_exit = jnp.where(valid, entry, 0), jnp.where(valid, t_exit, 0)
    crossing, periods = [], []
    for axis in range(3):
        start = q[axis] + d[axis] * entry
        plane = jnp.where(d[axis] > 0, jnp.floor(start) + 1, jnp.ceil(start) - 1)
        moving = jnp.abs(d[axis]) > 1e-12
        crossing.append(jnp.where(moving, (plane - q[axis]) / safe_d[axis], jnp.inf))
        periods.append(jnp.where(moving, 1 / jnp.abs(safe_d[axis]), jnp.inf))
    adjoint = mode == "adjoint"
    derivatives = mode in ("jacobian", "normal")
    if adjoint:
        ray_value = plt.load(image_ref.at[view * nu * nv + ray], mask=valid_pixel, other=0)
    zero = jnp.zeros((block,), jnp.float32)
    fractions = (0.5 - math.sqrt(3) / 6, 0.5 + math.sqrt(3) / 6)

    def body(carry):
        iteration, current, tx, ty, tz, prediction, g0, g1, g2, h0, h1, h2 = carry
        end = jnp.minimum(t_exit, jnp.minimum(jnp.minimum(tx, ty), tz))
        length = jnp.maximum(end - current, 0)
        active = valid & (length > 0)
        gs, hs = [g0, g1, g2], [h0, h1, h2]
        for fraction in fractions:
            time = current + fraction * length
            point = [q[a] + d[a] * time for a in range(3)]
            if adjoint:
                _trilinear_atomic_add(
                    out_ref,
                    ray_value * 0.5 * length,
                    *point,
                    nx=shape[0],
                    ny=shape[1],
                    nz=shape[2],
                    active=active,
                    interpret=interpret,
                )
            else:
                index = [jnp.floor(p).astype(jnp.int32) for p in point]
                frac = [p - i for p, i in zip(point, index, strict=False)]
                density, gradient = zero, [zero, zero, zero]
                for dx in range(2):
                    for dy in range(2):
                        for dz in range(2):
                            offsets = (dx, dy, dz)
                            loc = [i + o for i, o in zip(index, offsets, strict=False)]
                            weights = [
                                f if o else 1 - f for f, o in zip(frac, offsets, strict=False)
                            ]
                            mask = active
                            for a in range(3):
                                mask = mask & (loc[a] >= 0) & (loc[a] < shape[a])
                            position = (loc[0] * shape[1] + loc[1]) * shape[2] + loc[2]
                            sample = plt.load(volume_ref.at[position], mask=mask, other=0)
                            density = density + weights[0] * weights[1] * weights[2] * sample
                            if derivatives:
                                for a in range(3):
                                    gradient[a] = (
                                        gradient[a]
                                        + (2 * offsets[a] - 1)
                                        * weights[(a + 1) % 3]
                                        * weights[(a + 2) % 3]
                                        * sample
                                    )
                prediction = prediction + 0.5 * length * density
                if derivatives:
                    for a in range(3):
                        gs[a] = gs[a] + 0.5 * length * gradient[a]
                        hs[a] = hs[a] + 0.5 * length * time * gradient[a]
        crossing = [
            jnp.where(t <= end, t + dt, t) for t, dt in zip((tx, ty, tz), periods, strict=False)
        ]
        return (iteration + 1, end, *crossing, prediction, *gs, *hs)

    def condition(carry):
        return (carry[0] < sum(shape) + 9) & (jnp.max((carry[1] < t_exit).astype(jnp.int32)) > 0)

    result = jax.lax.while_loop(
        condition, body, (jnp.int32(0), entry, *crossing, zero, zero, zero, zero, zero, zero, zero)
    )
    if mode == "normal":
        return view, ray, pixel_u, pixel_v, valid_pixel, result[5:]
    if not adjoint:
        outputs = result[5:] if derivatives else (result[5],)
        for i, value in enumerate(outputs):
            plt.store(
                out_ref.at[i * nviews * nu * nv + view * nu * nv + ray], value, mask=valid_pixel
            )
    return None


def run(cf, data, grid, detector, *, mode="forward", interpret=False):
    """Execute a projection, matched adjoint, or diagnostic ray Jacobian."""
    n = cf.shape[0]
    rays = n * detector.nu * detector.nv
    shape = (grid.nx, grid.ny, grid.nz)
    size = math.prod(shape)
    adjoint = mode == "adjoint"
    count = size if adjoint else rays * (7 if mode == "jacobian" else 1)
    initial = jnp.zeros((size if adjoint else 1,), jnp.float32)
    call = pl.pallas_call(
        partial(
            _kernel,
            shape=shape,
            nu=detector.nu,
            nv=detector.nv,
            nviews=n,
            block=128,
            mode=mode,
            interpret=interpret,
        ),
        out_shape=jax.ShapeDtypeStruct((count,), jnp.float32),
        grid=(n, math.ceil(detector.nu * detector.nv / 128)),
        input_output_aliases={3: 0} if adjoint else {},
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=4),
        name=f"exact_trilinear_{mode}",
    )
    dummy = jnp.zeros((1,), jnp.float32)
    result = call(
        cf, dummy if adjoint else data.ravel(), data.ravel() if adjoint else dummy, initial
    )
    output_shape = (
        shape
        if adjoint
        else (
            (7, n, detector.nv, detector.nu)
            if mode == "jacobian"
            else (n, detector.nv, detector.nu)
        )
    )
    return result.reshape(output_shape)


def _normal_kernel(
    cf_ref,
    volume_ref,
    directions_ref,
    targets_ref,
    weights_ref,
    partials_ref,
    residual_ref,
    *,
    shape,
    nu,
    nv,
    nviews,
    block,
    interpret,
):
    view, ray, u, v, valid, sums = _kernel(
        cf_ref,
        volume_ref,
        targets_ref,
        None,
        partials_ref,
        shape=shape,
        nu=nu,
        nv=nv,
        nviews=nviews,
        block=block,
        mode="normal",
        interpret=interpret,
    )
    prediction, *derivatives = sums
    gb, gd = derivatives[:3], derivatives[3:]
    location = view * nu * nv + ray
    target = plt.load(targets_ref.at[location], mask=valid, other=0)
    weight = plt.load(weights_ref.at[location], mask=valid, other=0)
    residual = weight * (prediction - target)
    columns = []
    for dof in range(5):
        column = jnp.zeros_like(prediction)
        for axis in range(3):

            def derivative(row, dof=dof, axis=axis):
                return plt.load(
                    directions_ref.at[view, jnp.int32(dof), jnp.int32(row), jnp.int32(axis)]
                )

            column = (
                column
                + gb[axis] * (derivative(0) + u * derivative(1) + v * derivative(2))
                + gd[axis] * derivative(3)
            )
        columns.append(weight * column)
    values = [0.5 * jnp.sum(residual * residual)]
    values.extend(jnp.sum(column * residual) for column in columns)
    values.extend(jnp.sum(columns[i] * columns[j]) for i in range(5) for j in range(i, 5))
    for i, value in enumerate(values):
        plt.store(partials_ref.at[view, pl.program_id(1), jnp.int32(i)], value)
    plt.store(residual_ref.at[location], weight * residual, mask=valid)


def normal_equations(cf, volume, directions, targets, weights, grid, detector, *, interpret=False):
    """Return weighted per-view loss, five-parameter gradient/Hessian and data residual."""
    n = cf.shape[0]
    block = 128
    tiles = math.ceil(detector.nu * detector.nv / block)
    call = pl.pallas_call(
        partial(
            _normal_kernel,
            shape=(grid.nx, grid.ny, grid.nz),
            nu=detector.nu,
            nv=detector.nv,
            nviews=n,
            block=block,
            interpret=interpret,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((n, tiles, 21), jnp.float32),
            jax.ShapeDtypeStruct((n * detector.nu * detector.nv,), jnp.float32),
        ),
        grid=(n, tiles),
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=4),
        name="exact_trilinear_pose_normals",
    )
    partials, residual = call(cf, volume.ravel(), directions, targets.ravel(), weights.ravel())
    totals = jnp.sum(partials, axis=1)
    hessian = jnp.zeros((n, 5, 5), jnp.float32)
    index = 6
    for i in range(5):
        for j in range(i, 5):
            hessian = hessian.at[:, i, j].set(totals[:, index])
            hessian = hessian.at[:, j, i].set(totals[:, index])
            index += 1
    return totals[:, 0], totals[:, 1:6], hessian, residual.reshape(targets.shape)
