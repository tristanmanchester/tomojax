"""CUDA exact trilinear line integrals, transpose, and pose normal equations."""

from functools import cache, partial
from pathlib import Path

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from .cuda import initialize_buffer, launch_kernel
from .geometry.base import grid_volume_origin


def coefficients(poses, grid, detector):
    """Map integer detector coordinates and physical ray distance to voxel indices."""
    voxel = jnp.array([grid.vx, grid.vy, grid.vz], jnp.float32)
    origin = jnp.array(grid_volume_origin(grid), jnp.float32)
    u0 = detector.center[0] - (detector.nu - 1) * detector.du / 2
    v0 = detector.center[1] - (detector.nv - 1) * detector.dv / 2
    base = jnp.einsum(
        "vji,vj->vi",
        poses[:, :3, :3],
        jnp.array([u0, 0, v0], jnp.float32) - poses[:, :3, 3],
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


@cache
def _source(mode):
    source = Path(__file__).with_name("_trilinear_kernels.cu").read_text()
    return f"#define MODE {mode}\n" + source


def _launch(context, out, cf, data, *, shape, detector, mode):
    nv, nu = detector
    if mode == 1:
        initialize_buffer(out, stream=context.stream)
    launch_kernel(
        _source(mode),
        "integrate",
        options=("--fmad=false",),
        stream=context.stream,
        grid=(cf.shape[0] * -(-nu * nv // 128),),
        block=(128,),
        arguments=(cf, data, data, out, *map(np.int32, (*shape, nu, nv, cf.shape[0]))),
    )


def run(cf, data, grid, detector, *, mode="forward"):
    """Project, apply the exact transpose, or return seven ray-coordinate derivatives."""
    operation = {"forward": 0, "adjoint": 1, "jacobian": 2}[mode]
    n = cf.shape[0]
    shape = (grid.nx, grid.ny, grid.nz)
    output_shape = (
        shape[::-1]
        if operation == 1
        else (7, n, detector.nv, detector.nu)
        if operation == 2
        else (n, detector.nv, detector.nu)
    )
    return buffer_callback(
        partial(_launch, shape=shape, detector=(detector.nv, detector.nu), mode=operation),
        jax.ShapeDtypeStruct(output_shape, jnp.float32),
        vmap_method="sequential",
        command_buffer_compatible=True,
    )(cf, data)


def _launch_normals(context, outputs, cf, volume, directions, targets, weights, *, shape, detector):
    nv, nu = detector
    tiles = -(-nu * nv // 128)
    launch_kernel(
        _source(3),
        "normals",
        options=("--fmad=false",),
        stream=context.stream,
        grid=(cf.shape[0] * tiles,),
        block=(128,),
        arguments=(
            cf,
            volume,
            directions,
            targets,
            weights,
            *outputs,
            *map(np.int32, (*shape, nu, nv, tiles, cf.shape[0])),
        ),
    )


def normal_equations(cf, volume, directions, targets, weights, grid, detector):
    """Weighted per-view loss, five-parameter gradient/Hessian, and data residual."""
    n = cf.shape[0]
    tiles = -(-detector.nu * detector.nv // 128)
    partials, residual = buffer_callback(
        partial(
            _launch_normals, shape=(grid.nx, grid.ny, grid.nz), detector=(detector.nv, detector.nu)
        ),
        (
            jax.ShapeDtypeStruct((n, tiles, 21), jnp.float32),
            jax.ShapeDtypeStruct(targets.shape, jnp.float32),
        ),
        vmap_method="sequential",
        command_buffer_compatible=True,
    )(cf, volume, directions, targets, weights)
    totals = jnp.sum(partials, axis=1)
    rows, cols = jnp.triu_indices(5)
    hessian = jnp.zeros((n, 5, 5), jnp.float32)
    hessian = hessian.at[:, rows, cols].set(totals[:, 6:])
    hessian = hessian.at[:, cols, rows].set(totals[:, 6:])
    return totals[:, 0], totals[:, 1:6], hessian, residual
