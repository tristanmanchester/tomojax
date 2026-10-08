"""Voxel-driven CUDA backprojection for filtered parallel-beam projections."""

from __future__ import annotations

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin


def _backproject_filtered_kernel(
    poses_ref: Any,
    projections_ref: Any,
    *refs: Any,
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
    voxel: tuple[float, float, float],
    detector_shape: tuple[int, int],
    detector_spacing: tuple[float, float],
    detector_center: tuple[float, float],
    n_views: int,
    block_size: int,
    z_integer: bool,
) -> None:
    # With an accumulator, refs is (accumulator, aliased output).
    acc_ref, out_ref = refs if len(refs) == 2 else (None, refs[0])
    nx, ny, nz = shape
    nv, nu = detector_shape
    du, dv = detector_spacing
    flat = pl.program_id(0) * block_size + jnp.arange(block_size, dtype=jnp.int32)
    valid = flat < nx * ny * nz
    ix, iy, iz = flat // (ny * nz), (flat // nz) % ny, flat % nz
    x = ix.astype(jnp.float32) * voxel[0] + origin[0]
    y = iy.astype(jnp.float32) * voxel[1] + origin[1]
    z = iz.astype(jnp.float32) * voxel[2] + origin[2]
    u_offset, v_offset = (nu - 1) / 2.0, (nv - 1) / 2.0
    if z_integer:
        iv = jnp.floor((z - detector_center[1]) / dv + v_offset + 0.5).astype(jnp.int32)

    def body(view: jnp.ndarray, accum: jnp.ndarray) -> jnp.ndarray:
        def pose(row: int, col: int) -> jnp.ndarray:
            return plt.load(poses_ref.at[view, jnp.int32(row), jnp.int32(col)])

        world_x = pose(0, 0) * x + pose(0, 1) * y + pose(0, 3)
        if not z_integer:
            world_x += pose(0, 2) * z
        u = (world_x - detector_center[0]) / du + u_offset
        iu = jnp.floor(u).astype(jnp.int32)
        wu = u - iu.astype(jnp.float32)

        def sample(u_idx: jnp.ndarray, v_idx: jnp.ndarray) -> jnp.ndarray:
            mask = valid & (u_idx >= 0) & (u_idx < nu) & (v_idx >= 0) & (v_idx < nv)
            # z is the contiguous output coordinate. Storing filtered data as
            # (view, u, v) makes neighboring z lanes read neighboring addresses.
            return plt.load(projections_ref.at[view, u_idx, v_idx], mask=mask, other=0.0)

        if z_integer:
            value = sample(iu, iv) * (1.0 - wu) + sample(iu + 1, iv) * wu
        else:
            world_z = pose(2, 0) * x + pose(2, 1) * y + pose(2, 2) * z + pose(2, 3)
            v = (world_z - detector_center[1]) / dv + v_offset
            iv0 = jnp.floor(v).astype(jnp.int32)
            wv = v - iv0.astype(jnp.float32)
            row0 = sample(iu, iv0) * (1.0 - wu) + sample(iu + 1, iv0) * wu
            row1 = sample(iu, iv0 + 1) * (1.0 - wu) + sample(iu + 1, iv0 + 1) * wu
            value = row0 * (1.0 - wv) + row1 * wv
        return accum + value

    result = jax.lax.fori_loop(0, n_views, body, jnp.zeros((block_size,), dtype=jnp.float32))
    if acc_ref is not None:
        result = plt.load(acc_ref, mask=valid, other=0.0) + result
    plt.store(out_ref, result, mask=valid)


@functools.lru_cache(maxsize=32)
def _backprojection_call(
    grid: Grid,
    detector: Detector,
    n_views: int,
    block_size: int,
    *,
    z_integer: bool,
    interpret: bool,
    accumulate: bool = False,
) -> Any:
    count = grid.nx * grid.ny * grid.nz
    in_specs = [pl.no_block_spec, pl.no_block_spec]
    if accumulate:
        in_specs.append(pl.BlockSpec((block_size,), lambda block: (block,)))
    return pl.pallas_call(
        functools.partial(
            _backproject_filtered_kernel,
            shape=(grid.nx, grid.ny, grid.nz),
            origin=grid_volume_origin(grid),
            voxel=(grid.vx, grid.vy, grid.vz),
            detector_shape=(detector.nv, detector.nu),
            detector_spacing=(detector.du, detector.dv),
            detector_center=detector.center,
            n_views=n_views,
            block_size=block_size,
            z_integer=z_integer,
        ),
        out_shape=jax.ShapeDtypeStruct((count,), jnp.float32),
        grid=((count + block_size - 1) // block_size,),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((block_size,), lambda block: (block,)),
        input_output_aliases={2: 0} if accumulate else {},
        compiler_params=plt.CompilerParams(num_warps=4),
        interpret=interpret,
        name="tomojax_fbp_voxel_backprojection",
    )


def backproject_filtered_pallas(
    poses: jnp.ndarray,
    filtered: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    z_integer: bool,
    interpret: bool = False,
    block_size: int = 512,
    accumulate: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sum filtered views per voxel, avoiding atomics and intermediate volumes.

    ``accumulate`` is added to the result in place.
    """
    call = _backprojection_call(
        grid,
        detector,
        int(poses.shape[0]),
        block_size,
        z_integer=z_integer,
        interpret=interpret,
        accumulate=accumulate is not None,
    )
    operands = [poses, jnp.transpose(filtered, (0, 2, 1))]
    if accumulate is not None:
        operands.append(accumulate.reshape(-1))
    return call(*operands).reshape((grid.nx, grid.ny, grid.nz))
