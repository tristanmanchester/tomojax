"""Voxel-driven CUDA backprojection for filtered parallel-beam projections."""

from __future__ import annotations

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin


def _voxel_coordinates(shape, origin, voxel, block_size, block_z):
    nx, ny, nz = shape
    lanes = pl.program_id(0) * block_size + jnp.arange(block_size, dtype=jnp.int32)
    if block_z:
        # Share lateral geometry across contiguous detector rows instead of
        # repeating it independently for each voxel in a flattened tile.
        iz = (pl.program_id(1) * block_z + jnp.arange(block_z, dtype=jnp.int32))[None, :]
        ix, iy = (lanes // ny)[:, None], (lanes % ny)[:, None]
        valid = (lanes < nx * ny)[:, None] & (iz < nz)
    else:
        ix, iy, iz = lanes // (ny * nz), (lanes // nz) % ny, lanes % nz
        valid = lanes < nx * ny * nz
    x = ix.astype(jnp.float32) * voxel[0] + origin[0]
    y = iy.astype(jnp.float32) * voxel[1] + origin[1]
    z = iz.astype(jnp.float32) * voxel[2] + origin[2]
    return x, y, z, iz, valid


def _integer_detector_rows(z, iz, origin_z, center_v, dv, nv, block_z):
    if block_z:
        # Integer arithmetic exposes contiguous row loads to the compiler;
        # rounding floating z coordinates separately hides that relationship.
        first = round((origin_z - center_v) / dv + (nv - 1) / 2)
        return first + iz
    v_offset = ((nv - 1) % 2) / 2.0
    return jnp.floor((z - center_v) / dv + v_offset + 0.5).astype(jnp.int32) + (nv - 1) // 2


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
    block_z: int,
    z_mode: str,
) -> None:
    # With an accumulator, refs is (accumulator, aliased output).
    acc_ref, out_ref = refs if len(refs) == 2 else (None, refs[0])
    nv, nu = detector_shape
    du, dv = detector_spacing
    x, y, z, iz, valid = _voxel_coordinates(shape, origin, voxel, block_size, block_z)
    # Add integer pixel origins only after extracting the interpolation fractions.
    u_offset, v_offset = ((nu - 1) % 2) / 2.0, ((nv - 1) % 2) / 2.0
    if z_mode == "integer":
        iv = _integer_detector_rows(z, iz, origin[2], detector_center[1], dv, nv, block_z)
    elif z_mode == "fixed":
        world_z = plt.load(poses_ref.at[jnp.int32(0), jnp.int32(2), jnp.int32(2)]) * z + plt.load(
            poses_ref.at[jnp.int32(0), jnp.int32(2), jnp.int32(3)]
        )
        v = (world_z - detector_center[1]) / dv + v_offset
        lower_v = jnp.floor(v)
        fixed_wv = v - lower_v
        fixed_iv0 = lower_v.astype(jnp.int32) + (nv - 1) // 2

    def body(view: jnp.ndarray, accum: jnp.ndarray) -> jnp.ndarray:
        def pose(row: int, col: int) -> jnp.ndarray:
            return plt.load(poses_ref.at[view, jnp.int32(row), jnp.int32(col)])

        world_x = pose(0, 0) * x + pose(0, 1) * y + pose(0, 3)
        if z_mode == "general":
            world_x += pose(0, 2) * z
        u = (world_x - detector_center[0]) / du + u_offset
        # Keep the floating floor for fractions, without an integer round trip.
        lower_u = jnp.floor(u)
        wu = u - lower_u
        iu = lower_u.astype(jnp.int32) + (nu - 1) // 2

        def sample(u_idx: jnp.ndarray, v_idx: jnp.ndarray) -> jnp.ndarray:
            mask = valid & (u_idx >= 0) & (u_idx < nu) & (v_idx >= 0) & (v_idx < nv)
            # z is the contiguous output coordinate. Storing filtered data as
            # (view, u, v) makes neighboring z lanes read neighboring addresses.
            return plt.load(projections_ref.at[view, u_idx, v_idx], mask=mask, other=0.0)

        if z_mode == "integer":
            value = sample(iu, iv) * (1.0 - wu) + sample(iu + 1, iv) * wu
        else:
            if z_mode == "fixed":
                iv0, wv = fixed_iv0, fixed_wv
            else:
                world_z = pose(2, 0) * x + pose(2, 1) * y + pose(2, 2) * z + pose(2, 3)
                v = (world_z - detector_center[1]) / dv + v_offset
                lower_v = jnp.floor(v)
                wv = v - lower_v
                iv0 = lower_v.astype(jnp.int32) + (nv - 1) // 2
            row0 = sample(iu, iv0) * (1.0 - wu) + sample(iu + 1, iv0) * wu
            row1 = sample(iu, iv0 + 1) * (1.0 - wu) + sample(iu + 1, iv0 + 1) * wu
            value = row0 * (1.0 - wv) + row1 * wv
        return accum + value

    result = jax.lax.fori_loop(0, n_views, body, jnp.zeros(valid.shape, dtype=jnp.float32))
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
    z_mode: str,
    interpret: bool,
    accumulate: bool = False,
    block_z: int = 0,
) -> Any:
    count = grid.nx * grid.ny * grid.nz
    if block_z:
        shape = (grid.nx * grid.ny, grid.nz)
        launch = ((shape[0] + block_size - 1) // block_size, (grid.nz + block_z - 1) // block_z)
        spec = pl.BlockSpec((block_size, block_z), lambda xy, z: (xy, z))
    else:
        shape = (count,)
        launch = ((count + block_size - 1) // block_size,)
        spec = pl.BlockSpec((block_size,), lambda block: (block,))
    in_specs = [pl.no_block_spec, pl.no_block_spec]
    if accumulate:
        in_specs.append(spec)
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
            block_z=block_z,
            z_mode=z_mode,
        ),
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=launch,
        in_specs=in_specs,
        out_specs=spec,
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
    z_mode: str,
    interpret: bool = False,
    block_size: int | None = None,
    accumulate: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sum filtered views per voxel, avoiding atomics and intermediate volumes.

    ``z_mode='integer'`` needs aligned z-axis poses; ``'fixed'`` allows
    fractional rows and a common z translation. Both require rotations about
    z, without coupling to x or y. ``'general'`` handles arbitrary rigid poses.
    ``accumulate`` is added to the result in place. Compact
    filtered-stack and output working sets use one voxel per lane of the
    four-warp program. Large axial outputs amortize pose loads across more
    voxels. Aligned contiguous rows use a two-dimensional tile with shared
    lateral coordinates; fractional/general poses retain flattened tiles.
    """
    if z_mode not in {"integer", "fixed", "general"}:
        raise ValueError("FBP z_mode must be integer, fixed, or general")
    block_z = 0
    if block_size is None:
        output_bytes = 4 * grid.nx * grid.ny * grid.nz
        working_bytes = filtered.size * filtered.dtype.itemsize + output_bytes
        compact = working_bytes <= 64 * 1024**2
        if compact:
            block_size = 128
        elif z_mode != "general" and output_bytes >= 64 * 1024**2:
            block_size = 2048
            first = (grid_volume_origin(grid)[2] - detector.center[1]) / detector.dv
            first += (detector.nv - 1) / 2
            if (
                z_mode == "integer"
                and grid.nz % 4 == 0
                and detector.nv % 4 == 0
                and grid.vz == detector.dv
                and abs(first - round(first)) <= 1e-5
                and round(first) % 4 == 0
            ):
                block_z = min(16, 1 << (grid.nz - 1).bit_length())
                block_size //= block_z
        else:
            block_size = 512
    call = _backprojection_call(
        grid,
        detector,
        int(poses.shape[0]),
        block_size,
        z_mode=z_mode,
        interpret=interpret,
        accumulate=accumulate is not None,
        block_z=block_z,
    )
    operands = [poses, jnp.transpose(filtered, (0, 2, 1))]
    if accumulate is not None:
        shape = (grid.nx * grid.ny, grid.nz) if block_z else (-1,)
        operands.append(accumulate.reshape(shape))
    return call(*operands).reshape((grid.nx, grid.ny, grid.nz))
