"""Voxel-driven CUDA backprojection for filtered parallel-beam projections."""

from __future__ import annotations

from collections.abc import Callable
import functools
import importlib
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin

if TYPE_CHECKING:
    from tomojax._typed_arrays import Device


@functools.lru_cache(maxsize=32)
def _cuda_l2_bytes(device: Device) -> int | None:
    """Optional cache query for this JAX device, never CuPy's ambient device."""
    version = getattr(getattr(device, "client", None), "platform_version", "")
    local = getattr(device, "local_hardware_id", None)
    if device.platform != "gpu" or not version.lower().startswith("cuda") or local is None:
        return None
    try:
        runtime = importlib.import_module("cupy.cuda.runtime")
        size = int(runtime.getDeviceProperties(int(local)).get("l2CacheSize", 0))
        return size if size > 0 else None
    except Exception:
        # Cache sizing is a performance hint. Pallas needs neither CuPy nor a
        # working optional runtime probe to retain its original FFT batch size.
        return None


def filtered_view_batch_size(
    batch_size: int, *, grid: Grid, detector: Detector, device: Device
) -> int:
    """Cache-size dense axial FBP batches; retain FFT sizing for small/sparse outputs.

    Extra batches reread/write the whole accumulator, even for axial tiles that
    skip all views. That traffic can outweigh filtered-view caching for small
    or axially sparse outputs, which retain the original FFT budget. Callers
    require z-axis poses with no axial translation.
    """
    size = _cuda_l2_bytes(device)
    if size is None:
        return batch_size
    first = grid_volume_origin(grid)[2] - detector.center[1]
    last = first + (grid.nz - 1) * grid.vz
    support = (detector.nv + 1) * detector.dv / 2
    if 4 * grid.nx * grid.ny * grid.nz <= size or first < -support or last > support:
        return batch_size
    capacity = max(1, size // (4 * detector.nv * detector.nu))
    return min(batch_size, 1 << (capacity.bit_length() - 1))


def fft_2d_view_batch_size(batch_size: int, *, fft_shape: tuple[int, int], device: Device) -> int:
    """Cache-bound the real and half-complex working sets of a batched 2D FFT.

    Unlike independent row transforms, the two-dimensional transform revisits
    its full padded plane. Estimate eight bytes per spatial point for its real
    input and half-complex spectrum, without replacing the FFT memory budget or
    its allocation-failure retry. An unavailable optional probe changes nothing.
    """
    size = _cuda_l2_bytes(device)
    if size is None:
        return batch_size
    capacity = max(1, size // (8 * fft_shape[0] * fft_shape[1]))
    return min(batch_size, 1 << (capacity.bit_length() - 1))


def _column_tile(nz: int, block_size: int) -> tuple[int, int]:
    rows = min(128, block_size, 1 << (nz - 1).bit_length())
    return block_size // rows, rows


def _voxel_indices(
    shape: tuple[int, int, int], *, block_size: int, z_mode: str, interpret: bool
) -> tuple[jnp.ndarray, ...]:
    nx, ny, nz = shape
    if z_mode == "general":
        flat = pl.program_id(0) * block_size + jnp.arange(block_size, dtype=jnp.int32)
        valid = flat < nx * ny * nz
        ix, iy, iz = flat // (ny * nz), (flat // nz) % ny, flat % nz
    else:
        # A two-dimensional tile calculates each column coordinate once, then
        # broadcasts it along contiguous detector rows instead of repeating the
        # horizontal coordinate arithmetic in every z lane for every view.
        columns, rows = _column_tile(nz, block_size)
        column = pl.program_id(0) * columns + jnp.arange(columns, dtype=jnp.int32)
        iz = pl.program_id(1) * rows + jnp.arange(rows, dtype=jnp.int32)[None, :]
        ix, iy = (column // ny)[:, None], (column % ny)[:, None]
        flat = column[:, None] * nz + iz
        valid = (column[:, None] < nx * ny) & (iz < nz)
        # Interpret stores use indexed updates: masked padded rows must not
        # alias the next column's first voxel. CUDA suppresses masked stores;
        # keep affine addresses there so loads/stores can remain vectorized.
        if interpret:
            flat = jnp.where(valid, flat, nx * ny * nz)
    return flat, valid, ix, iy, iz


def _sum_supported_views(
    body: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    n_views: int,
    initial: jnp.ndarray,
    rows: jnp.ndarray | None,
) -> jnp.ndarray:
    def summed() -> jnp.ndarray:
        return jax.lax.fori_loop(0, n_views, body, initial)

    # A column tile whose axial rows lie entirely beyond the detector sees
    # zero in every view. Skip the view loop, not merely its masked loads.
    if rows is None:
        return summed()
    return jax.lax.cond(jnp.sum(rows.astype(jnp.int32)) > 0, summed, lambda: initial)


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
    z_mode: str,
    interpret: bool,
) -> None:
    # With an accumulator, refs is (accumulator, aliased output).
    acc_ref, out_ref = refs if len(refs) == 2 else (None, refs[0])
    nv, nu = detector_shape
    du, dv = detector_spacing
    flat, valid, ix, iy, iz = _voxel_indices(
        shape, block_size=block_size, z_mode=z_mode, interpret=interpret
    )
    x = ix.astype(jnp.float32) * voxel[0] + origin[0]
    y = iy.astype(jnp.float32) * voxel[1] + origin[1]
    z = iz.astype(jnp.float32) * voxel[2] + origin[2]
    # Add integer pixel origins only after extracting the interpolation fractions.
    u_offset, v_offset = ((nu - 1) % 2) / 2.0, ((nv - 1) % 2) / 2.0
    supported_rows = None
    if z_mode == "integer":
        iv = (
            jnp.floor((z - detector_center[1]) / dv + v_offset + 0.5).astype(jnp.int32)
            + (nv - 1) // 2
        )
        supported_rows = (iv >= 0) & (iv < nv)
    elif z_mode == "fixed":
        world_z = plt.load(poses_ref.at[jnp.int32(0), jnp.int32(2), jnp.int32(2)]) * z + plt.load(
            poses_ref.at[jnp.int32(0), jnp.int32(2), jnp.int32(3)]
        )
        v = (world_z - detector_center[1]) / dv + v_offset
        lower_v = jnp.floor(v)
        fixed_wv = v - lower_v
        fixed_iv0 = lower_v.astype(jnp.int32) + (nv - 1) // 2
        supported_rows = (fixed_iv0 >= -1) & (fixed_iv0 < nv)

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

    result = _sum_supported_views(
        body, n_views, jnp.zeros(flat.shape, dtype=jnp.float32), supported_rows
    )
    if acc_ref is not None:
        result = plt.load(acc_ref.at[flat], mask=valid, other=0.0) + result
    plt.store(out_ref.at[flat], result, mask=valid)


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
) -> Any:
    count = grid.nx * grid.ny * grid.nz
    in_specs = [pl.no_block_spec, pl.no_block_spec]
    if z_mode == "general":
        launch_grid = ((count + block_size - 1) // block_size,)
    else:
        columns, rows = _column_tile(grid.nz, block_size)
        launch_grid = (
            (grid.nx * grid.ny + columns - 1) // columns,
            (grid.nz + rows - 1) // rows,
        )
    if accumulate:
        in_specs.append(pl.no_block_spec)
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
            z_mode=z_mode,
            interpret=interpret,
        ),
        out_shape=jax.ShapeDtypeStruct((count,), jnp.float32),
        grid=launch_grid,
        in_specs=in_specs,
        out_specs=pl.no_block_spec,
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
    four-warp program. Other workloads retain larger tiles for input reuse.
    Z-axis modes broadcast each column's horizontal coordinate along axial
    tiles and skip the view loop when all their rows lie outside the detector.
    """
    if z_mode not in {"integer", "fixed", "general"}:
        raise ValueError("FBP z_mode must be integer, fixed, or general")
    if block_size is None:
        working_bytes = filtered.size * filtered.dtype.itemsize + 4 * grid.nx * grid.ny * grid.nz
        compact = working_bytes <= 64 * 1024**2
        block_size = 128 if compact else 512
    call = _backprojection_call(
        grid,
        detector,
        int(poses.shape[0]),
        block_size,
        z_mode=z_mode,
        interpret=interpret,
        accumulate=accumulate is not None,
    )
    operands = [poses, jnp.transpose(filtered, (0, 2, 1))]
    if accumulate is not None:
        operands.append(accumulate.reshape(-1))
    return call(*operands).reshape((grid.nx, grid.ny, grid.nz))
