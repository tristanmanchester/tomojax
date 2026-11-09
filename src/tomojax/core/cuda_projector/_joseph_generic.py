"""Exact CUDA plane sampling and voxel gathers for arbitrary parallel-ray poses."""

from __future__ import annotations

from functools import cache
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cuda import broadcast_offset, launch_kernel
from tomojax.core.volume_layout import swap_last_axes

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


@cache
def source(name: str, **constants: int | str) -> str:
    """Specialize a kernel's layout, retaining runtime volume/detector dimensions."""
    text = Path(__file__).with_name(name + ".cu").read_text()
    for header_name in ("joseph_sampling.cuh", "joseph_gather.cuh", "joseph_tile.cuh"):
        if f'#include "{header_name}"' in text:
            header = Path(__file__).parents[1].joinpath(header_name).read_text()
            text = text.replace(f'#include "{header_name}"', header)
    return "\n".join(f"#define {key} {value}" for key, value in constants.items()) + "\n" + text


def storage_tile(detector: Detector, *, z_fast: bool) -> int:
    """Forward rows per thread tile: rows follow contiguous z only in z-fast storage."""
    return min(16, 1 << (detector.nv.bit_length() - 1)) if z_fast else 1


def forward(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str,
    absolute_weights: bool,
    z_fast: bool = False,
) -> jax.Array:
    """Sum bilinear or cubic plane samples in the reference plane order.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``,
    whose rows lanes then follow along contiguous z.
    """
    threads = 32 if detector.nv == 1 else 128
    tile_v = storage_tile(detector, z_fast=z_fast)
    tile_u = threads // tile_v
    tiles = ((detector.nu + tile_u - 1) // tile_u) * ((detector.nv + tile_v - 1) // tile_v)
    views = coeff.shape[0]

    def launch(context: Any, out: Any, cf: Any, x: Any) -> None:
        batch_shape = tuple(out.shape[:-3])
        batches = math.prod(batch_shape)
        if not batches:
            return
        text = source(
            "forward",
            INDEX_TYPE="int" if grid.nx * grid.ny * grid.nz < 2**31 else "long",
            TILE_V=tile_v,
            BLOCK=threads,
            CUBIC=int(interpolation == "cubic"),
            ABSOLUTE=int(absolute_weights),
            Z_FAST=int(z_fast),
            COEFF_OFFSET=broadcast_offset(
                batch_shape, tuple(cf.shape[:-2]), item_size=math.prod(cf.shape[-2:])
            ),
            VOLUME_OFFSET=broadcast_offset(
                batch_shape, tuple(x.shape[:-3]), item_size=math.prod(x.shape[-3:])
            ),
        )
        launch_kernel(
            text,
            "forward",
            options=("--fmad=false",),
            stream=context.stream,
            grid=(batches * views * tiles,),
            block=(threads,),
            arguments=(
                cf,
                x,
                out,
                *map(np.int32, (grid.nx, grid.ny, grid.nz, detector.nu, detector.nv, views)),
                *map(np.int32, (0, grid.nz, grid.nz, 0)),  # all planes, one band
            ),
        )

    return buffer_callback(
        launch,
        jax.ShapeDtypeStruct((views, detector.nv, detector.nu), jnp.float32),
        command_buffer_compatible=True,
        vmap_method="expand_dims",
    )(coeff, volume)


def adjoint(
    coeff: jax.Array,
    images: jax.Array,
    initial: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str,
    absolute_weights: bool,
    z_fast: bool = False,
) -> jax.Array:
    """Gather the exact plane weights at each voxel and add an initial volume.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    Detector v is contiguous inside the gather, matching adjacent threads' z runs.
    Cubic sampling shares weights over 4x4 transverse tiles. Three ordered axis
    passes retain each axis's view order; only cross-axis addition is reordered.
    """
    shape = (grid.nx, grid.ny, grid.nz)
    run = min(4, grid.nz)
    runs = grid.nx * grid.ny * ((grid.nz + run - 1) // run)
    views = coeff.shape[0]
    launch_plans: list[tuple[str, str, int]] = []
    for axis in range(3) if interpolation == "cubic" else (None,):
        if axis is None:
            text = source(
                "adjoint_unit",
                RUN=run,
                CUBIC=0,
                ABSOLUTE=int(absolute_weights),
                Z_FAST=int(z_fast),
            )
            launch_plans.append((text, "joseph_gather", runs))
        else:
            na, nb, nc = (shape[(axis + i) % 3] for i in range(3))
            count = na * ((nb + 3) // 4) * ((nc + 3) // 4)
            text = source(
                "adjoint_tile",
                AXIS=axis,
                CUBIC=1,
                ABSOLUTE=int(absolute_weights),
                Z_FAST=int(z_fast),
            )
            launch_plans.append((text, "joseph_gather_tile", count))
    plans = tuple(launch_plans)

    def launch(context: Any, out: Any, cf: Any, y: Any, x: Any) -> None:
        for kernel_text, kernel, count in plans:
            launch_kernel(
                kernel_text,
                kernel,
                options=("--fmad=false",),
                stream=context.stream,
                grid=((count + 127) // 128,),
                block=(128,),
                arguments=(
                    cf,
                    y,
                    out,
                    x,
                    *map(np.int32, (views, *shape, detector.nu, detector.nv)),
                ),
            )
            x = out

    return buffer_callback(
        launch,
        jax.ShapeDtypeStruct(shape if z_fast else shape[::-1], jnp.float32),
        input_output_aliases={2: 0},
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff, swap_detector_axes(images), initial)


def swap_detector_axes(images: jax.Array) -> jax.Array:
    """Transpose through an opaque call, keeping loop-carried sinograms in VU order."""
    views, nv, nu = images.shape
    if nv == 1 or nu == 1:
        return images.reshape(views, nu, nv)

    def launch(context: Any, out: Any, data: Any) -> None:
        swap_last_axes(data, out, (views, nv, nu), stream=context.stream)

    return buffer_callback(
        launch,
        jax.ShapeDtypeStruct((views, nu, nv), jnp.float32),
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(images)
