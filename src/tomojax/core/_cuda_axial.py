"""Exact separable CUDA Joseph operations for axial views.

The vertical transpose is evaluated once per detector column, then shared by
all horizontal voxels. The existing generic pair handles nonseparable views.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core._axial_kernels import (
    FORWARD,
    FORWARD_VEC4,
    aligned_axial,
    axial_coefficients,
    tile_grid,
    transpose_into,
    transpose_source,
    view_chunks,
)
from tomojax.core.cuda import Offset, initialize_buffer, launch_kernel
from tomojax.core.volume_layout import reverse_axes

if TYPE_CHECKING:
    from tomojax.core.cuda_types import CallbackBuffer, CallbackContext
    from tomojax.core.geometry.base import Detector, Grid


def _transpose_launch(
    context: CallbackContext,
    out: CallbackBuffer,
    coeff: CallbackBuffer,
    images: CallbackBuffer,
    initial: CallbackBuffer,
    *,
    shape: tuple[int, int, int],
    cubic: bool,
    z_fast: bool,
) -> None:
    import cupy as cp

    from tomojax.core._cuda_joseph import xla_stream

    with xla_stream(context, out):
        target, previous = cp.asarray(out), cp.asarray(initial)
        if target.data.ptr != previous.data.ptr:
            target[...] = previous
        transpose_into(
            cp,
            cp.asarray(coeff),
            cp.asarray(images),
            target,
            shape=shape,
            image_shape=(images.shape[0], images.shape[1], images.shape[2]),
            cubic=cubic,
            z_fast=z_fast,
            vu=False,
        )


def transpose_axial_cuda(
    coeff: jax.Array,
    images_uv: jax.Array,
    initial: jax.Array,
    *,
    grid: Grid,
    interpolation: str = "linear",
    z_fast: bool = False,
) -> jax.Array:
    """Add the exact linear or Keys-cubic axial transpose to a flat initial volume.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    """
    shape = (grid.nx, grid.ny, grid.nz)
    cubic = interpolation == "cubic"
    call = buffer_callback(
        partial(_transpose_launch, shape=shape, cubic=cubic, z_fast=z_fast),
        jax.ShapeDtypeStruct(initial.shape, jnp.float32),
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    return call(coeff, images_uv, initial)


def _banded_forward(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    text: str,
    tile: int,
    *,
    z_fast: bool,
) -> jax.Array:
    """Four-row axial FP, 32 planes at a time from a (z, y, x) volume.

    The volume is staged one z-contiguous band at a time, so every ray sums the
    same values in the same order as from z-fast storage, with one band of
    workspace. A z-fast volume is read in place in one pass.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    nu, nv = detector.nu, detector.nv
    depth = nz if z_fast else min(nz, 32)
    tiles = -(-nu // (128 // tile)) * -(-depth // (4 * tile))
    # Views 16 a launch share their reads in L2; narrow scans take more, so
    # that each launch has about a wave of blocks.
    chunk = max(16, -(-512 // tiles))
    staged = () if z_fast else (jax.ShapeDtypeStruct((nx, ny, depth), jnp.float32),)

    def launch(context: Any, outputs: Any, cf: Any, x: Any) -> None:
        out, *band = outputs if staged else (outputs,)
        views, stream, options = cf.shape[0], context.stream, ("--fmad=false",)
        launch_kernel(
            text,
            "axial_rows_outside",
            options=options,
            stream=stream,
            grid=(views, -(-nv // 4)),
            block=(128,),
            arguments=(cf, out, *map(np.int32, (nz, nu, nv))),
        )
        for z0 in range(0, nz, depth):
            if band:
                reverse_axes(x, band[0], (nz, ny, nx), stream=stream, z0=z0, depth=depth)
            for first in range(0, views, chunk):
                launch_kernel(
                    text,
                    "axial_forward_vec4",
                    options=options,
                    stream=stream,
                    grid=tile_grid(min(chunk, views - first), tiles),
                    block=(128,),
                    arguments=(
                        cf,
                        band[0] if band else x,
                        out,
                        *map(np.int32, (nx, ny, nz, nu, nv, first, z0, depth)),
                    ),
                )

    shape = jax.ShapeDtypeStruct((coeff.shape[0], nv, nu), jnp.float32)
    result = buffer_callback(
        launch,
        (shape, *staged) if staged else shape,
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff, volume)
    return result[0] if staged else result


def forward_dispatch_cuda(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    generic_source: str,
    tile_v: int,
    z_fast: bool = False,
) -> jax.Array:
    """Select generic, scalar axial or four-row axial FP without extra compilation.

    ``generic_source`` defines the volume storage (Z_FAST) for all three kernels.
    """
    nv, nu = detector.nv, detector.nu
    threads = 32 if nv == 1 else 128
    scalar_tile = min(32 if grid.nz >= 512 else 16, 1 << (nv.bit_length() - 1)) if z_fast else 1
    vector = grid.nz % 4 == 0 and nv >= 4
    vector_tile = min(8, 1 << ((nv // 4).bit_length() - 1)) if vector else 0
    kernels = [generic_source, FORWARD.replace("TV", str(scalar_tile))]
    if vector:
        kernels.append(FORWARD_VEC4.replace("TV", str(vector_tile)))
    text = "\n".join(kernels)
    shape = jax.ShapeDtypeStruct((coeff.shape[0], nv, nu), jnp.float32)

    def call(c: jax.Array, x: jax.Array, mode: int) -> jax.Array:
        def launch(context: Any, out: Any, cf: Any, image: Any) -> None:
            dims = (grid.nx, grid.ny, grid.nz, nu, nv)
            if mode == 0:
                tiles = -(-nu // (threads // tile_v)) * -(-nv // tile_v)
                launch_kernel(
                    text,
                    "forward",
                    options=("--fmad=false",),
                    stream=context.stream,
                    grid=(tiles * cf.shape[0],),
                    block=(threads,),
                    arguments=(
                        cf,
                        image,
                        out,
                        *map(np.int32, (*dims, cf.shape[0], 0, grid.nz, grid.nz, 0)),
                    ),
                )
            else:
                launch_kernel(
                    text,
                    "axial_forward",
                    options=("--fmad=false",),
                    stream=context.stream,
                    grid=tile_grid(
                        cf.shape[0], -(-nu // (128 // scalar_tile)) * -(-nv // scalar_tile)
                    ),
                    block=(128,),
                    arguments=(cf, image, out, *map(np.int32, (*dims, 0, grid.nz, grid.nz, 0))),
                )

        return buffer_callback(
            launch, shape, command_buffer_compatible=True, vmap_method="sequential"
        )(c, x)

    def axial(values: tuple[jax.Array, jax.Array]) -> jax.Array:
        c, x = values
        if not vector:
            return call(c, x, 1)
        return jax.lax.cond(
            aligned_axial(c),
            lambda args: _banded_forward(
                args[0], args[1], grid, detector, text, vector_tile, z_fast=z_fast
            ),
            lambda args: call(args[0], args[1], 1),
            (c, x),
        )

    return jax.lax.cond(
        axial_coefficients(coeff),
        axial,
        lambda args: call(args[0], args[1], 0),
        (coeff, volume),
    )


def forward_bands_cuda(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    depth: int,
    axial: bool,
) -> jax.Array:
    """The linear forward of a (z, y, x) volume from z-fast bands of ``depth`` < nz planes.

    Each band, with one halo plane, is staged once for all views. The scalar
    kernel sums an ``axial`` row whole in the band holding its lower plane; the
    generic kernel's rays add each band's samples in their own z order, resuming
    their sums from the output. So the bits are those of the forward from
    z-fast storage.
    """
    from tomojax.core.cuda_projector._joseph_generic import source, storage_tile

    nx, ny, nz, nu, nv = grid.nx, grid.ny, grid.nz, detector.nu, detector.nv
    planes, threads = depth + 1, 32 if nv == 1 else 128
    tile_v = storage_tile(detector, z_fast=True)
    scalar_tile = min(32 if nz >= 512 else 16, 1 << (nv.bit_length() - 1))
    generic = source(
        "forward",
        INDEX_TYPE="int" if nx * ny * nz < 2**31 else "long",
        TILE_V=tile_v,
        BLOCK=threads,
        CUBIC=0,
        ABSOLUTE=0,
        Z_FAST=1,
        BAND=1,
    )
    text = generic + FORWARD.replace("TV", str(scalar_tile))
    tiles = -(-nu // (threads // tile_v)) * -(-nv // tile_v)
    scalar_tiles = -(-nu // (128 // scalar_tile)) * -(-nv // scalar_tile)
    views = coeff.shape[0]
    if axial:
        kernel, blocks, block = "axial_forward", tile_grid(views, scalar_tiles), 128
    else:
        kernel, blocks, block = "forward", (views * tiles,), threads
    starts = range(0, nz, depth)

    def launch(context: Any, outputs: Any, cf: Any, x: Any) -> None:
        out, band = outputs
        initialize_buffer(out, stream=context.stream)
        # Generic rays whose z falls take the bands top down in a second pass.
        for descending in (0,) if axial else (0, 1):
            for z0 in starts[::-1] if descending else starts:
                reverse_axes(x, band, (nz, ny, nx), stream=context.stream, z0=z0, depth=planes)
                zhi = min(z0 + depth, nz)
                ranges = (z0, planes, zhi, z0) if axial else (views, z0, planes, zhi, descending)
                launch_kernel(
                    text,
                    kernel,
                    options=("--fmad=false",),
                    stream=context.stream,
                    grid=blocks,
                    block=(block,),
                    arguments=(cf, band, out, *map(np.int32, (nx, ny, nz, nu, nv, *ranges))),
                )

    projections, _ = buffer_callback(
        launch,
        (
            jax.ShapeDtypeStruct((views, nv, nu), jnp.float32),
            jax.ShapeDtypeStruct((nx, ny, planes), jnp.float32),
        ),
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff.astype(jnp.float32), volume.astype(jnp.float32))
    return projections


def transpose_axial_vu_cuda(
    coeff: jax.Array,
    images: jax.Array,
    initial: jax.Array | None,
    *,
    grid: Grid,
    interpolation: str,
    z_fast: bool = False,
    batch: int | None = None,
) -> jax.Array:
    """Add the exact axial transpose using direct VU input and JAX-owned scratch.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    A (z, y, x) volume is added to 32 planes at a time in a z-fast band, so
    each voxel adds the same chunk sums in the same order in either storage.
    ``batch`` views at a time make the chunks of one call per batch. With no
    ``initial`` the first chunk writes the new output without reading it.
    """
    views, nv, nu = images.shape
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    fresh = initial is None
    size, chunks = view_chunks(views, nu, nz, batch)
    public_depth = min(nz, 32)
    # Keep vertical scratch within the original band's combined image and
    # volume workspace. A private volume can receive each band directly.
    capacity = public_depth * (size * nu + nx * ny)
    depth = min(nz, capacity // (size * nu)) if z_fast else public_depth
    if depth < nz:
        # Every full band can use the axial kernel's eight-plane vector loads.
        depth -= depth % 8

    def add(stream: int, c: Any, y: Any, weighted: Any, target: Any, z0: int, planes: int) -> None:
        # Aligned float4 loads amortize global-load instructions. Odd axial
        # extents retain guarded scalar reads, including the last partial run.
        vector = planes % 8 == 0
        run, threads = (8, 256) if vector else (4, 128)
        text = transpose_source(vector=vector, cubic=interpolation == "cubic", z_fast=True, vu=True)
        runs = nx * ny * -(-planes // run)
        for first, count in chunks:
            part = Offset(c, 14 * first)
            launch_kernel(
                text,
                "joseph_vertical",
                stream=stream,
                grid=(-(-(count * nu * planes) // 256),),
                block=(256,),
                arguments=(
                    part,
                    Offset(y, first * nv * nu),
                    weighted,
                    *map(np.int32, (count, planes, nu, nv, z0)),
                ),
            )
            launch_kernel(
                text,
                "joseph_axial",
                stream=stream,
                grid=(-(-runs // threads),),
                block=(threads,),
                arguments=(
                    part,
                    weighted,
                    target,
                    *map(
                        np.int32,
                        (count, nx, ny, planes, nu, fresh and first == 0, nz if z_fast else planes),
                    ),
                ),
            )

    def launch(context: Any, outputs: Any, c: Any, y: Any, *x: Any) -> None:
        out, weighted, *band = outputs
        stream = context.stream
        if x or (not band and not chunks):
            initialize_buffer(out, stream=stream, source=x[0] if x else None)
        if not band:
            for z0 in range(0, nz, depth):
                add(stream, c, y, weighted, Offset(out, z0), z0, min(depth, nz - z0))
            return
        for z0 in range(0, nz, depth):
            planes = min(depth, nz - z0)
            if x:
                reverse_axes(out, band[0], (nz, ny, nx), stream=stream, z0=z0, depth=planes)
            elif not chunks:
                initialize_buffer(band[0], stream=stream)
            add(stream, c, y, weighted, band[0], z0, planes)
            reverse_axes(
                band[0], Offset(out, z0 * ny * nx), (nx, ny, planes), stream=stream, depth=nx
            )

    staged = () if z_fast else (jax.ShapeDtypeStruct((nx, ny, depth), jnp.float32),)
    shape = (nx, ny, nz) if z_fast else (nz, ny, nx)
    result, *_ = buffer_callback(
        launch,
        (
            jax.ShapeDtypeStruct(shape if initial is None else initial.shape, jnp.float32),
            jax.ShapeDtypeStruct((size, nu, depth), jnp.float32),
            *staged,
        ),
        input_output_aliases={} if initial is None else {2: 0},
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff, images, *(() if initial is None else (initial,)))
    return result.reshape(shape)


def transpose_axial_pair_cuda(
    coeff: jax.Array,
    images: tuple[jax.Array, jax.Array],
    initial: tuple[jax.Array, jax.Array],
    grid: Grid,
) -> tuple[jax.Array, jax.Array]:
    """Accumulate two linear transposes in z-fast storage with one horizontal ray walk."""
    views, nv, nu = images[0].shape
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    size, chunks = view_chunks(views, nu, nz)
    # Use two vertical rows per RHS pair and keep the same view reductions.
    depth = min(nz, 32)
    output = jax.ShapeDtypeStruct((nx, ny, nz), jnp.float32)
    scratch = jax.ShapeDtypeStruct((size, nu, depth), jnp.float32)

    def launch(
        ctx: CallbackContext,
        buffers: tuple[CallbackBuffer, CallbackBuffer, CallbackBuffer, CallbackBuffer],
        c: CallbackBuffer,
        y: CallbackBuffer,
        h: CallbackBuffer,
        score: CallbackBuffer,
        curvature: CallbackBuffer,
    ) -> None:
        dst, other, weighted, weights_other = buffers
        initialize_buffer(dst, stream=ctx.stream, source=score)
        initialize_buffer(other, stream=ctx.stream, source=curvature)
        for z0 in range(0, nz, depth):
            planes = min(depth, nz - z0)
            vector = planes % 8 == 0
            run, threads = (8, 256) if vector else (4, 128)
            text = transpose_source(vector=vector, cubic=False, z_fast=True, vu=True, paired=True)
            runs = nx * ny * -(-planes // run)
            for first, count in chunks:
                part = Offset(c, 14 * first)
                for source, work in ((y, weighted), (h, weights_other)):
                    launch_kernel(
                        text,
                        "joseph_vertical",
                        stream=ctx.stream,
                        grid=(-(-(count * nu * planes) // 256),),
                        block=(256,),
                        arguments=(
                            part,
                            Offset(source, first * nv * nu),
                            work,
                            *map(np.int32, (count, planes, nu, nv, z0)),
                        ),
                    )
                launch_kernel(
                    text,
                    "joseph_axial",
                    stream=ctx.stream,
                    grid=(-(-runs // threads),),
                    block=(threads,),
                    arguments=(
                        part,
                        weighted,
                        Offset(dst, z0),
                        weights_other,
                        Offset(other, z0),
                        *map(np.int32, (count, nx, ny, planes, nu, 0, nz)),
                    ),
                )

    score, curvature, *_ = buffer_callback(
        launch,
        (output, output, scratch, scratch),
        input_output_aliases={3: 0, 4: 1},
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff, *images, *initial)
    return score, curvature
