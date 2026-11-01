"""CUDA C gather transpose for linear Joseph plane sampling.

Each thread owns ``RUN`` consecutive z voxels and loops once over the union of
their detector footprints, so every detector load and loop step serves several
voxels. Weights use the same arithmetic as the forward projection. The kernel
is compiled at run time with CuPy and launched on XLA's stream through
``buffer_callback``, inside compiled solver loops. Automatic dispatch uses it
from 2**21 voxels, or 2**19 for separable axial batches. Smaller volumes retain
the allocator-free CUDA transpose to avoid CuPy initialization overhead.
Axial batches use a bounded separable vertical prepass; other views
retain this gather. Runs without CuPy and ``TOMOJAX_CUDA_KERNELS=0`` also use
that allocator-free path; the environment override can force either implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache, partial
import importlib.util
import os
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core._joseph_cuda_kernels import (
    RUN,
    THREADS,
    forward_kernel,
    gather_kernel,
    gather_source,
)
from tomojax.core.backend_policy import cuda_backend

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid

# Keep tiny grids on the allocator-free path: CuPy import, kernel loading and the callback
# program incur a fixed cost that the persistent compile cache cannot reuse.
# Medium grids already amortize that setup over ordinary solver iterations;
# requiring a full-resolution volume would exclude useful multiresolution work.
# This is a size-only crossover, independent of scan identity and geometry.
_MIN_VOXELS = 2**21
# Separable axial kernels amortize their setup sooner than the generic gather.
_MIN_AXIAL_VOXELS = 2**19


def use_cuda_gather(grid: Grid, *, axial: bool = False) -> bool:
    """Return whether to use the CuPy gather for this volume.

    ``TOMOJAX_CUDA_KERNELS=1`` uses it for every size, ``0`` never.
    ``axial=True`` allows the lower separable-kernel threshold; the caller must
    still check the coefficients and workspace before selecting that kernel.
    """
    setting = os.environ.get("TOMOJAX_CUDA_KERNELS", "auto")
    if setting == "0" or not cuda_gather_available():
        return False
    minimum = _MIN_AXIAL_VOXELS if axial else _MIN_VOXELS
    return setting == "1" or grid.nx * grid.ny * grid.nz >= minimum


@cache
def cuda_gather_available() -> bool:
    """Return whether the CuPy gather can run on JAX's default CUDA device."""
    if not cuda_backend():
        return False
    return importlib.util.find_spec("cupy") is not None


class _XlaStream:
    """XLA's compute stream, exposed through the CUDA stream protocol."""

    def __init__(self, handle: int) -> None:
        self.handle = handle

    def __cuda_stream__(self) -> tuple[int, int]:
        return (0, self.handle)


@contextmanager
def xla_stream(context: Any, buffer: Any) -> Iterator[Any]:
    """Make ``buffer``'s GPU current and XLA's stream for ``context`` CuPy's stream.

    A ``buffer_callback`` context names its stream but not its device; on several
    GPUs each call runs on the one holding its buffers, which CuPy must use for
    kernels, scratch memory and library handles.
    """
    import cupy as cp

    with cp.cuda.Device(int(buffer.__dlpack_device__()[1])):
        stream = cp.cuda.Stream.from_external(_XlaStream(int(context.stream)))
        with stream:
            yield stream


def _launch(
    context: Any,
    out: Any,
    coeff: Any,
    images: Any,
    accumulate: Any,
    *,
    shape: tuple[int, int, int],
    z_fast: bool,
) -> None:
    import cupy as cp

    nx, ny, nz = shape
    views, nu, nv = images.shape
    with xla_stream(context, out):
        target, initial = cp.asarray(out), cp.asarray(accumulate)
        if target.data.ptr != initial.data.ptr:
            target[...] = initial
        runs = nx * ny * -(-nz // RUN)
        gather_kernel(z_fast=z_fast)(
            (-(-runs // THREADS),),
            (THREADS,),
            (
                cp.asarray(coeff),
                cp.asarray(images),
                target,
                np.int32(views),
                np.int32(nx),
                np.int32(ny),
                np.int32(nz),
                np.int32(nu),
                np.int32(nv),
                np.int32(0),
            ),
        )


def gather_transpose_cuda(
    coeff: jax.Array,
    images_uv: jax.Array,
    grid: Grid,
    detector: Detector,
    accumulate: jax.Array | None = None,
    *,
    z_fast: bool = False,
) -> jax.Array:
    """Add the linear Joseph transpose of ``(view, u, v)`` images to ``accumulate``.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    """
    del detector  # Detector sizes come from the image array.
    shape = (grid.nx, grid.ny, grid.nz)
    count = grid.nx * grid.ny * grid.nz
    initial = (
        jnp.zeros((count,), jnp.float32)
        if accumulate is None
        else accumulate.astype(jnp.float32).reshape(count)
    )
    call = buffer_callback(
        partial(_launch, shape=shape, z_fast=z_fast),
        jax.ShapeDtypeStruct((count,), jnp.float32),
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    coeff = coeff.astype(jnp.float32)

    def generic(_: None) -> jax.Array:
        return call(coeff, images_uv, initial)

    # The axial transpose bounds its vertical-resampling scratch by splitting
    # views. Nonseparable views keep the existing gather unchanged.
    from tomojax.core._cuda_axial import axial_coefficients, transpose_axial_cuda

    return jax.lax.cond(
        axial_coefficients(coeff),
        lambda _: transpose_axial_cuda(coeff, images_uv, initial, grid=grid, z_fast=z_fast),
        generic,
        operand=None,
    ).reshape(shape if z_fast else shape[::-1])


def gather_bands_cuda(coeff: jax.Array, images: jax.Array, grid: Grid, *, batch: int) -> jax.Array:
    """The linear Joseph transpose of ``(view, v, u)`` images into a new (z, y, x) volume.

    The gather fills 32 planes at a time of a z-fast band, at their global z,
    ``batch`` views a launch as one call per batch would; each voxel so adds
    the same batch sums in the same order as from z-fast storage. A band and
    one batch of ``(view, u, v)`` images are the workspace.
    """
    from tomojax.core.cuda import Offset, initialize_buffer, launch_kernel
    from tomojax.core.volume_layout import reverse_axes, swap_last_axes

    views, nv, nu = images.shape
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    depth, size = min(nz, 32), min(batch, views)
    gather = gather_source(z_fast=True)

    def launch(context: Any, outputs: Any, c: Any, y: Any) -> None:
        out, band, swapped = outputs
        stream = context.stream
        for z0 in range(0, nz, depth):
            planes = min(depth, nz - z0)
            initialize_buffer(band, stream=stream)
            runs = nx * ny * -(-planes // RUN)
            for first in range(0, views, size):
                count = min(size, views - first)
                swap_last_axes(Offset(y, first * nv * nu), swapped, (count, nv, nu), stream=stream)
                launch_kernel(
                    gather,
                    "joseph_gather",
                    stream=stream,
                    grid=(-(-runs // THREADS),),
                    block=(THREADS,),
                    arguments=(
                        Offset(c, 14 * first),
                        swapped,
                        band,
                        *map(np.int32, (count, nx, ny, planes, nu, nv, z0)),
                    ),
                )
            reverse_axes(band, Offset(out, z0 * ny * nx), (nx, ny, planes), stream=stream, depth=nx)

    volume, *_ = buffer_callback(
        launch,
        (
            jax.ShapeDtypeStruct((nz, ny, nx), jnp.float32),
            jax.ShapeDtypeStruct((nx, ny, depth), jnp.float32),
            jax.ShapeDtypeStruct((size, nu, nv), jnp.float32),
        ),
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff.astype(jnp.float32), images.astype(jnp.float32))
    return volume


def _launch_uniform(context: Any, out: Any, coeff: Any, *, grid: Grid, detector: Detector) -> None:
    import cupy as cp

    shape = (grid.nx, grid.ny, grid.nz)
    with xla_stream(context, out):
        geometry, target = cp.asarray(coeff), cp.asarray(out)
        kernel = forward_kernel(narrow=grid.nx * grid.ny * grid.nz < 2**31, uniform=True)
        tiles = -(-detector.nu // 128) * detector.nv
        for start in range(0, coeff.shape[0], 64):
            stop = min(start + 64, coeff.shape[0])
            # The uniform specialization never dereferences its volume argument.
            kernel(
                (tiles * (stop - start),), (THREADS,),
                (geometry[start:stop], geometry, target[start:stop],
                 *map(np.int32, (*shape, detector.nu, detector.nv, stop - start, 0, grid.nz))),
            )  # fmt: skip


def project_ones_cuda(coeff: jax.Array, grid: Grid, detector: Detector) -> jax.Array:
    """Exact unit-support row sums without allocating or reading a ones volume."""
    call = buffer_callback(
        partial(_launch_uniform, grid=grid, detector=detector),
        jax.ShapeDtypeStruct((coeff.shape[0], detector.nv, detector.nu), jnp.float32),
        vmap_method="sequential",
    )

    def on_jax(c):
        from .joseph import forward_jax

        ones = jnp.ones((grid.nz, grid.ny, grid.nx), jnp.float32)
        return forward_jax(c, ones, grid, detector)

    return jax.lax.platform_dependent(coeff.astype(jnp.float32), cuda=call, default=on_jax)
