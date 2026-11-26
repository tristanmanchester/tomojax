"""CUDA C gather transpose for linear Joseph plane sampling.

Each thread owns ``RUN`` consecutive z voxels and loops once over the union of
their detector footprints, so every detector load and loop step serves several
voxels. Weights use the same arithmetic as the forward projection. The kernel
is compiled at run time and launched on XLA's stream through
``buffer_callback``, inside compiled solver loops. Automatic dispatch uses it
from 2**21 voxels, or 2**19 for separable axial batches. Smaller volumes use
the allocator-free CUDA transpose.
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
from typing import TYPE_CHECKING

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core._joseph_cuda_kernels import RUN, THREADS, forward_kernel, gather_source
from tomojax.core.backend_policy import cuda_backend

if TYPE_CHECKING:
    from tomojax.core.cuda import Offset
    from tomojax.core.cuda_types import CallbackBuffer, CallbackContext, CuPyValue
    from tomojax.core.geometry.base import Detector, Grid

# Keep tiny grids on the allocator-free path. Kernel loading and callback setup
# have a fixed per-process cost.
# Medium grids already amortize that setup over ordinary solver iterations;
# requiring a full-resolution volume would exclude useful multiresolution work.
# This is a size-only crossover, independent of scan identity and geometry.
_MIN_VOXELS = 2**21
# Separable axial kernels amortize their setup sooner than the generic gather.
_MIN_AXIAL_VOXELS = 2**19


def use_cuda_gather(grid: Grid, *, axial: bool = False) -> bool:
    """Return whether to use the run-based CUDA gather for this volume.

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
    """Return whether the optional CUDA projector kernels are available."""
    if not cuda_backend(jax.default_backend()):
        return False
    return importlib.util.find_spec("cupy") is not None


class _XlaStream:
    """XLA's compute stream, exposed through the CUDA stream protocol."""

    def __init__(self, handle: int) -> None:
        self.handle = handle

    def __cuda_stream__(self) -> tuple[int, int]:
        return (0, self.handle)


@contextmanager
def xla_stream(context: CallbackContext, buffer: CallbackBuffer) -> Iterator[CuPyValue]:
    """Make ``buffer``'s GPU current and XLA's stream for ``context`` CuPy's stream.

    A ``buffer_callback`` context names its stream but not its device; on several
    GPUs each call runs on the one holding its buffers, which CuPy must use for
    kernels, scratch memory and library handles.
    """
    import cupy as cp

    with cp.cuda.Device(buffer.__dlpack_device__()[1]):
        stream = cp.cuda.Stream.from_external(_XlaStream(context.stream))
        with stream:
            yield stream


def _launch_gather(
    kernel: str,
    stream: int,
    coeff: CallbackBuffer | Offset,
    images: CallbackBuffer,
    out: CallbackBuffer,
    listed: tuple[CallbackBuffer, ...],
    shape: tuple[int, int, int],
    detector: tuple[int, int],
    views: int,
    z0: int,
) -> None:
    """Accumulate into a run list or a whole band using the same Joseph weights."""
    from tomojax.core.cuda import launch_kernel

    nx, ny, nz = shape
    runs = nx * ny * -(-nz // RUN)
    launch_kernel(
        kernel,
        "joseph_gather",
        stream=stream,
        grid=(-(-runs // THREADS),),
        block=(THREADS,),
        arguments=(
            coeff,
            images,
            out,
            *(listed or (coeff, coeff)),
            *map(np.int32, (bool(listed), views, nx, ny, nz, *detector, z0)),
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
    active: tuple[jax.Array, ...] | None = None,
) -> jax.Array:
    """Add the linear Joseph transpose of ``(view, u, v)`` images to ``accumulate``.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    ``active`` runs (``active_runs``) limit the gather to them; others keep
    ``accumulate``.
    """
    from tomojax.core.cuda import initialize_buffer

    shape = (grid.nx, grid.ny, grid.nz)
    kernel = gather_source(z_fast=z_fast)
    del detector  # Detector sizes come from the image batch.
    views, nu, nv = images_uv.shape

    def launch(
        context: CallbackContext,
        out: CallbackBuffer,
        coeff: CallbackBuffer,
        images: CallbackBuffer,
        initial: CallbackBuffer,
        *listed: CallbackBuffer,
    ) -> None:
        initialize_buffer(out, stream=context.stream, source=initial)
        _launch_gather(
            kernel, context.stream, coeff, images, out, listed, shape, (nu, nv), views, 0
        )

    count = grid.nx * grid.ny * grid.nz
    initial = (
        jnp.zeros((count,), jnp.float32)
        if accumulate is None
        else accumulate.astype(jnp.float32).reshape(count)
    )
    call = buffer_callback(
        launch,
        jax.ShapeDtypeStruct((count,), jnp.float32),
        command_buffer_compatible=True,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    coeff = coeff.astype(jnp.float32)

    def generic(_: None) -> jax.Array:
        return call(coeff, images_uv, initial, *(active or ()))

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
    from tomojax.core.cuda import Offset, initialize_buffer
    from tomojax.core.volume_layout import reverse_axes, swap_last_axes

    views, nv, nu = images.shape
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    depth, size = min(nz, 32), min(batch, views)
    gather = gather_source(z_fast=True)

    def launch(
        context: CallbackContext,
        outputs: tuple[CallbackBuffer, CallbackBuffer, CallbackBuffer],
        c: CallbackBuffer,
        y: CallbackBuffer,
    ) -> None:
        out, band, swapped = outputs
        stream = context.stream
        for z0 in range(0, nz, depth):
            planes = min(depth, nz - z0)
            initialize_buffer(band, stream=stream)
            for first in range(0, views, size):
                count = min(size, views - first)
                swap_last_axes(Offset(y, first * nv * nu), swapped, (count, nv, nu), stream=stream)
                _launch_gather(
                    gather,
                    stream,
                    Offset(c, 14 * first),
                    swapped,
                    band,
                    (),
                    (nx, ny, planes),
                    (nu, nv),
                    count,
                    z0,
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


def _launch_uniform(
    context: CallbackContext,
    out: CallbackBuffer,
    coeff: CallbackBuffer,
    *,
    grid: Grid,
    detector: Detector,
) -> None:
    import cupy as cp

    shape = (grid.nx, grid.ny, grid.nz)
    with xla_stream(context, out):
        geometry, target = cp.asarray(coeff), cp.asarray(out)
        kernel = forward_kernel(narrow=grid.nx * grid.ny * grid.nz < 2**31, uniform=True)
        tiles = -(-detector.nu // 128) * detector.nv
        for start in range(0, coeff.shape[0], 64):
            stop = min(start + 64, coeff.shape[0])
            dims = (*shape, detector.nu, detector.nv, stop - start, 0, grid.nz, grid.nz, 0)
            # The uniform specialization never dereferences its volume argument.
            kernel(
                (tiles * (stop - start),), (THREADS,),
                (geometry[start:stop], geometry, target[start:stop],
                 *map(np.int32, dims)),
            )  # fmt: skip


def project_ones_cuda(coeff: jax.Array, grid: Grid, detector: Detector) -> jax.Array:
    """Exact unit-support row sums without allocating or reading a ones volume."""
    call = buffer_callback(
        partial(_launch_uniform, grid=grid, detector=detector),
        jax.ShapeDtypeStruct((coeff.shape[0], detector.nv, detector.nu), jnp.float32),
        vmap_method="sequential",
    )

    def on_jax(c: jax.Array) -> jax.Array:
        from .joseph import forward_jax

        ones = jnp.ones((grid.nz, grid.ny, grid.nx), jnp.float32)
        return forward_jax(c, ones, grid, detector)

    return jax.lax.platform_dependent(coeff.astype(jnp.float32), cuda=call, default=on_jax)


def active_runs(mask: jax.Array, *, z_fast: bool = False) -> tuple[jax.Array, jax.Array]:
    """The linear gather's z runs holding a nonzero ``mask`` voxel: (ids, count).

    Ids come first, padded to the run total. Build them once for many gathers.
    """
    xyz = (mask if z_fast else jnp.transpose(mask, (2, 1, 0))) != 0
    xyz = jnp.pad(xyz, ((0, 0), (0, 0), (0, -xyz.shape[2] % RUN)))
    runs = xyz.reshape(*xyz.shape[:2], -1, RUN).any(-1).ravel()
    ids = jnp.nonzero(runs, size=runs.size, fill_value=0)[0].astype(jnp.int32)
    return ids, jnp.sum(runs, dtype=jnp.int32)[None]
