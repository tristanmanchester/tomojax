"""CUDA C gather transpose for linear Joseph plane sampling.

Each thread owns ``RUN`` consecutive z voxels and loops once over the union of
their detector footprints, so every detector load and loop step serves several
voxels. Weights use the same arithmetic as the forward projection. The kernel
is compiled at run time with CuPy and launched on XLA's stream through
``buffer_callback``, inside compiled solver loops. Automatic dispatch uses it
from 2**21 voxels, including coarse reconstruction and alignment grids. Smaller
volumes retain the Pallas transpose to avoid the fixed callback/kernel setup
cost for short operations. Runs without CuPy and ``TOMOJAX_CUDA_KERNELS=0``
also use Pallas; the environment override can force either implementation.
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

from tomojax.core._joseph_cuda_kernels import RUN, THREADS, gather_kernel
from tomojax.core.backend_policy import cuda_backend

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid

# Keep tiny grids on Pallas: CuPy import, kernel loading and the callback
# program incur a fixed cost that the persistent compile cache cannot reuse.
# Medium grids already amortize that setup over ordinary solver iterations;
# requiring a full-resolution volume would exclude useful multiresolution work.
# This is a size-only crossover, independent of scan identity and geometry.
_MIN_VOXELS = 2**21


def use_cuda_gather(grid: Grid) -> bool:
    """Return whether to use the CuPy gather for this volume.

    ``TOMOJAX_CUDA_KERNELS=1`` uses it for every size, ``0`` never.
    """
    setting = os.environ.get("TOMOJAX_CUDA_KERNELS", "auto")
    if setting == "0" or not cuda_gather_available():
        return False
    return setting == "1" or grid.nx * grid.ny * grid.nz >= _MIN_VOXELS


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
    context: Any, out: Any, coeff: Any, images: Any, accumulate: Any, *, shape: tuple[int, int, int]
) -> None:
    import cupy as cp

    nx, ny, nz = shape
    views, nu, nv = images.shape
    with xla_stream(context, out):
        target, initial = cp.asarray(out), cp.asarray(accumulate)
        if target.data.ptr != initial.data.ptr:
            target[...] = initial
        runs = nx * ny * -(-nz // RUN)
        gather_kernel()(
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
            ),
        )


def gather_transpose_cuda(
    coeff: jax.Array,
    images_uv: jax.Array,
    grid: Grid,
    detector: Detector,
    accumulate: jax.Array | None = None,
) -> jax.Array:
    """Add the linear Joseph transpose of ``(view, u, v)`` images to ``accumulate``."""
    del detector  # Detector sizes come from the image array.
    shape = (grid.nx, grid.ny, grid.nz)
    count = grid.nx * grid.ny * grid.nz
    initial = (
        jnp.zeros((count,), jnp.float32)
        if accumulate is None
        else accumulate.astype(jnp.float32).reshape(count)
    )
    call = buffer_callback(
        partial(_launch, shape=shape),
        jax.ShapeDtypeStruct((count,), jnp.float32),
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    return call(coeff.astype(jnp.float32), images_uv, initial).reshape(shape)
