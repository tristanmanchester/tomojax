"""CUDA C gather transpose for linear Joseph plane sampling.

Each thread owns ``_RUN`` consecutive z voxels and loops once over the union of
their detector footprints, so every detector load and loop step serves several
voxels. Weights use the same arithmetic as the forward projection. The kernel
is compiled at run time with CuPy and launched on XLA's stream through
``buffer_callback``, inside compiled solver loops. It serves volumes of 2**24
voxels and more; smaller volumes, runs without CuPy and
``TOMOJAX_CUDA_KERNELS=0`` use the Pallas transpose.
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

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid

_RUN = 4
_VIEWS_PER_STAGE = 32
_THREADS = 128
# Below this volume size the one-off cost (CuPy import, kernel load, and a
# callback program the persistent compile cache cannot reuse, about 0.2-0.4 s
# per process) outweighs the faster gather; the Pallas transpose is used.
_MIN_VOXELS = 2**24

_SOURCE = r"""
extern "C" __global__ void joseph_gather(
    const float* __restrict__ coeff, const float* __restrict__ images,
    float* __restrict__ out, int nviews, int nx, int ny, int nz, int nu, int nv)
{
    const int R = RUN, VB = VIEWS;
    __shared__ float cf[VB * 14];
    long run = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int runs_z = (nz + R - 1) / R;
    bool valid = run < (long)nx * ny * runs_z;
    int ix = valid ? run / ((long)ny * runs_z) : 0;
    int iy = valid ? (run / runs_z) % ny : 0;
    int iz0 = valid ? R * (int)(run % runs_z) : 0;
    float total[R];
    #pragma unroll
    for (int j = 0; j < R; ++j) total[j] = 0.f;
    for (int first = 0; first < nviews; first += VB) {
        __syncthreads();
        for (int i = threadIdx.x; i < VB * 14; i += blockDim.x)
            cf[i] = first * 14 + i < nviews * 14 ? coeff[first * 14 + i] : 0.f;
        __syncthreads();
        int count = min(VB, nviews - first);
        for (int view = 0; view < count; ++view) {
            const float* c = cf + view * 14;
            int a = (int)c[0];
            float ub = c[1], vb = c[2], kb = c[3], cb = c[4];
            float uc = c[5], vc = c[6], kc = c[7], cc = c[8], weight = c[9];
            float iub = c[10], iuc = c[11], ivb = c[12], ivc = c[13];
            int k0 = a == 0 ? ix : (a == 1 ? iy : iz0);
            int b0 = a == 0 ? iy : (a == 1 ? iz0 : ix);
            int c0 = a == 0 ? iz0 : (a == 1 ? ix : iy);
            // Step of the run's voxel index along the plane coordinates.
            int dk = a == 2, db = a == 1, dc = a == 0;
            float hb[R], hc[R], tb[R], tc[R];
            float lo_u = 3.4e38f, hi_u = -3.4e38f, lo_v = 3.4e38f, hi_v = -3.4e38f;
            #pragma unroll
            for (int j = 0; j < R; ++j) {
                float k = (float)(k0 + j * dk);
                hb[j] = __fadd_rn(__fmul_rn(kb, k), cb);
                hc[j] = __fadd_rn(__fmul_rn(kc, k), cc);
                tb[j] = (float)(b0 + j * db);
                tc[j] = (float)(c0 + j * dc);
                float cu = iub * (tb[j] - hb[j]) + iuc * (tc[j] - hc[j]);
                float cv = ivb * (tb[j] - hb[j]) + ivc * (tc[j] - hc[j]);
                lo_u = fminf(lo_u, cu); hi_u = fmaxf(hi_u, cu);
                lo_v = fminf(lo_v, cv); hi_v = fmaxf(hi_v, cv);
            }
            float ru = fabsf(iub) + fabsf(iuc), rv = fabsf(ivb) + fabsf(ivc);
            int u_start = max((int)floorf(lo_u - ru) + 1, 0);
            int u_stop = min((int)ceilf(hi_u + ru), nu);
            int v_start = max((int)floorf(lo_v - rv) + 1, 0);
            int v_stop = min((int)ceilf(hi_v + rv), nv);
            const float* image = images + (long)(first + view) * nu * nv;
            for (int u = u_start; u < u_stop; ++u) {
                for (int v = v_start; v < v_stop; ++v) {
                    float value = image[(long)u * nv + v] * weight;
                    // Round each product as the forward projection does; weights
                    // must match it exactly to keep the transpose matched.
                    float pb = __fadd_rn(__fmul_rn(ub, (float)u), __fmul_rn(vb, (float)v));
                    float pc = __fadd_rn(__fmul_rn(uc, (float)u), __fmul_rn(vc, (float)v));
                    #pragma unroll
                    for (int j = 0; j < R; ++j) {
                        float qb = __fadd_rn(pb, hb[j]), qc = __fadd_rn(pc, hc[j]);
                        total[j] += value * (fmaxf(1.f - fabsf(qb - tb[j]), 0.f)
                                             * fmaxf(1.f - fabsf(qc - tc[j]), 0.f));
                    }
                }
            }
        }
    }
    if (valid) {
        long base = ((long)ix * ny + iy) * nz + iz0;
        #pragma unroll
        for (int j = 0; j < R; ++j)
            if (iz0 + j < nz) out[base + j] += total[j];
    }
}
"""


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
    device = jax.devices()[0]
    if jax.default_backend() != "gpu" or "cuda" not in device.client.platform_version.lower():
        return False
    return importlib.util.find_spec("cupy") is not None


@cache
def _kernel() -> Any:
    import cupy as cp

    source = _SOURCE.replace("RUN", str(_RUN)).replace("VIEWS", str(_VIEWS_PER_STAGE))
    return cp.RawModule(code=source).get_function("joseph_gather")


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
        runs = nx * ny * -(-nz // _RUN)
        _kernel()(
            (-(-runs // _THREADS),),
            (_THREADS,),
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
