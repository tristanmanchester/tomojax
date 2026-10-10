"""The CUDA FDK backprojector: texture-filtered half-float views, voxel by voxel.

``buffer_callback`` launches it on XLA's stream of the GPU holding the buffers,
so each of several GPUs runs its own launches (see :func:`xla_stream`). Its
batches of views may also be filtered first, by a cuBLAS matrix product.
"""

from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cone import xla_stream

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid

_RUN = 8
_TILE = (8, 2)


_SOURCE = r"""
#include <cuda_fp16.h>
#define NC 25
#define RUN 8
#define TX 8
#define TY 2
#define HALF_PEAK 32768.f

// The largest |x| of n values, into *peak (zeroed beforehand): nonnegative floats
// order as their bits do.
extern "C" __global__ void abs_peak(const float* __restrict__ x, long long n, float* peak)
{
    float m = 0.f;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(x[i]));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax((int*)peak, __float_as_int(m));
}

// Filtered (rows, nv) images as half textures read them: scaled so the peak is
// HALF_PEAK (half floats reach 65504), each row led by `lead` zeros and padded to `pitch`.
extern "C" __global__ void to_half(
    const float* __restrict__ src, const float* __restrict__ peak, __half* __restrict__ dst,
    int nv, int pitch, int lead, long long total)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    long long row = i / pitch;
    int v = (int)(i - row * pitch) - lead;
    float gain = *peak > 0.f ? HALF_PEAK / *peak : 1.f;
    dst[i] = __float2half_rn(v >= 0 && v < nv ? src[row * nv + v] * gain : 0.f);
}

// Filtered images (view, u, v), from to_half, are sampled through one texture per
// view: the texture unit interpolates bilinearly (weights to 1/256, as ASTRA's FDK)
// and returns zero beyond the detector; detector row v is texture column
// v + vshift - 0.5. A warp covers 32 consecutive z voxels of one (x, y) column and
// each lane steps through RUN of them 32 apart, so a warp's samples run down an image
// column. A block holds a TX x TY tile of columns, which share the texture cache.
extern "C" __global__ void fdk_backproject(
    const float* __restrict__ coeff, const unsigned long long* __restrict__ textures,
    const float* __restrict__ peak, float* __restrict__ vol, int nviews, int nx, int ny,
    int nz, float vshift, float scale2)
{
    __shared__ float cf[32 * NC];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int groups = (nz + 32 * RUN - 1) / (32 * RUN), tiles_y = (ny + TY - 1) / TY;
    int tile = blockIdx.x / groups, group = blockIdx.x % groups;
    int ix = (tile / tiles_y) * TX + wid / TY, iy = (tile % tiles_y) * TY + wid % TY;
    bool valid = ix < nx && iy < ny;
    if (!valid) ix = iy = 0;
    int iz0 = 32 * RUN * group + lane;
    // Steps j holding any of the warp's voxels (fewer for thin slabs); warp-uniform.
    int runs = min(RUN, (nz - 32 * RUN * group + 31) / 32);
    float acc[RUN];
    #pragma unroll
    for (int j = 0; j < RUN; ++j) acc[j] = 0.f;
    for (int first = 0; first < nviews; first += 32) {
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * NC; i += blockDim.x)
            cf[i] = first * NC + i < nviews * NC ? coeff[(long)first * NC + i] : 0.f;
        __syncthreads();
        if (!valid) continue;
        int count = min(32, nviews - first);
        for (int w = 0; w < count; ++w) {
            const float* c = cf + w * NC;
            cudaTextureObject_t image = textures[first + w];
            float d0 = (float)ix - c[0], d1 = (float)iy - c[1];
            if (c[15] == 0.f && c[19] == 0.f) {
                // Detector normal and u axis have no z part (turntable views): the
                // depth factor and column are fixed along the column, v is linear in z.
                float lam = c[16] / (c[13] * d0 + c[14] * d1);
                float u = c[23] + lam * (c[17] * d0 + c[18] * d1) + 0.5f;
                float gg = lam * lam;
                float vbase = c[24] + lam * (c[20] * d0 + c[21] * d1 + c[22] * ((float)iz0 - c[2]));
                float vstep = 32.f * lam * c[22];
                #pragma unroll
                for (int j = 0; j < RUN; ++j) {
                    if (j >= runs) break;
                    acc[j] += tex2D<float>(image, vbase + (float)j * vstep + vshift, u) * gg;
                }
                continue;
            }
            // Other views: the x and y parts are fixed along the column; a fast
            // reciprocal (2 ulp) is ample for a filtered backprojection.
            float n01 = c[13] * d0 + c[14] * d1, u01 = c[17] * d0 + c[18] * d1;
            float v01 = c[20] * d0 + c[21] * d1;
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                if (j >= runs) break;
                float d2 = (float)(iz0 + 32 * j) - c[2];
                float lam = __fdividef(c[16], n01 + c[15] * d2);
                float u = c[23] + lam * (u01 + c[19] * d2);
                float v = c[24] + lam * (v01 + c[22] * d2);
                acc[j] += tex2D<float>(image, v + vshift, u + 0.5f) * (lam * lam);
            }
        }
    }
    if (valid) {
        if (*peak > 0.f) scale2 *= *peak / HALF_PEAK;
        long base = ((long)ix * ny + iy) * nz;
        #pragma unroll
        for (int j = 0; j < RUN; ++j)
            if (iz0 + 32 * j < nz) vol[base + iz0 + 32 * j] += acc[j] * scale2;
    }
}
"""


@cache
def _module() -> Any:
    import cupy as cp

    return cp.RawModule(code=_SOURCE)


# Per GPU, texture objects whose kernels have finished, freed on that GPU's next
# launch (CUDA calls are not allowed in the stream callback that retires them).
_RETIRED: dict[int, list[object]] = {}
# The texture start alignment CUDA requires (cudaDeviceProp.textureAlignment), and
# the half pixels in it.
_ALIGN = 512
_LEAD = _ALIGN // 2


@cache
def _check_alignment(device: int) -> None:
    """Raise unless ``_ALIGN`` meets this GPU's texture start and pitch alignment."""
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(device)
    for key in ("textureAlignment", "texturePitchAlignment"):
        need = int(props[key])
        if need > _ALIGN or _ALIGN % need:
            raise RuntimeError(f"FDK textures start on {_ALIGN} bytes; this GPU needs {key} {need}")


def _textures(images: Any, width: int) -> tuple[list[Any], int]:
    """One bilinear, zero-bordered texture per ``(rows, pitch)`` half image of ``images``.

    Textures must start on a ``_ALIGN`` boundary, but XLA's buffers need not, so each
    texture starts up to ``_LEAD`` pixels into its image's rows, which lead with that
    many zeros; returns the textures and that start.
    """
    import cupy as cp

    runtime, texture = cp.cuda.runtime, cp.cuda.texture
    channel = texture.ChannelFormatDescriptor(16, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    sampling = texture.TextureDescriptor(
        (runtime.cudaAddressModeBorder,) * 2,
        runtime.cudaFilterModeLinear,
        runtime.cudaReadModeElementType,
        borderColors=(0, 0, 0, 0),
    )
    _check_alignment(cp.cuda.Device().id)
    rows, pitch = int(images.shape[1]), int(images.shape[2])
    start = -images.data.ptr % _ALIGN // 2
    textures = [
        texture.TextureObject(
            texture.ResourceDescriptor(
                runtime.cudaResourceTypePitch2D,
                arr=image[0, start:],
                chDesc=channel,
                width=_LEAD - start + width,
                height=rows,
                pitchInBytes=2 * pitch,
            ),
            sampling,
        )
        for image in images
    ]
    return textures, start


def _filter_into(stream: Any, images: Any, rows: Any, operator: Any) -> None:
    """``images[k] = operator.T @ rows[k].T``: filtered rows as ``(view, u, v)`` images.

    cuBLAS, called directly: XLA would autotune the product for seconds on first use.
    """
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    handle = cp.cuda.Device().cublas_handle
    cublas.setStream(handle, stream.ptr)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    views, nv, nu = rows.shape
    width = operator.shape[1]
    one, zero = np.ones(1, np.float32), np.zeros(1, np.float32)
    # Column-major, images[k] is (nv, width) = rows[k] (nv, nu) @ operator (nu, width).
    cublas.sgemmStridedBatched(
        handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_T, nv, width, nu,
        one.ctypes.data, rows.data.ptr, nu, nv * nu, operator.data.ptr, width, 0,
        zero.ctypes.data, images.data.ptr, nv, nv * width, views,
    )  # fmt: skip


def _to_half(images: Any, half: Any) -> Any:
    """Convert ``(view, u, v)`` images into ``half``'s texture layout; returns their peak."""
    import cupy as cp

    module = _module()
    peak = cp.zeros(1, cp.float32)
    module.get_function("abs_peak")((256,), (256,), (images, np.int64(images.size), peak))
    total = half.size
    module.get_function("to_half")(
        (-(-total // 256),), (256,),
        (images, peak, half, np.int32(images.shape[2]), np.int32(half.shape[2]),
         np.int32(_LEAD), np.int64(total)),
    )  # fmt: skip
    return peak


def _launch(context: Any, out: Any, *buffers: Any, grid: Grid, det: Detector, scale: float) -> None:
    import cupy as cp

    with xla_stream(context, out[0]) as stream:
        retired = _RETIRED.setdefault(cp.cuda.Device().id, [])  # the buffers' GPU
        retired.clear()
        if len(out) == 3:  # unfiltered rows and the filter's operator
            target, half, images = (cp.asarray(b) for b in out)
            coeff, rows, operator, initial = (cp.asarray(b) for b in buffers)
            _filter_into(stream, images, rows, operator)
        else:
            target, half = (cp.asarray(b) for b in out)
            coeff, images, initial = (cp.asarray(b) for b in buffers)
        if target.data.ptr != initial.data.ptr:
            target[...] = initial
        peak = _to_half(images, half)
        textures, start = _textures(half, det.nv)
        handles = cp.asarray([t.ptr for t in textures], cp.uint64)
        tiles = -(-grid.nx // _TILE[0]) * -(-grid.ny // _TILE[1]) * -(-grid.nz // (32 * _RUN))
        _module().get_function("fdk_backproject")(
            (tiles,),
            (32 * _TILE[0] * _TILE[1],),
            (
                coeff, handles, peak, target, np.int32(coeff.shape[0]),
                np.int32(grid.nx), np.int32(grid.ny), np.int32(grid.nz),
                np.float32(_LEAD - start + 0.5), np.float32(scale**2),
            ),
        )  # fmt: skip
        stream.launch_host_func(retired.append, (textures, handles, peak))


def backproject_cuda(
    filtered: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    scale: float,
    out: jax.Array,
    operator: jax.Array | None = None,
) -> jax.Array:
    """Add the backprojection of ``(view, nv, width)`` filtered rows to ``out``.

    With an ``operator`` (see :func:`_filter_operator`), ``filtered`` are the
    weighted rows it filters, and the kernel filters them first. Either way the
    images become half floats for the texture unit, which filters those twice as
    fast; each batch is scaled to its peak, so they keep 11 significant bits.
    """
    views, width = int(filtered.shape[0]), int(filtered.shape[2])
    if operator is not None:
        width = int(operator.shape[1])
    # Rows lead with _LEAD zeros (see _textures) and pad to a multiple of _LEAD
    # pixels, so every view's image aligns alike.
    pitch = -(-(_LEAD + detector.nv) // _LEAD) * _LEAD
    shapes = (
        jax.ShapeDtypeStruct((grid.nx, grid.ny, grid.nz), jnp.float32),
        jax.ShapeDtypeStruct((views, width, pitch), jnp.float16),
    )
    launch = partial(_launch, grid=grid, det=detector, scale=float(scale))
    if operator is None:
        call = buffer_callback(launch, shapes, input_output_aliases={2: 0})
        return call(coeff, jnp.swapaxes(filtered, 1, 2), out)[0]
    images = jax.ShapeDtypeStruct((views, width, detector.nv), jnp.float32)
    call = buffer_callback(launch, (*shapes, images), input_output_aliases={3: 0})
    return call(coeff, filtered, operator, out)[0]
