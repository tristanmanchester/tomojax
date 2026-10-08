"""Feldkamp-Davis-Kress (FDK) reconstruction for cone-beam scans.

Each projection is cosine-weighted, ramp-filtered along detector rows on the
isocentre plane, and backprojected voxel by voxel with the ``(SOD / U)^2``
distance weight, where U is the voxel's distance from the source along the
detector normal. Full turns weight each view by half its angular step;
shorter arcs of at least a half turn plus the fan angle use Parker weights.

FDK is exact only in the plane through the source orbit; away from it the
reconstruction carries the usual cone-beam artefacts, growing with the cone
angle. It is a fast first reconstruction and initialiser for the iterative
solvers, which model the cone geometry exactly.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache, partial
import logging
import math
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cone import cone_coefficients, use_cuda_cone, xla_stream
from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.geometry.cone import beam_of
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.recon.filters import get_fbp_filter_np

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry, Grid
    from tomojax.core.geometry.cone import ConeBeam

LOG = logging.getLogger(__name__)

_RUN = 8
# The CUDA path filters rows up to this wide by matrix product (see _filter_operator):
# on an RTX 4070 that beats the FFTs 3x at 768 pixels and breaks even near 2500.
_DENSE_FILTER_MAX = 2048
_TILE = (8, 2)


@dataclass(frozen=True, kw_only=True)
class FDKConfig:
    """FDK options.

    ``filter`` is ``ramp``, ``shepp-logan`` or ``hann``. ``backend``
    ``auto`` uses the CUDA kernel when CuPy and a CUDA device are available.
    ``views_per_batch`` bounds the filtered views held on the device at once.
    """

    filter: str = "ramp"
    backend: str = "auto"
    views_per_batch: int = 64


def _fan_angles(beam: ConeBeam, detector: Detector) -> np.ndarray:
    """Fan angle of each detector column from the ray through the rotation axis."""
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.det_center[0]
    return np.arctan(u / float(beam.source_to_detector)) - np.arctan(
        float(beam.axis_offset) / float(beam.source_to_axis)
    )


def _full_turn_column_weights(beam: ConeBeam, detector: Detector) -> np.ndarray:
    """``(nu,)`` weights of a full turn: 1/2, or Wang's for an offset detector.

    A full turn measures each ray twice, once from each side, so each column
    takes half the angular measure. When the detector is offset so that the
    rotation axis projects off its centre, columns beyond the short side's
    reach are measured once and weigh 1, and a sin^2 ramp across the overlap
    (0 at the short edge, 1/2 at the axis, 1 at the mirrored column) keeps the
    weights of each ray's two measurements summing to one.
    """
    gamma = _fan_angles(beam, detector)
    lo, hi = float(gamma.min()), float(gamma.max())
    pixel = float(detector.du) / float(beam.source_to_detector)
    if lo > pixel or hi < -pixel:
        raise ValueError("FDK needs the rotation axis to project onto the detector")
    if abs(hi + lo) <= pixel:
        return np.full(detector.nu, 0.5)
    side = 1.0 if hi + lo > 0 else -1.0
    overlap = max(min(-lo, hi), pixel)
    x = side * gamma
    ramp = np.sin(np.pi / 4 * (1 + np.clip(x / overlap, -1.0, 1.0))) ** 2
    return np.where(x >= overlap, 1.0, ramp)


def _full_turn(geometry: Geometry, n_views: int) -> bool:
    thetas = getattr(geometry, "thetas_deg", None)
    if thetas is None or n_views < 2:
        return False
    angles = np.sort(np.deg2rad(np.asarray(thetas, dtype=np.float64)[:n_views]))
    step = float(np.median(np.diff(angles)))
    return float(angles[-1] - angles[0]) + step >= 2 * np.pi - 0.5 * step


def _virtual_columns(
    geometry: Geometry, beam: ConeBeam, detector: Detector, n_views: int
) -> tuple[int, int]:
    """Columns to add below and above an offset detector, mirroring it about the axis.

    In a full turn an offset detector reconstructs the whole field its long
    side covers; voxels on the short side then project past the detector, where
    the filtered rows (zero data, nonzero ramp tail) must still be read.
    """
    if not _full_turn(geometry, n_views) or not _windowable(beam):
        return 0, 0
    axis = (float(beam.axis_offset) * beam.magnification - float(detector.det_center[0])) / float(
        detector.du
    ) + (detector.nu - 1) / 2
    below, above = axis, detector.nu - 1 - axis
    if abs(above - below) <= 1.0:
        return 0, 0
    pad = min(math.ceil(abs(above - below)), detector.nu)
    return (pad, 0) if above > below else (0, pad)


def _extended(beam: ConeBeam, detector: Detector, pad_lo: int, pad_hi: int) -> Detector:
    """``detector`` with ``pad_lo`` columns added below u and ``pad_hi`` above."""
    from dataclasses import replace

    if not (pad_lo or pad_hi):
        return detector
    _, u_dir, _ = beam.detector_frame(detector)
    shift = (pad_hi - pad_lo) / 2 * float(detector.du) * u_dir
    return replace(
        detector,
        nu=detector.nu + pad_lo + pad_hi,
        det_center=(detector.det_center[0] + shift[0], detector.det_center[1] + shift[2]),
    )


def view_weights(geometry: Geometry, detector: Detector, n_views: int) -> np.ndarray:
    """Return ``(views, nu)`` FDK angular weights.

    Full turns weigh each view by its angular measure, halved (Wang's weights
    for an offset detector); short scans use Parker weights. Views repeating
    an angle (several turns, at several heights say) share its measure, so a
    scan of several turns averages their reconstructions.
    """
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("view_weights needs a cone-beam geometry")
    thetas = getattr(geometry, "thetas_deg", None)
    if thetas is None:
        raise ValueError("FDK needs a geometry with rotation angles (thetas_deg)")
    all_angles = np.deg2rad(np.asarray(thetas, dtype=np.float64)[:n_views])
    # Angles a whole number of turns apart, to a millionth of a turn, are one angle.
    key = np.round(all_angles / (2 * np.pi) * 1e6).astype(np.int64) % 1_000_000
    _, first, view_angle, repeats = np.unique(
        key, return_index=True, return_inverse=True, return_counts=True
    )
    weights = _distinct_view_weights(beam, detector, all_angles[first])
    return weights[view_angle] / repeats[view_angle][:, None]


def _distinct_view_weights(beam: ConeBeam, detector: Detector, angles: np.ndarray) -> np.ndarray:
    """:func:`view_weights` for views at distinct ``angles`` (radians)."""
    n_views = len(angles)
    order = np.argsort(angles)
    sorted_angles = angles[order]
    gaps = np.diff(sorted_angles)
    if n_views < 2 or np.any(gaps <= 0):
        raise ValueError("FDK needs at least two distinct rotation angles")
    step = float(np.median(gaps))
    arc = float(sorted_angles[-1] - sorted_angles[0]) + step
    # Angular measure of each view: half the gap to each neighbour.
    measure = np.empty(n_views)
    if arc >= 2 * np.pi - 0.5 * step:
        wrapped = np.concatenate(
            [sorted_angles[-1:] - 2 * np.pi, sorted_angles, sorted_angles[:1] + 2 * np.pi]
        )
        measure[order] = 0.5 * (wrapped[2:] - wrapped[:-2])
        return measure[:, None] * _full_turn_column_weights(beam, detector)[None, :]
    padded = np.concatenate([[sorted_angles[0] - step], sorted_angles, [sorted_angles[-1] + step]])
    measure[order] = 0.5 * (padded[2:] - padded[:-2])
    # Parker weights over the fan angle of each detector column.
    gamma = _fan_angles(beam, detector)
    pixel = float(detector.du) / float(beam.source_to_detector)
    if abs(float(gamma.max() + gamma.min())) > 0.05 * float(gamma.max() - gamma.min()):
        LOG.warning(
            "FDK short scan with an offset detector: Parker weights assume the axis "
            "projects near the detector centre (fan %.2f to %.2f degrees); columns beyond "
            "the short side are weighted as if their conjugate rays were measured",
            np.rad2deg(gamma.min() - pixel / 2),
            np.rad2deg(gamma.max() + pixel / 2),
        )
    delta = float(np.max(np.abs(gamma)))
    if arc < np.pi + 2 * delta - 1e-6:
        raise ValueError(
            f"FDK short scans need at least 180 degrees plus the fan angle "
            f"({np.rad2deg(np.pi + 2 * delta):.1f} degrees); got {np.rad2deg(arc):.1f}"
        )
    beta = (angles - sorted_angles[0])[:, None]
    g = gamma[None, :]
    weight = np.ones((n_views, detector.nu))
    rise = beta < 2 * (delta - g)
    weight = np.where(rise, np.sin(np.pi / 4 * beta / np.maximum(delta - g, 1e-12)) ** 2, weight)
    end = np.pi + 2 * delta
    fall = beta > np.pi - 2 * g
    weight = np.where(
        fall, np.sin(np.pi / 4 * (end - beta) / np.maximum(delta + g, 1e-12)) ** 2, weight
    )
    weight = np.where(beta > end, 0.0, weight)
    return weight * measure[:, None]


def _cosine_weights(beam: ConeBeam, detector: Detector) -> np.ndarray:
    """``(nv, nu)`` cosine of each pixel's ray to the detector normal."""
    centre, u_dir, v_dir = beam.detector_frame(detector)
    normal = np.cross(u_dir, v_dir)
    normal /= np.linalg.norm(normal)
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv
    pixel = centre + u[None, :, None] * u_dir + v[:, None, None] * v_dir
    ray = pixel - beam.source()
    return np.abs(ray @ normal) / np.linalg.norm(ray, axis=-1)


def _filter_operator(spectrum: np.ndarray, nu: int, pad_lo: int, pad_hi: int) -> np.ndarray:
    """The ``(nu, pad_lo + nu + pad_hi)`` matrix that ramp-filters rows as ``spectrum`` does.

    Filtering is a circular convolution with the zero-padded row; as a matrix
    product it runs on the GPU's matrix units several times faster than the
    FFTs for rows up to a couple of thousand pixels.
    """
    n_fft = 2 * (spectrum.shape[0] - 1)
    impulse = np.fft.irfft(spectrum.astype(np.float64), n=n_fft)
    columns = np.arange(-pad_lo, nu + pad_hi)
    return impulse[(columns[None, :] - np.arange(nu)[:, None]) % n_fft].astype(np.float32)


def _filter(
    views: jax.Array,
    cosine: jax.Array,
    weights: jax.Array,
    kernel: jax.Array,
    pad_lo: int = 0,
    pad_hi: int = 0,
) -> jax.Array:
    """Weight and ramp-filter rows, keeping ``pad_lo``/``pad_hi`` columns beyond them.

    Filtered rows extend past the data; an offset detector's backprojection
    needs that tail on its short side (see :func:`_virtual_columns`).
    """
    # Angular (Parker, offset-detector) weights vary along the row, so they
    # apply before the ramp filter, as do the cosine weights.
    rows = views * cosine * weights[:, None, :]
    n_fft = 2 * (int(kernel.shape[0]) - 1)
    nu = views.shape[-1]
    spectrum = jnp.fft.rfft(rows, n=n_fft, axis=-1) * kernel
    out = jnp.fft.irfft(spectrum, n=n_fft, axis=-1)
    if pad_lo:
        return jnp.concatenate([out[..., n_fft - pad_lo :], out[..., : nu + pad_hi]], axis=-1)
    return out[..., : nu + pad_hi]


def _backproject_jax(
    filtered: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    scale: float,
    out: jax.Array,
) -> jax.Array:
    """Voxel-driven backprojection of filtered ``(views, nv, nu)`` images."""
    index = [jnp.arange(n, dtype=jnp.float32) for n in (grid.nx, grid.ny, grid.nz)]
    qx, qy, qz = jnp.meshgrid(*index, indexing="ij")

    def step(total: jax.Array, view: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, None]:
        image, c = view
        d0, d1, d2 = qx - c[0], qy - c[1], qz - c[2]
        lam = c[16] / (c[13] * d0 + c[14] * d1 + c[15] * d2)
        u = c[23] + lam * (c[17] * d0 + c[18] * d1 + c[19] * d2)
        v = c[24] + lam * (c[20] * d0 + c[21] * d1 + c[22] * d2)
        u0, v0 = jnp.floor(u), jnp.floor(v)
        value = jnp.zeros_like(u)
        for du in (0.0, 1.0):
            for dv in (0.0, 1.0):
                ui, vi = u0 + du, v0 + dv
                w = (1 - jnp.abs(u - ui)) * (1 - jnp.abs(v - vi))
                valid = (ui >= 0) & (ui < detector.nu) & (vi >= 0) & (vi < detector.nv)
                sample = image[
                    jnp.clip(vi, 0, detector.nv - 1).astype(jnp.int32),
                    jnp.clip(ui, 0, detector.nu - 1).astype(jnp.int32),
                ]
                value = value + jnp.where(valid, sample * w, 0.0)
        return total + value * (lam * scale) ** 2, None

    total, _ = jax.lax.scan(step, out, (filtered, coeff))
    return total


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


def _backproject_cuda(
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


@partial(
    jax.jit,
    static_argnames=("grid", "detector", "scale", "cuda", "pad_lo", "pad_hi"),
    donate_argnames=("out",),
)
def _fdk_batch(
    views: jax.Array,
    start: jax.Array | int,
    coeff: jax.Array,
    cosine: jax.Array,
    weights: jax.Array,
    kernel: jax.Array,
    operator: jax.Array | None,
    out: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    scale: float,
    cuda: bool,
    pad_lo: int,
    pad_hi: int,
) -> jax.Array:
    # The batch's views of the scan's coefficients and weights.
    coeff = jax.lax.dynamic_slice_in_dim(coeff, start, views.shape[0])
    weights = jax.lax.dynamic_slice_in_dim(weights, start, views.shape[0])
    if operator is not None:
        rows = views * cosine * weights[:, None, :]
        return _backproject_cuda(rows, coeff, grid, detector, scale, out, operator)
    filtered = _filter(views, cosine, weights, kernel, pad_lo, pad_hi)
    backproject = _backproject_cuda if cuda else _backproject_jax
    return backproject(filtered, coeff, grid, detector, scale, out)


@dataclass(frozen=True)
class _Prepared:
    """Per-scan FDK weights and filter on the device; see :func:`_prepare`."""

    cuda: bool
    weights: jax.Array
    cosine: jax.Array
    kernel: jax.Array
    scale: float
    batch: int
    pad_lo: int = 0
    pad_hi: int = 0
    # The filter as a matrix (see _filter_operator), for the CUDA kernel's rows.
    operator: jax.Array | None = None

    def backprojected(self, beam: ConeBeam, detector: Detector) -> Detector:
        """The detector the filtered rows cover, with its virtual columns."""
        return _extended(beam, detector, self.pad_lo, self.pad_hi)


def _prepare(
    geometry: Geometry,
    detector: Detector,
    n_views: int,
    cfg: FDKConfig,
    columns: Detector | None = None,
) -> _Prepared:
    """FDK weights, filter and virtual columns for ``detector``.

    ``columns``, a detector with the same columns (the full detector of a band
    of rows), sets the per-column angular weights and virtual columns, so every
    band of a rolled detector gets the same ones.
    """
    columns = detector if columns is None else columns
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("fdk needs a cone-beam geometry; use fbp for parallel beams")
    if cfg.backend not in {"auto", "jax", "cuda"}:
        raise ValueError("fdk backend must be 'auto', 'jax' or 'cuda'")
    cuda = use_cuda_cone() if cfg.backend == "auto" else cfg.backend == "cuda"
    if cuda and not use_cuda_cone():
        raise ValueError("fdk: the CUDA kernel needs CuPy on a CUDA device")
    du_iso = float(detector.du) / beam.magnification
    pad_lo, pad_hi = _virtual_columns(geometry, beam, columns, n_views)
    # The kernel spans the virtual row too, so its tail does not wrap around.
    width = detector.nu + pad_lo + pad_hi
    kernel = get_fbp_filter_np(cfg.filter, width, du_iso, "float32")
    operator = None
    if cuda and width <= _DENSE_FILTER_MAX:
        operator = jax.device_put(_filter_operator(kernel, detector.nu, pad_lo, pad_hi))
    return _Prepared(
        cuda=cuda,
        # device_put, unlike jnp.asarray, compiles nothing for a new shape.
        weights=jax.device_put(np.asarray(view_weights(geometry, columns, n_views), np.float32)),
        cosine=jax.device_put(np.asarray(_cosine_weights(beam, detector), np.float32)),
        kernel=jax.device_put(kernel),
        operator=operator,
        scale=1.0 / beam.magnification,
        batch=max(1, int(cfg.views_per_batch)),
        pad_lo=pad_lo,
        pad_hi=pad_hi,
    )


def _device_views(projections: jax.Array | np.ndarray, start: int, stop: int) -> jax.Array:
    part = projections[start:stop]
    if isinstance(part, jax.Array):
        return part.astype(jnp.float32)
    return jax.device_put(np.asarray(part, np.float32))


_filter_jit = jax.jit(_filter, static_argnames=("pad_lo", "pad_hi"))


@partial(jax.jit, static_argnames=("grid", "detector", "scale", "cuda"), donate_argnames=("out",))
def _backproject_batch(
    filtered: jax.Array,
    coeff: jax.Array,
    out: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    scale: float,
    cuda: bool,
) -> jax.Array:
    backproject = _backproject_cuda if cuda else _backproject_jax
    return backproject(filtered, coeff, grid, detector, scale, out)


def _filter_views(prep: _Prepared, projections: jax.Array | np.ndarray, n_views: int) -> jax.Array:
    """Weighted, ramp-filtered ``(views, nv, nu)`` projections on the device."""
    parts = []
    for start in range(0, n_views, prep.batch):
        stop = min(start + prep.batch, n_views)
        views = _device_views(projections, start, stop)
        weights = prep.weights[start:stop]
        parts.append(
            _filter_jit(views, prep.cosine, weights, prep.kernel, prep.pad_lo, prep.pad_hi)
        )
    return jnp.concatenate(parts, axis=0)


def _backproject_filtered(
    geometry: Geometry, grid: Grid, detector: Detector, filtered: jax.Array, prep: _Prepared
) -> jax.Array:
    """Backproject filtered projections from :func:`_filter_views` into ``grid``."""
    beam = beam_of(geometry)
    assert beam is not None
    n_views = int(filtered.shape[0])
    detector = prep.backprojected(beam, detector)
    coeff = cone_coefficients(stack_view_poses(geometry, n_views), grid, detector, beam)
    out = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    for start in range(0, n_views, prep.batch):
        stop = min(start + prep.batch, n_views)
        out = _backproject_batch(
            filtered[start:stop], coeff[start:stop], out,
            grid=grid, detector=detector, scale=prep.scale, cuda=prep.cuda,
        )  # fmt: skip
    return out


def fdk(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    config: FDKConfig | None = None,
) -> jax.Array:
    """Reconstruct a cone-beam scan with FDK; returns an ``(nx, ny, nz)`` volume.

    ``projections`` are ``(views, nv, nu)`` line integrals; NumPy and memmap
    stacks stream to the device ``views_per_batch`` views at a time. Full turns
    on an offset detector (the axis projecting off its centre) get Wang's
    weights and reconstruct the field the detector's long side covers.
    """
    return _fdk(geometry, grid, detector, projections, config=config)


def _fdk(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    config: FDKConfig | None = None,
    columns: Detector | None = None,
) -> jax.Array:
    cfg = FDKConfig() if config is None else config
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("fdk needs a cone-beam geometry; use fbp for parallel beams")
    validate_grid(grid, "fdk grid")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fdk projections"
    )
    prep = _prepare(geometry, detector, n_views, cfg, columns)
    virtual = prep.backprojected(beam, detector)
    coeff = cone_coefficients(stack_view_poses(geometry, n_views), grid, virtual, beam)
    out = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    starts = list(range(0, n_views, prep.batch))
    # Copy the next batch to the device while the current one is filtered and
    # backprojected.
    with ThreadPoolExecutor(max_workers=1) as copier:
        pending = copier.submit(_device_views, projections, 0, min(prep.batch, n_views))
        for index, start in enumerate(starts):
            views = pending.result()
            if index + 1 < len(starts):
                following = starts[index + 1]
                pending = copier.submit(
                    _device_views, projections, following, min(following + prep.batch, n_views)
                )
            out = _fdk_batch(
                views, start, coeff, prep.cosine, prep.weights, prep.kernel, prep.operator, out,
                grid=grid, detector=virtual, scale=prep.scale, cuda=prep.cuda,
                pad_lo=prep.pad_lo, pad_hi=prep.pad_hi,
            )  # fmt: skip
    return out


@dataclass(frozen=True, kw_only=True)
class FDKHostConfig:
    """Slab options for :func:`fdk_host`.

    ``slices_per_batch`` z slices are reconstructed per slab; ``None`` sizes slabs
    to about a third of the free device memory. ``fdk`` holds the filter and backend
    options.
    """

    slices_per_batch: int | None = None
    fdk: FDKConfig = field(default_factory=FDKConfig)


def _slab_rows(
    geometry: Geometry, grid: Grid, detector: Detector, poses: np.ndarray, z0: int, z1: int
) -> tuple[int, int]:
    """Detector rows that the z-slab ``[z0, z1)`` projects onto in any view."""
    beam = beam_of(geometry)
    assert beam is not None
    origin = np.asarray(grid_volume_origin(grid))
    spacing = np.asarray([grid.vx, grid.vy, grid.vz])
    lo = origin - spacing / 2
    hi = origin + (np.asarray([grid.nx, grid.ny, grid.nz]) - 0.5) * spacing
    zlo, zhi = origin[2] + (z0 - 1.5) * grid.vz, origin[2] + (z1 + 0.5) * grid.vz
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (zlo, zhi)]
    )
    # A box projects inside the hull of its projected corners.
    world = np.einsum("nij,kj->nki", poses[:, :3, :3], corners) + poses[:, None, :3, 3]
    centre, u_dir, v_dir = beam.detector_frame(detector)
    normal = np.cross(u_dir, v_dir)
    source = beam.source()
    ray = world - source
    hit = source + ray * (((centre - source) @ normal) / (ray @ normal))[..., None]
    rows = ((hit - centre) @ v_dir) / detector.dv + (detector.nv - 1) / 2
    return max(0, int(np.floor(rows.min())) - 1), min(detector.nv, int(np.ceil(rows.max())) + 2)


def _windowable(beam: ConeBeam) -> bool:
    """Whether a band of detector rows is itself a detector of ``beam`` (no pitch or yaw)."""
    return float(beam.detector_pitch_deg) == 0.0 and float(beam.detector_yaw_deg) == 0.0


def _detector_window(
    beam: ConeBeam,
    detector: Detector,
    r0: int,
    r1: int,
    c1: int | None = None,
    f: int = 1,
) -> Detector:
    """Rows ``[r0, r1)`` and columns ``[0, c1)`` of ``detector``, binned ``f x f``.

    The window keeps its place on the rolled detector: its centre moves along
    the detector's own u and v. Needs :func:`_windowable` beams, whose detector
    axes stay in the plane of ``det_center``.
    """
    from dataclasses import replace

    rows, cols = (r1 - r0) // f, (detector.nu if c1 is None else c1) // f
    su = ((f * cols - 1) / 2 - (detector.nu - 1) / 2) * detector.du
    sv = (r0 + (f * rows - 1) / 2 - (detector.nv - 1) / 2) * detector.dv
    _, u_dir, v_dir = beam.detector_frame(detector)
    shift = su * u_dir + sv * v_dir
    return replace(
        detector,
        nu=cols,
        nv=rows,
        du=detector.du * f,
        dv=detector.dv * f,
        det_center=(detector.det_center[0] + shift[0], detector.det_center[1] + shift[2]),
    )


def _slab_grid(grid: Grid, z0: int, z1: int) -> Grid:
    """Grid of z slices ``[z0, z1)``."""
    from dataclasses import replace

    origin = grid_volume_origin(grid)
    return replace(
        grid,
        nz=z1 - z0,
        vol_origin=(origin[0], origin[1], origin[2] + z0 * grid.vz),
        vol_center=None,
    )


def fdk_host(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    *,
    config: FDKHostConfig | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """FDK with host input and output, reconstructed in z slabs on the device.

    Accepts NumPy arrays and memmaps; ``out`` may be a writable ``(nx, ny, nz)``
    FP32 memmap. Each slab filters only the detector rows it projects onto (all
    rows for a pitched or yawed detector), so projections and volume can both
    exceed device memory.
    """
    from tomojax.backends import device_free_memory_bytes

    cfg = FDKHostConfig() if config is None else config
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("fdk_host needs a cone-beam geometry; use fbp_host for parallel beams")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fdk_host projections"
    )
    shape = (grid.nx, grid.ny, grid.nz)
    result = np.empty(shape, np.float32) if out is None else out
    if tuple(result.shape) != shape:
        raise ValueError(f"fdk_host out must have shape {shape}")
    depth = cfg.slices_per_batch
    if depth is None:
        free = device_free_memory_bytes() or 2 * 1024**3
        depth = max(1, int(free // (3 * max(1, 4 * grid.nx * grid.ny))))
    depth = min(int(depth), grid.nz)
    poses = np.asarray(stack_view_poses(geometry, n_views), np.float64)
    windowed = _windowable(beam)
    # Slab copies into ``result`` (slow for memmaps) overlap the next slab's work.
    pending: Future[None] | None = None

    def store(z0: int, z1: int, volume: np.ndarray) -> None:
        result[:, :, z0:z1] = volume

    with ThreadPoolExecutor(max_workers=1) as writer:
        for z0 in range(0, grid.nz, depth):
            z1 = min(z0 + depth, grid.nz)
            slab = _slab_grid(grid, z0, z1)
            if windowed:
                r0, r1 = _slab_rows(geometry, slab, detector, poses, 0, z1 - z0)
                rows = _detector_window(beam, detector, r0, r1)
            else:
                r0, r1, rows = 0, detector.nv, detector
            # The full detector sets per-column weights, the same for every slab.
            volume = _fdk(
                geometry, slab, rows, _RowView(projections, r0, r1), config=cfg.fdk,
                columns=detector,
            )  # fmt: skip
            host = np.asarray(volume)
            if pending is not None:
                pending.result()
            pending = writer.submit(store, z0, z1, host)
        if pending is not None:
            pending.result()
    return result


class _RowView:
    """Lazy ``projections[:, r0:r1]`` that slices views first, so memmaps stay on disk."""

    def __init__(self, projections: np.ndarray, r0: int, r1: int) -> None:
        self.projections, self.r0, self.r1 = projections, r0, r1
        self.shape = (projections.shape[0], r1 - r0, projections.shape[2])
        self.dtype = np.dtype(np.float32)
        self.ndim = 3

    def __getitem__(self, views: slice) -> np.ndarray:
        return np.asarray(self.projections[views, self.r0 : self.r1], np.float32)

    def __len__(self) -> int:
        return self.shape[0]


__all__ = ["FDKConfig", "FDKHostConfig", "fdk", "fdk_host", "view_weights"]
