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

from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cone import cone_coefficients, use_cuda_cone, xla_stream
from tomojax.core.geometry.cone import beam_of
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.recon.filters import get_fbp_filter_np

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry, Grid
    from tomojax.core.geometry.cone import ConeBeam

_RUN = 8


@dataclass(frozen=True)
class FDKConfig:
    """FDK options.

    ``filter_name`` is ``ramp``, ``shepp-logan`` or ``hann``. ``backend``
    ``auto`` uses the CUDA kernel when CuPy and a CUDA device are available.
    ``views_per_batch`` bounds the filtered views held on the device at once.
    """

    filter_name: str = "ramp"
    backend: str = "auto"
    views_per_batch: int = 64


def view_weights(geometry: Geometry, detector: Detector, n_views: int) -> np.ndarray:
    """Return ``(views, nu)`` FDK angular weights: half steps, or Parker for short scans."""
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("view_weights needs a cone-beam geometry")
    thetas = getattr(geometry, "thetas_deg", None)
    if thetas is None:
        raise ValueError("FDK needs a geometry with rotation angles (thetas_deg)")
    angles = np.deg2rad(np.asarray(thetas, dtype=np.float64)[:n_views])
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
        return np.repeat(0.5 * measure[:, None], detector.nu, axis=1)
    padded = np.concatenate([[sorted_angles[0] - step], sorted_angles, [sorted_angles[-1] + step]])
    measure[order] = 0.5 * (padded[2:] - padded[:-2])
    # Parker weights over the fan angle of each detector column.
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.det_center[0]
    gamma = np.arctan(u / float(beam.source_to_detector))
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


def _filter(
    views: jax.Array, cosine: jax.Array, weights: jax.Array, kernel: jax.Array
) -> jax.Array:
    n_fft = 2 * (int(kernel.shape[0]) - 1)
    rows = views * cosine
    spectrum = jnp.fft.rfft(rows, n=n_fft, axis=-1) * kernel
    filtered = jnp.fft.irfft(spectrum, n=n_fft, axis=-1)[..., : views.shape[-1]]
    return filtered * weights[:, None, :]


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
#define NC 25
#define RUN 8
// filtered images (view, u, v); one thread per RUN consecutive z voxels.
extern "C" __global__ void fdk_backproject(
    const float* __restrict__ coeff, const float* __restrict__ img, float* __restrict__ vol,
    int nviews, int nx, int ny, int nz, int nu, int nv, float scale)
{
    __shared__ float cf[32 * NC];
    long run = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int runs_z = (nz + RUN - 1) / RUN;
    bool valid = run < (long)nx * ny * runs_z;
    int ix = valid ? run / ((long)ny * runs_z) : 0;
    int iy = valid ? (run / runs_z) % ny : 0;
    int iz0 = valid ? RUN * (int)(run % runs_z) : 0;
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
            const float* image = img + (long)(first + w) * nu * nv;
            float d0 = (float)ix - c[0], d1 = (float)iy - c[1];
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                float d2 = (float)(iz0 + j) - c[2];
                float lam = c[16] / (c[13] * d0 + c[14] * d1 + c[15] * d2);
                float u = c[23] + lam * (c[17] * d0 + c[18] * d1 + c[19] * d2);
                float v = c[24] + lam * (c[20] * d0 + c[21] * d1 + c[22] * d2);
                float fu = floorf(u), fv = floorf(v);
                int u0 = (int)fu, v0 = (int)fv;
                float wu = u - fu, wv = v - fv;
                float s = 0.f;
                if (u0 >= 0 && u0 < nu) {
                    const float* col = image + (long)u0 * nv;
                    if (v0 >= 0 && v0 < nv) s += (1.f - wu) * (1.f - wv) * __ldg(col + v0);
                    if (v0 + 1 >= 0 && v0 + 1 < nv) s += (1.f - wu) * wv * __ldg(col + v0 + 1);
                }
                if (u0 + 1 >= 0 && u0 + 1 < nu) {
                    const float* col = image + (long)(u0 + 1) * nv;
                    if (v0 >= 0 && v0 < nv) s += wu * (1.f - wv) * __ldg(col + v0);
                    if (v0 + 1 >= 0 && v0 + 1 < nv) s += wu * wv * __ldg(col + v0 + 1);
                }
                float g = lam * scale;
                acc[j] += s * g * g;
            }
        }
    }
    if (valid) {
        long base = ((long)ix * ny + iy) * nz + iz0;
        #pragma unroll
        for (int j = 0; j < RUN; ++j) if (iz0 + j < nz) vol[base + j] += acc[j];
    }
}
"""


@cache
def _module() -> Any:
    import cupy as cp

    return cp.RawModule(code=_SOURCE)


def _launch(
    context: Any, out: Any, coeff: Any, images: Any, accumulate: Any, *, grid: Grid, det: Detector,
    scale: float,
) -> None:  # fmt: skip
    import cupy as cp

    with xla_stream(context):
        target, initial = cp.asarray(out), cp.asarray(accumulate)
        if target.data.ptr != initial.data.ptr:
            target[...] = initial
        runs = grid.nx * grid.ny * -(-grid.nz // _RUN)
        _module().get_function("fdk_backproject")(
            (-(-runs // 128),),
            (128,),
            (
                cp.asarray(coeff), cp.asarray(images), target, np.int32(coeff.shape[0]),
                np.int32(grid.nx), np.int32(grid.ny), np.int32(grid.nz),
                np.int32(det.nu), np.int32(det.nv), np.float32(scale),
            ),
        )  # fmt: skip


def _backproject_cuda(
    filtered: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    scale: float,
    out: jax.Array,
) -> jax.Array:
    call = buffer_callback(
        partial(_launch, grid=grid, det=detector, scale=float(scale)),
        jax.ShapeDtypeStruct((grid.nx, grid.ny, grid.nz), jnp.float32),
        input_output_aliases={2: 0},
    )
    return call(coeff, jnp.swapaxes(filtered, 1, 2), out)


@partial(jax.jit, static_argnames=("grid", "detector", "scale", "cuda"))
def _fdk_batch(
    views: jax.Array,
    coeff: jax.Array,
    cosine: jax.Array,
    weights: jax.Array,
    kernel: jax.Array,
    out: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    scale: float,
    cuda: bool,
) -> jax.Array:
    filtered = _filter(views, cosine, weights, kernel)
    backproject = _backproject_cuda if cuda else _backproject_jax
    return backproject(filtered, coeff, grid, detector, scale, out)


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
    stacks stream to the device ``views_per_batch`` views at a time.
    """
    cfg = FDKConfig() if config is None else config
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("fdk needs a cone-beam geometry; use fbp for parallel beams")
    if cfg.backend not in {"auto", "jax", "cuda"}:
        raise ValueError("fdk backend must be 'auto', 'jax' or 'cuda'")
    cuda = use_cuda_cone() if cfg.backend == "auto" else cfg.backend == "cuda"
    if cuda and not use_cuda_cone():
        raise ValueError("fdk: the CUDA kernel needs CuPy on a CUDA device")
    validate_grid(grid, "fdk grid")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fdk projections"
    )
    coeff = cone_coefficients(stack_view_poses(geometry, n_views), grid, detector, beam)
    weights = jnp.asarray(view_weights(geometry, detector, n_views), jnp.float32)
    cosine = jnp.asarray(_cosine_weights(beam, detector), jnp.float32)
    du_iso = float(detector.du) / beam.magnification
    kernel = jnp.asarray(get_fbp_filter_np(cfg.filter_name, detector.nu, du_iso, "float32"))
    out = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    batch = max(1, int(cfg.views_per_batch))
    for start in range(0, n_views, batch):
        stop = min(start + batch, n_views)
        part = projections[start:stop]
        views = (
            part.astype(jnp.float32)
            if isinstance(part, jax.Array)
            else jnp.asarray(np.asarray(part, np.float32))
        )
        out = _fdk_batch(
            views, coeff[start:stop], cosine, weights[start:stop], kernel, out,
            grid=grid, detector=detector, scale=1.0 / beam.magnification, cuda=cuda,
        )  # fmt: skip
    return out


__all__ = ["FDKConfig", "fdk", "view_weights"]
