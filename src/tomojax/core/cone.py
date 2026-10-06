"""Cone-beam Joseph projection: rays from a point source sampled on voxel planes.

Each ray is sampled once per voxel-centre plane along the view's dominant axis
and interpolated bilinearly within the plane, as in the parallel Joseph model;
here the in-plane coordinates follow the ray from the source to its pixel, so
magnification changes from plane to plane. Line integrals are in physical
units.

Each ray is sampled along its own dominant axis, so a projection changes
continuously as a pose turns a ray through 45 degrees. Per-view geometry is a
25-value coefficient row in voxel-index units of the object frame (source,
detector origin and pixel steps, a separable-view flag, and the detector-plane
projection used to bound voxel footprints). Rows come from
world_from_object poses and a :class:`~tomojax.core.geometry.cone.ConeBeam`,
in JAX, so they are differentiable with respect to the poses.

Two implementations agree to FP32 rounding:

- JAX: one code path for every view through flat-index gathers; the forward
  supports autodiff in the volume and the poses, and :func:`cone_backproject`
  is its explicit transpose (a plane-by-plane scatter, never a stored tape).
- CUDA (CuPy, launched on XLA's stream with ``buffer_callback``): per-ray
  forward and voxel-run gather transpose, plus two-pass separable kernels for
  views whose detector v axis is parallel to the volume's z axis (unperturbed
  turntable scans), where the bilinear weight factors into a column and a z
  part. Forward and transpose use identical coordinate arithmetic.
"""

from __future__ import annotations

from functools import cache, partial
import os
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import grid_volume_origin

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid
    from tomojax.core.geometry.cone import ConeBeam

NC = 25
_HI = jax.lax.Precision.HIGHEST
_VIEW_CHUNK = 32  # adjoint views per launch, so their images stay in L2


def cone_coefficients(
    poses: jax.Array, grid: Grid, detector: Detector, beam: ConeBeam
) -> jax.Array:
    """Return ``(views, 25)`` FP32 per-view coefficients for the cone kernels."""
    poses = jnp.asarray(poses, jnp.float32)
    rot, trans = poses[:, :3, :3], poses[:, :3, 3]
    origin = jnp.asarray(grid_volume_origin(grid), jnp.float32)
    spacing = jnp.asarray([grid.vx, grid.vy, grid.vz], jnp.float32)
    centre, u_dir, v_dir = (jnp.asarray(x, jnp.float32) for x in beam.detector_frame(detector))
    source = jnp.asarray(beam.source(), jnp.float32)
    corner = (
        centre
        - (detector.nu - 1) / 2 * float(detector.du) * u_dir
        - (detector.nv - 1) / 2 * float(detector.dv) * v_dir
    )

    def to_object(point: jax.Array) -> jax.Array:  # lab point -> object index coordinates
        local = jnp.einsum("nji,nj->ni", rot, point[None] - trans, precision=_HI)
        return (local - origin) / spacing

    def direction(vector: jax.Array) -> jax.Array:
        return jnp.einsum("nji,j->ni", rot, vector, precision=_HI) / spacing

    S = to_object(source)
    D0 = to_object(corner)
    DU = direction(float(detector.du) * u_dir)
    DV = direction(float(detector.dv) * v_dir)
    # Views of an unperturbed turntable (detector v parallel to z, no ray steeper
    # than 45 degrees in z) let the CUDA kernels separate columns from z.
    u = jnp.arange(detector.nu, dtype=jnp.float32)
    row0 = D0[:, None, :] + u[None, :, None] * DU[:, None, :] - S[:, None, :]
    rows = jnp.stack([row0, row0 + (detector.nv - 1) * DV[:, None, :]], axis=1)
    lateral = jnp.min(jnp.maximum(jnp.abs(rows[..., 0]), jnp.abs(rows[..., 1])), axis=(1, 2))
    steep = jnp.max(jnp.abs(rows[..., 2]), axis=(1, 2))
    flag = (DV[:, 0] == 0) & (DV[:, 1] == 0) & (DV[:, 2] > 0) & (steep < lateral)
    axis = jax.lax.stop_gradient(flag.astype(jnp.float32))
    n = jnp.cross(DU, DV)
    h = jnp.sum(n * (D0 - S), axis=1)
    dual_u = jnp.cross(DV, n)
    Au = dual_u / jnp.sum(DU * dual_u, axis=1, keepdims=True)
    dual_v = jnp.cross(n, DU)
    Av = dual_v / jnp.sum(DV * dual_v, axis=1, keepdims=True)
    cu = jnp.sum((S - D0) * Au, axis=1)
    cv = jnp.sum((S - D0) * Av, axis=1)
    return jnp.concatenate(
        [S, D0, DU, DV, axis[:, None], n, h[:, None], Au, Av, cu[:, None], cv[:, None]], axis=1
    )


# ----------------------------------------------------------------------------- JAX


def _ray_frame(ray: jax.Array, grid: Grid) -> dict[str, jax.Array]:
    """Per-ray plane axis a (largest index-space component) and in-plane axes b, c."""
    a = jnp.argmax(jnp.abs(ray), axis=-1).astype(jnp.int32)
    b = jnp.where(a == 0, 1, 0)
    c = jnp.where(a == 2, 1, 2)
    sizes = jnp.asarray([grid.nx, grid.ny, grid.nz], jnp.int32)
    strides = jnp.asarray([grid.ny * grid.nz, grid.nz, 1], jnp.int32)
    return {
        "a": a, "b": b, "c": c,
        "na": sizes[a], "nb": sizes[b], "nc": sizes[c],
        "sa": strides[a], "sb": strides[b], "sc": strides[c],
    }  # fmt: skip


def _rays(row: jax.Array, detector: Detector) -> jax.Array:
    """``(nv, nu, 3)`` ray vectors (pixel minus source) in index units."""
    u = jnp.arange(detector.nu, dtype=jnp.float32)
    v = jnp.arange(detector.nv, dtype=jnp.float32)
    pixel = row[3:6] + u[None, :, None] * row[6:9] + v[:, None, None] * row[9:12]
    return pixel - row[0:3]


def _component(x: jax.Array, axis: jax.Array) -> jax.Array:
    return jnp.take_along_axis(x, axis[..., None], axis=-1)[..., 0]


def _path_weight(ray: jax.Array, frame: dict[str, jax.Array], grid: Grid) -> jax.Array:
    spacing = jnp.asarray([grid.vx, grid.vy, grid.vz], jnp.float32)
    ra = _component(ray, frame["a"])
    return jnp.linalg.norm(ray * spacing, axis=-1) / jnp.maximum(jnp.abs(ra), 1e-12)


def _plane_samples(
    row: jax.Array, frame: dict[str, jax.Array], ray: jax.Array, k: jax.Array
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Flat indices and weights of the four bilinear taps of every ray on its plane k."""
    S = jnp.broadcast_to(row[0:3], ray.shape)
    ra = _component(ray, frame["a"])
    ra = jnp.where(jnp.abs(ra) < 1e-12, 1e-12, ra)
    t = (k - _component(S, frame["a"])) / ra
    fb = _component(S, frame["b"]) + t * _component(ray, frame["b"])
    fc = _component(S, frame["c"]) + t * _component(ray, frame["c"])
    fl_b, fl_c = jnp.floor(fb), jnp.floor(fc)
    indices, weights = [], []
    for db in (0.0, 1.0):
        for dc in (0.0, 1.0):
            jb, jc = fl_b + db, fl_c + dc
            w = jnp.maximum(1 - jnp.abs(fb - jb), 0) * jnp.maximum(1 - jnp.abs(fc - jc), 0)
            jb_i, jc_i = jb.astype(jnp.int32), jc.astype(jnp.int32)
            valid = (
                (jb_i >= 0) & (jb_i < frame["nb"]) & (jc_i >= 0) & (jc_i < frame["nc"])
                & (k < frame["na"])
            )  # fmt: skip
            index = k.astype(jnp.int32) * frame["sa"] + jb_i * frame["sb"] + jc_i * frame["sc"]
            indices.append(jnp.where(valid, index, 0))
            weights.append(jnp.where(valid, w, 0.0))
    return indices, weights


def _forward_view_jax(
    volume: jax.Array, row: jax.Array, grid: Grid, detector: Detector
) -> jax.Array:
    ray = _rays(row, detector)
    frame = _ray_frame(jax.lax.stop_gradient(ray), grid)
    flat = volume.reshape(-1)
    planes = max(grid.nx, grid.ny, grid.nz)

    def step(total: jax.Array, k: jax.Array) -> tuple[jax.Array, None]:
        indices, weights = _plane_samples(row, frame, ray, k)
        for index, weight in zip(indices, weights, strict=True):
            total = total + flat[index] * weight
        return total, None

    total, _ = jax.lax.scan(
        step,
        jnp.zeros(ray.shape[:-1], jnp.float32),
        jnp.arange(planes, dtype=jnp.float32),
    )
    return total * _path_weight(ray, frame, grid)


def _forward_jax(volume: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector) -> jax.Array:
    return jax.vmap(lambda row: _forward_view_jax(volume, row, grid, detector))(coeff)


def _adjoint_jax(
    images: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector, accumulate: jax.Array
) -> jax.Array:
    """Explicit transpose of :func:`_forward_jax`: scatter each plane's taps."""
    rays = jax.vmap(lambda row: _rays(row, detector))(coeff)
    frames = _ray_frame(rays, grid)
    values = images * _path_weight(rays, frames, grid)
    flat = accumulate.reshape(-1)

    def step(total: jax.Array, k: jax.Array) -> tuple[jax.Array, None]:
        indices, weights = jax.vmap(lambda row, frame, ray: _plane_samples(row, frame, ray, k))(
            coeff, frames, rays
        )
        for index, weight in zip(indices, weights, strict=True):
            total = total.at[index.reshape(-1)].add((values * weight).reshape(-1))
        return total, None

    planes = max(grid.nx, grid.ny, grid.nz)
    total, _ = jax.lax.scan(step, flat, jnp.arange(planes, dtype=jnp.float32))
    return total.reshape(grid.nx, grid.ny, grid.nz)


# ----------------------------------------------------------------------------- CUDA

_SOURCE = r"""
#define NC 25
#define TU 8
#define TV 128
#define QLEN 224
#define TB 64
#define TC 32
#define ULEN 160
#define RUN 8
#define SEL(i, x0, x1, x2) ((i) == 0 ? (x0) : ((i) == 1 ? (x1) : (x2)))
__device__ __forceinline__ float trif(float f, float j) { return fmaxf(1.f - fabsf(f - j), 0.f); }
__device__ __forceinline__ bool separable(const float* c) { return c[12] > 0.5f; }
// The ray's plane axis: its largest index-space component, ties to the lower axis.
__device__ __forceinline__ int ray_axis(float r0, float r1, float r2) {
    int a = 0; float m = fabsf(r0);
    if (fabsf(r1) > m) { a = 1; m = fabsf(r1); }
    if (fabsf(r2) > m) a = 2;
    return a;
}
__device__ __forceinline__ void ray_of(const float* c, float fu, float fv,
                                       float& r0, float& r1, float& r2) {
    r0 = (c[3] + fu * c[6] + fv * c[9]) - c[0];
    r1 = (c[4] + fu * c[7] + fv * c[10]) - c[1];
    r2 = (c[5] + fu * c[8] + fv * c[11]) - c[2];
}
__device__ __forceinline__ float path_w(float r0, float r1, float r2, float ra,
                                        float sx, float sy, float sz) {
    float px = r0 * sx, py = r1 * sy, pz = r2 * sz;
    return sqrtf(px * px + py * py + pz * pz) / fabsf(ra);
}

// Per-ray Joseph sum for one pixel, any view.
__device__ float ray_sum(const float* __restrict__ vol, const float* c, float fu, float fv,
                         int nx, int ny, int nz)
{
    float r0, r1, r2; ray_of(c, fu, fv, r0, r1, r2);
    int a = ray_axis(r0, r1, r2), b = a == 0 ? 1 : 0, cc = a == 2 ? 1 : 2;
    int na = SEL(a, nx, ny, nz), nb = SEL(b, nx, ny, nz), nc = SEL(cc, nx, ny, nz);
    long s0 = (long)ny * nz, s1 = nz;
    long sa = SEL(a, s0, s1, 1L), sb = SEL(b, s0, s1, 1L), sc = SEL(cc, s0, s1, 1L);
    float ra = SEL(a, r0, r1, r2), rb = SEL(b, r0, r1, r2), rc = SEL(cc, r0, r1, r2);
    float Sa = c[a], Sb = c[b], Sc = c[cc];
    float inv = 1.0f / ra;
    float lo = -1e30f, hi = 1e30f;
    float slope = rb * inv, base = Sb - Sa * slope;
    if (fabsf(slope) < 1e-12f) { if (base <= -1.f || base >= (float)nb) return 0.f; }
    else { float k1 = (-1.f - base) / slope, k2 = ((float)nb - base) / slope;
           lo = fmaxf(lo, fminf(k1, k2)); hi = fminf(hi, fmaxf(k1, k2)); }
    slope = rc * inv; base = Sc - Sa * slope;
    if (fabsf(slope) < 1e-12f) { if (base <= -1.f || base >= (float)nc) return 0.f; }
    else { float k1 = (-1.f - base) / slope, k2 = ((float)nc - base) / slope;
           lo = fmaxf(lo, fminf(k1, k2)); hi = fminf(hi, fmaxf(k1, k2)); }
    int k0 = max((int)floorf(lo), 0), k1 = min((int)ceilf(hi), na - 1);
    float sum = 0.f;
    for (int k = k0; k <= k1; ++k) {
        float t = ((float)k - Sa) * inv;
        float fb = Sb + t * rb, fc = Sc + t * rc;
        float fb0 = floorf(fb), fc0 = floorf(fc);
        int b0 = (int)fb0, c0 = (int)fc0;
        float wb0 = trif(fb, fb0), wb1 = trif(fb, fb0 + 1.f);
        float wc0 = trif(fc, fc0), wc1 = trif(fc, fc0 + 1.f);
        bool b0in = b0 >= 0 && b0 < nb, b1in = b0 + 1 >= 0 && b0 + 1 < nb;
        bool c0in = c0 >= 0 && c0 < nc, c1in = c0 + 1 >= 0 && c0 + 1 < nc;
        const float* p = vol + (long)k * sa + (long)b0 * sb + (long)c0 * sc;
        if (b0in && c0in) sum += __ldg(p) * (wb0 * wc0);
        if (b0in && c1in) sum += __ldg(p + sc) * (wb0 * wc1);
        if (b1in && c0in) sum += __ldg(p + sb) * (wb1 * wc0);
        if (b1in && c1in) sum += __ldg(p + sb + sc) * (wb1 * wc1);
    }
    return sum;
}

// grid (u tiles, v tiles, views); out (view, u, v). Separable views share one plane
// axis per detector column (a in {x, y}), so a column's rays share their in-plane row.
extern "C" __global__ void cone_forward(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz)
{
    __shared__ float Q[TU][QLEN];
    __shared__ float s_w0[TU], s_w1[TU];
    __shared__ int s_b0[TU], clo[TU], clen[TU];
    int view = blockIdx.z;
    const float* c = coeff + (long)view * NC;
    int u0 = blockIdx.x * TU, v0 = blockIdx.y * TV, tid = threadIdx.x, v = v0 + tid;
    float fv = (float)v;
    float acc[TU];
    #pragma unroll
    for (int i = 0; i < TU; ++i) acc[i] = 0.f;
    if (!separable(c)) {
        if (v < nv) {
            #pragma unroll
            for (int i = 0; i < TU; ++i)
                if (u0 + i < nu) acc[i] = ray_sum(vol, c, (float)(u0 + i), fv, nx, ny, nz);
        }
    } else {
        float Sc = c[2], DVc = c[11];
        int ax[TU];
        float inv[TU], rcv[TU], Sa_[TU];
        int ax_t = 0;
        float rb_t = 0.f, inv_t = 0.f, rc0_t = 0.f, Sa_t = 0.f, Sb_t = 0.f;
        #pragma unroll
        for (int i = 0; i < TU; ++i) {
            float fu = (float)min(u0 + i, nu - 1);
            float r0 = (c[3] + fu * c[6]) - c[0], r1 = (c[4] + fu * c[7]) - c[1];
            int a = fabsf(r1) > fabsf(r0) ? 1 : 0;
            ax[i] = a;
            float ra = a == 0 ? r0 : r1;
            inv[i] = 1.0f / ra;
            Sa_[i] = c[a];
            rcv[i] = ((c[5] + fu * c[8]) + fv * DVc) - Sc;
            if (i == tid) {
                ax_t = a; inv_t = inv[i]; Sa_t = c[a]; Sb_t = c[1 - a];
                rb_t = a == 0 ? r1 : r0;
                rc0_t = c[5] + fu * c[8];
            }
        }
        int vlast = min(v0 + TV, nv) - 1;
        long sxl = (long)ny * nz;
        int kmax = max(nx, ny);
        for (int k = 0; k < kmax; ++k) {
            __syncthreads();
            if (tid < TU) {
                int na = ax_t == 0 ? nx : ny;
                float t = ((float)k - Sa_t) * inv_t;
                float fb = Sb_t + t * rb_t, fl = floorf(fb);
                s_b0[tid] = (int)fl; s_w0[tid] = trif(fb, fl); s_w1[tid] = trif(fb, fl + 1.f);
                float f1 = Sc + t * ((rc0_t + (float)v0 * DVc) - Sc);
                float f2 = Sc + t * ((rc0_t + (float)vlast * DVc) - Sc);
                int lo = (int)floorf(fminf(f1, f2));
                clo[tid] = lo;
                clen[tid] = k < na ? (int)floorf(fmaxf(f1, f2)) + 2 - lo : 0;
            }
            __syncthreads();
            {
                int i = tid / (TV / TU), j0 = tid % (TV / TU);
                int a = ax[i], nb = a == 0 ? ny : nx;
                long sa = a == 0 ? sxl : (long)nz, sb = a == 0 ? (long)nz : sxl;
                const float* plane = vol + (long)k * sa;
                int len = min(clen[i], QLEN), lo = clo[i], b0 = s_b0[i];
                float w0 = (b0 >= 0 && b0 < nb) ? s_w0[i] : 0.f;
                float w1 = (b0 + 1 >= 0 && b0 + 1 < nb) ? s_w1[i] : 0.f;
                const float* p0 = plane + (long)min(max(b0, 0), nb - 1) * sb;
                const float* p1 = plane + (long)min(max(b0 + 1, 0), nb - 1) * sb;
                for (int j = j0; j < len; j += TV / TU) {
                    int jc = lo + j;
                    bool in = jc >= 0 && jc < nz;
                    Q[i][j] = in ? __ldg(p0 + jc) * w0 + __ldg(p1 + jc) * w1 : 0.f;
                }
            }
            __syncthreads();
            if (v < nv) {
                #pragma unroll
                for (int i = 0; i < TU; ++i) {
                    if (clen[i] == 0) continue;
                    float t = ((float)k - Sa_[i]) * inv[i];
                    float fc = Sc + t * rcv[i], fl = floorf(fc);
                    int j = (int)fl - clo[i];
                    float w0 = trif(fc, fl), w1 = trif(fc, fl + 1.f);
                    if (clen[i] <= QLEN) {
                        if (j >= 0 && j + 1 < clen[i]) acc[i] += Q[i][j] * w0 + Q[i][j + 1] * w1;
                    } else {
                        // Column range too long for Q: read the two rows directly.
                        int a = ax[i], nb = a == 0 ? ny : nx;
                        long sa = a == 0 ? sxl : (long)nz, sb = a == 0 ? (long)nz : sxl;
                        const float* plane = vol + (long)k * sa;
                        int b0 = s_b0[i], c0 = (int)fl;
                        float wb0 = (b0 >= 0 && b0 < nb) ? s_w0[i] : 0.f;
                        float wb1 = (b0 + 1 >= 0 && b0 + 1 < nb) ? s_w1[i] : 0.f;
                        const float* p0 = plane + (long)min(max(b0, 0), nb - 1) * sb;
                        const float* p1 = plane + (long)min(max(b0 + 1, 0), nb - 1) * sb;
                        if (c0 >= 0 && c0 < nz)
                            acc[i] += (__ldg(p0 + c0) * wb0 + __ldg(p1 + c0) * wb1) * w0;
                        if (c0 + 1 >= 0 && c0 + 1 < nz)
                            acc[i] += (__ldg(p0 + c0 + 1) * wb0 + __ldg(p1 + c0 + 1) * wb1) * w1;
                    }
                }
            }
        }
    }
    if (v < nv) {
        #pragma unroll
        for (int i = 0; i < TU; ++i) {
            int u = u0 + i;
            if (u < nu) {
                float r0, r1, r2; ray_of(c, (float)u, fv, r0, r1, r2);
                float ra = SEL(ray_axis(r0, r1, r2), r0, r1, r2);
                out[((long)view * nu + u) * nv + v] = acc[i] * path_w(r0, r1, r2, ra, sx, sy, sz);
            }
        }
    }
}

extern "C" __global__ void weight_images(
    const float* __restrict__ coeff, const float* __restrict__ img, float* __restrict__ out,
    int nviews, int nu, int nv, float sx, float sy, float sz)
{
    long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long)nviews * nu * nv) return;
    int v = id % nv; int u = (id / nv) % nu; int view = id / ((long)nu * nv);
    const float* c = coeff + (long)view * NC;
    float r0, r1, r2; ray_of(c, (float)u, (float)v, r0, r1, r2);
    float ra = SEL(ray_axis(r0, r1, r2), r0, r1, r2);
    out[id] = img[id] * path_w(r0, r1, r2, ra, sx, sy, sz);
}

// Separable views, the columns whose plane axis is a: grid (z tiles, row tiles, planes).
// img is path-weighted.
extern "C" __global__ void sep_adjoint(
    const float* __restrict__ coeff, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol,
    int nx, int ny, int nz, int nu, int nv)
{
    __shared__ float R[ULEN][TC + 1];
    __shared__ float s_fb[ULEN], s_t[ULEN], s_rc0[ULEN], s_f0[ULEN], s_idf[ULEN];
    __shared__ int s_taps[ULEN];
    __shared__ float cf[NC];
    __shared__ int skip;
    int b = 1 - a;
    int nb = a == 0 ? ny : nx;
    int c0 = blockIdx.x * TC, b0 = blockIdx.y * TB, k = blockIdx.z;
    int tid = threadIdx.x;
    int my_b = b0 + tid / 4, my_c = (tid % 4) * (TC / 4);
    float my_bf = (float)my_b;
    float acc[TC / 4];
    #pragma unroll
    for (int j = 0; j < TC / 4; ++j) acc[j] = 0.f;
    for (int w = 0; w < nviews; ++w) {
        __syncthreads();
        if (tid < NC) cf[tid] = coeff[(long)w * NC + tid];
        __syncthreads();
        if (tid == 0) skip = !separable(cf);
        __syncthreads();
        if (skip) continue;
        const float* c = cf;
        float Sa = c[a], Sb = c[b], Sc = c[2], DVc = c[11];
        float K = (float)k - Sa;
        float ra0 = c[3 + a] - c[a], rb0 = c[3 + b] - c[b], DUa = c[6 + a], DUb = c[6 + b];
        float g1 = (float)b0 - 1.f - Sb, g2 = (float)(b0 + TB) - Sb;
        float u1 = (K * rb0 - g1 * ra0) / (g1 * DUa - K * DUb);
        float u2 = (K * rb0 - g2 * ra0) / (g2 * DUa - K * DUb);
        int ulo_all = max((int)floorf(fminf(u1, u2)) - 1, 0);
        int uhi = min((int)ceilf(fmaxf(u1, u2)) + 1, nu - 1);
        const float* image = img + (long)w * nu * nv;
        for (int ulo = ulo_all; ulo <= uhi; ulo += ULEN) {
            int ulen = min(ULEN, uhi - ulo + 1);
            __syncthreads();
            for (int i = tid; i < ulen; i += blockDim.x) {
                float fu = (float)(ulo + i);
                float r0 = (c[3] + fu * c[6]) - c[0], r1 = (c[4] + fu * c[7]) - c[1];
                int col_axis = fabsf(r1) > fabsf(r0) ? 1 : 0;
                float ra = a == 0 ? r0 : r1, rb = a == 0 ? r1 : r0;
                float t = K * (1.0f / ra);
                float rc0 = c[5] + fu * c[8];
                // Columns sampled along the other axis contribute nothing here.
                s_t[i] = t; s_fb[i] = col_axis == a ? Sb + t * rb : -1e30f; s_rc0[i] = rc0;
                // z cells rise with v at 1/idf per row: each cell needs at most taps rows.
                float idf = 1.f / (t * DVc);
                s_f0[i] = Sc + t * (rc0 - Sc); s_idf[i] = idf;
                s_taps[i] = col_axis == a ? (int)floorf(2.f * fabsf(idf)) + 2 : 0;
            }
            __syncthreads();
            for (int i = tid / TC, j = tid % TC; i < ulen; i += blockDim.x / TC) {
                float jcf = (float)(c0 + j);
                float t = s_t[i], rc0 = s_rc0[i];
                int va = (int)floorf((jcf - 1.f - s_f0[i]) * s_idf[i]);
                int taps = s_taps[i];   // the same for the whole warp: no divergence
                const float* col = image + (long)(ulo + i) * nv;
                float s = 0.f;
                #pragma unroll 4
                for (int q = 0; q < taps; ++q) {
                    int vv = va + q;
                    if (vv < 0 || vv >= nv) continue;
                    s += __ldg(col + vv) * trif(Sc + t * ((rc0 + (float)vv * DVc) - Sc), jcf);
                }
                R[i][j] = s;
            }
            __syncthreads();
            if (my_b < nb) {
                float h1 = my_bf - 1.f - Sb, h2 = my_bf + 1.f - Sb;
                float w1 = (K * rb0 - h1 * ra0) / (h1 * DUa - K * DUb);
                float w2 = (K * rb0 - h2 * ra0) / (h2 * DUa - K * DUb);
                int ia = max((int)floorf(fminf(w1, w2)) - 1 - ulo, 0);
                int ib = min((int)ceilf(fmaxf(w1, w2)) + 1 - ulo, ulen - 1);
                for (int i = ia; i <= ib; ++i) {
                    float wb = trif(s_fb[i], my_bf);
                    if (wb == 0.f) continue;
                    #pragma unroll
                    for (int j = 0; j < TC / 4; ++j) acc[j] += wb * R[i][my_c + j];
                }
            }
        }
    }
    int na = a == 0 ? nx : ny;
    if (my_b < nb && k < na) {
        int ix = a == 0 ? k : my_b, iy = a == 0 ? my_b : k;
        long base = ((long)ix * ny + iy) * nz + c0 + my_c;
        #pragma unroll
        for (int j = 0; j < TC / 4; ++j) if (c0 + my_c + j < nz) vol[base + j] += acc[j];
    }
}

// Non-separable views: one thread per RUN z voxels, rays sampled on their own axis.
// img is path-weighted.
extern "C" __global__ void general_adjoint(
    const float* __restrict__ coeff, int nviews, const float* __restrict__ img,
    float* __restrict__ vol, int nx, int ny, int nz, int nu, int nv)
{
    __shared__ float cf[32 * NC];
    long run = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int runs_z = (nz + RUN - 1) / RUN;
    bool valid = run < (long)nx * ny * runs_z;
    int ix = valid ? run / ((long)ny * runs_z) : 0;
    int iy = valid ? (run / runs_z) % ny : 0;
    int iz0 = valid ? RUN * (int)(run % runs_z) : 0;
    float fx = (float)ix, fy = (float)iy;
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
            if (separable(c)) continue;
            // Footprint of every ray that can reach these voxels, on any plane axis:
            // the box spanning the voxels +-1 in x, y and z.
            float lx = fx - 1.f, hx = fx + 1.f, ly = fy - 1.f, hy = fy + 1.f;
            float lz = iz0 - 1.f, hz = iz0 + RUN;
            float umin = 1e30f, umax = -1e30f, vmin = 1e30f, vmax = -1e30f;
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                float d0 = (q & 1 ? hx : lx) - c[0], d1 = (q & 2 ? hy : ly) - c[1];
                float d2 = (q & 4 ? hz : lz) - c[2];
                float lam = c[16] / (c[13] * d0 + c[14] * d1 + c[15] * d2);
                float uu = c[23] + lam * (c[17] * d0 + c[18] * d1 + c[19] * d2);
                float vv = c[24] + lam * (c[20] * d0 + c[21] * d1 + c[22] * d2);
                umin = fminf(umin, uu); umax = fmaxf(umax, uu);
                vmin = fminf(vmin, vv); vmax = fmaxf(vmax, vv);
            }
            int u0 = max((int)floorf(umin), 0), u1 = min((int)ceilf(umax), nu - 1);
            int v0 = max((int)floorf(vmin), 0), v1 = min((int)ceilf(vmax), nv - 1);
            const float* image = img + (long)(first + w) * nu * nv;
            for (int u = u0; u <= u1; ++u) {
                for (int v = v0; v <= v1; ++v) {
                    float r0, r1, r2; ray_of(c, (float)u, (float)v, r0, r1, r2);
                    int a = ray_axis(r0, r1, r2);
                    float value = image[(long)u * nv + v];
                    if (a == 0) {
                        float t = (fx - c[0]) * (1.0f / r0);
                        float wb = trif(c[1] + t * r1, fy);
                        if (wb == 0.f) continue;
                        float fc = c[2] + t * r2;
                        #pragma unroll
                        for (int j = 0; j < RUN; ++j)
                            acc[j] += value * (wb * trif(fc, (float)(iz0 + j)));
                    } else if (a == 1) {
                        float t = (fy - c[1]) * (1.0f / r1);
                        float wb = trif(c[0] + t * r0, fx);
                        if (wb == 0.f) continue;
                        float fc = c[2] + t * r2;
                        #pragma unroll
                        for (int j = 0; j < RUN; ++j)
                            acc[j] += value * (wb * trif(fc, (float)(iz0 + j)));
                    } else {
                        float inv = 1.0f / r2;
                        #pragma unroll
                        for (int j = 0; j < RUN; ++j) {
                            float t = ((float)(iz0 + j) - c[2]) * inv;
                            float wb = trif(c[0] + t * r0, fx);
                            acc[j] += value * (wb * trif(c[1] + t * r1, fy));
                        }
                    }
                }
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


def use_cuda_cone() -> bool:
    """Return whether cone projections run as CUDA kernels (``TOMOJAX_CUDA_KERNELS=0`` disables)."""
    if os.environ.get("TOMOJAX_CUDA_KERNELS", "auto") == "0":
        return False
    from tomojax.core._cuda_joseph import cuda_gather_available

    return cuda_gather_available()


@cache
def _module() -> Any:
    import cupy as cp

    return cp.RawModule(code=_SOURCE)


def xla_stream(context: Any) -> Any:
    """Return XLA's CUDA stream for a ``buffer_callback`` context, as a CuPy stream."""
    import cupy as cp

    from tomojax.core._cuda_joseph import _XlaStream

    return cp.cuda.Stream.from_external(_XlaStream(int(context.stream)))


def _launch_forward(
    context: Any, out: Any, coeff: Any, volume: Any, *, grid: Grid, det: Detector
) -> None:
    import cupy as cp

    views = int(coeff.shape[0])
    with xla_stream(context):
        _module().get_function("cone_forward")(
            (-(-det.nu // 8), -(-det.nv // 128), views),
            (128,),
            (
                cp.asarray(coeff), cp.asarray(volume), cp.asarray(out),
                np.int32(grid.nx), np.int32(grid.ny), np.int32(grid.nz),
                np.int32(det.nu), np.int32(det.nv),
                np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz),
            ),
        )  # fmt: skip


def _launch_adjoint(
    context: Any, out: Any, coeff: Any, images: Any, accumulate: Any, *, grid: Grid, det: Detector
) -> None:
    import cupy as cp

    views = int(coeff.shape[0])
    module = _module()
    with xla_stream(context):
        target, initial = cp.asarray(out), cp.asarray(accumulate)
        if target.data.ptr != initial.data.ptr:
            target[...] = initial
        coeff, images = cp.asarray(coeff), cp.asarray(images)
        weighted = cp.empty_like(images)
        total = views * det.nu * det.nv
        module.get_function("weight_images")(
            (-(-total // 256),),
            (256,),
            (
                coeff, images, weighted, np.int32(views), np.int32(det.nu), np.int32(det.nv),
                np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz),
            ),
        )  # fmt: skip
        sizes = (grid.nx, grid.ny, grid.nz, det.nu, det.nv)
        dims = tuple(np.int32(x) for x in sizes)
        for start in range(0, views, _VIEW_CHUNK):
            count = min(_VIEW_CHUNK, views - start)
            c_part, i_part = coeff[start : start + count], weighted[start : start + count]
            for axis, (na, nb) in ((0, (grid.nx, grid.ny)), (1, (grid.ny, grid.nx))):
                module.get_function("sep_adjoint")(
                    (-(-grid.nz // 32), -(-nb // 64), na),
                    (256,),
                    (c_part, np.int32(count), np.int32(axis), i_part, target, *dims),
                )
            runs = grid.nx * grid.ny * -(-grid.nz // 8)
            module.get_function("general_adjoint")(
                (-(-runs // 128),), (128,), (c_part, np.int32(count), i_part, target, *dims)
            )


def _forward_cuda(volume: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector) -> jax.Array:
    call = buffer_callback(
        partial(_launch_forward, grid=grid, det=detector),
        jax.ShapeDtypeStruct((coeff.shape[0], detector.nu, detector.nv), jnp.float32),
        vmap_method="sequential",
    )
    return jnp.swapaxes(call(coeff, volume.astype(jnp.float32)), 1, 2)


def _adjoint_cuda(
    images: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector, accumulate: jax.Array
) -> jax.Array:
    call = buffer_callback(
        partial(_launch_adjoint, grid=grid, det=detector),
        jax.ShapeDtypeStruct((grid.nx, grid.ny, grid.nz), jnp.float32),
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    images_uv = jnp.swapaxes(images.astype(jnp.float32), 1, 2)
    return call(coeff, images_uv, accumulate.astype(jnp.float32))


# ----------------------------------------------------------------------------- public


def _resolve(backend: str) -> str:
    if backend not in {"auto", "jax", "cuda"}:
        raise ValueError("cone backend must be 'auto', 'jax' or 'cuda'")
    if backend == "auto":
        return "cuda" if use_cuda_cone() else "jax"
    if backend == "cuda" and not use_cuda_cone():
        raise ValueError("the CUDA cone kernels need CuPy on a CUDA device")
    return backend


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _forward_cuda_vjp(
    volume: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector
) -> jax.Array:
    return _forward_cuda(volume, coeff, grid, detector)


def _forward_cuda_fwd(
    volume: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector
) -> tuple[jax.Array, jax.Array]:
    return _forward_cuda(volume, coeff, grid, detector), coeff


def _forward_cuda_bwd(
    grid: Grid, detector: Detector, coeff: jax.Array, cotangent: jax.Array
) -> tuple[jax.Array, jax.Array]:
    zeros = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    return _adjoint_cuda(cotangent, coeff, grid, detector, zeros), jnp.zeros_like(coeff)


_forward_cuda_vjp.defvjp(_forward_cuda_fwd, _forward_cuda_bwd)


def cone_project(
    volume: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str = "auto",
) -> jax.Array:
    """Project a volume through every view's coefficients; returns ``(views, nv, nu)``.

    The JAX backend differentiates in the volume and the coefficients (hence
    the poses). The CUDA backend differentiates in the volume only, through
    its matched transpose.
    """
    if _resolve(backend) == "cuda":
        return _forward_cuda_vjp(jnp.asarray(volume, jnp.float32), coeff, grid, detector)
    return _forward_jax(jnp.asarray(volume, jnp.float32), coeff, grid, detector)


def cone_backproject(
    images: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str = "auto",
    accumulate: jax.Array | None = None,
) -> jax.Array:
    """Return ``accumulate`` plus the exact transpose of :func:`cone_project`."""
    initial = (
        jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
        if accumulate is None
        else jnp.asarray(accumulate, jnp.float32)
    )
    images = jnp.asarray(images, jnp.float32)
    if _resolve(backend) == "cuda":
        return _adjoint_cuda(images, coeff, grid, detector, initial)
    return _adjoint_jax(images, coeff, grid, detector, initial)


__all__ = [
    "cone_backproject",
    "cone_coefficients",
    "cone_project",
    "use_cuda_cone",
    "xla_stream",
]
