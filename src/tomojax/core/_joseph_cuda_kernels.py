"""JAX-independent CUDA kernels for the exact linear Joseph projector pair."""

from functools import cache
from pathlib import Path
from typing import Any

RUN = 4
_VIEWS_PER_STAGE = 32
THREADS = 128

_GATHER_SOURCE = r"""
// A unit tent on the two plane coordinates maps to a fixed parallelogram
// on the detector. Its center moves affinely along a z run. Translate this
// per-view bounding box to the first voxel instead of inverting every voxel.
__device__ float2 footprint_axis(float ib, float ic, float db, float dc, float eb, float ec) {
    float end = (RUN - 1) * (ib * db + ic * dc);
    float radius = fabsf(ib) + fabsf(ic);
    // Enclose both the rounded forward samples and the rounded inverse center.
    radius += 16 * 1.1920928955078125e-7f * (fabsf(ib)*eb + fabsf(ic)*ec + radius + 1);
    return make_float2(fminf(end, 0.f) - radius, fmaxf(end, 0.f) + radius);
}

template<int A, int R>
__device__ __forceinline__ void gather_view(
    const float* c, const float2* box, const float* image, int ix, int iy, int iz0,
    int nu, int nv, float (&total)[R]) {
    const int a = A;
    float ub = c[1], vb = c[2], kb = c[3], cb = c[4];
    float uc = c[5], vc = c[6], kc = c[7], cc = c[8], weight = c[9];
    float iub = c[10], iuc = c[11], ivb = c[12], ivc = c[13];
    int k0 = a == 0 ? ix : (a == 1 ? iy : iz0);
    int b0 = a == 0 ? iy : (a == 1 ? iz0 : ix);
    int c0 = a == 0 ? iz0 : (a == 1 ? ix : iy);
    // Step of the run's voxel index along the plane coordinates.
    int dk = a == 2, db = a == 1, dc = a == 0;
    float hb[R], hc[R], tb[R], tc[R];
    #pragma unroll
    for (int j = 0; j < R; ++j) {
        float k = (float)(k0 + j * dk);
        hb[j] = __fadd_rn(__fmul_rn(kb, k), cb);
        hc[j] = __fadd_rn(__fmul_rn(kc, k), cc);
        tb[j] = (float)(b0 + j * db);
        tc[j] = (float)(c0 + j * dc);
    }
    float cu = iub * (tb[0] - hb[0]) + iuc * (tc[0] - hc[0]);
    float cv = ivb * (tb[0] - hb[0]) + ivc * (tc[0] - hc[0]);
    int u_start = max((int)floorf(cu + box[0].x) + 1, 0);
    int u_stop = min((int)ceilf(cu + box[0].y), nu);
    int v_start = max((int)floorf(cv + box[1].x) + 1, 0);
    int v_stop = min((int)ceilf(cv + box[1].y), nv);
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
                total[j] += value * (__saturatef(1.f - fabsf(qb - tb[j]))
                                     * __saturatef(1.f - fabsf(qc - tc[j])));
            }
        }
    }
}

__device__ __forceinline__ void gather_run(
    long run, bool valid, const float* __restrict__ coeff, const float* __restrict__ images,
    float* __restrict__ out, int nviews, int nx, int ny, int nz, int nu, int nv, int z0)
{
    // Volume planes z0 to z0 + nz: geometry at global z, storage local.
    const int R = RUN, VB = VIEWS;
    __shared__ float cf[VB * 14];
    __shared__ float2 boxes[VB * 2];
    int ix = 0, iy = 0, iz0 = 0;
    if (valid) FIRST_VOXEL(run, R, ix, iy, iz0);
    float total[R];
    #pragma unroll
    for (int j = 0; j < R; ++j) total[j] = 0.f;
    for (int first = 0; first < nviews; first += VB) {
        __syncthreads();
        for (int i = threadIdx.x; i < VB * 14; i += blockDim.x)
            cf[i] = first * 14 + i < nviews * 14 ? coeff[first * 14 + i] : 0.f;
        __syncthreads();
        int count = min(VB, nviews - first);
        if (threadIdx.x < count) {
            const float* c = cf + threadIdx.x * 14;
            int a = (int)c[0], na = a == 0 ? nx : (a == 1 ? ny : z0 + nz);
            float eb = (a == 0 ? ny : (a == 1 ? z0+nz : nx)) + fabsf(c[4]) + fabsf(c[3])*na
                       + fabsf(c[1])*nu + fabsf(c[2])*nv + R;
            float ec = (a == 0 ? z0+nz : (a == 1 ? nx : ny)) + fabsf(c[8]) + fabsf(c[7])*na
                       + fabsf(c[5])*nu + fabsf(c[6])*nv + R;
            float db = a == 2 ? -c[3] : (float)(a == 1), dc = a == 2 ? -c[7] : (float)(a == 0);
            boxes[2*threadIdx.x]   = footprint_axis(c[10], c[11], db, dc, eb, ec);
            boxes[2*threadIdx.x+1] = footprint_axis(c[12], c[13], db, dc, eb, ec);
        }
        __syncthreads();
        for (int view = 0; view < count; ++view) {
            const float* c = cf + view * 14;
            const float* image = images + (long)(first + view) * nu * nv;
            const float2* box = boxes + 2*view;
            switch ((int)c[0]) {
                case 0: gather_view<0,R>(c, box, image, ix, iy, z0 + iz0, nu, nv, total); break;
                case 1: gather_view<1,R>(c, box, image, ix, iy, z0 + iz0, nu, nv, total); break;
                case 2: gather_view<2,R>(c, box, image, ix, iy, z0 + iz0, nu, nv, total); break;
            }
        }
    }
    if (valid) {
        long base = VOXEL(long, ix, iy, iz0);
        #pragma unroll
        for (int j = 0; j < R; ++j)
            if (iz0 + j < nz) out[base + j * Z_STRIDE] += total[j];
    }
}
// A listed launch covers only the listed runs; whole blocks past the list
// return before any barrier. Unlisted launches never read the list pointers.
extern "C" __global__ void joseph_gather(
    const float* __restrict__ coeff, const float* __restrict__ images,
    float* __restrict__ out, const int* __restrict__ runs, const int* __restrict__ listed_count,
    int listed, int nviews, int nx, int ny, int nz, int nu, int nv, int z0)
{
    long slot = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long count = listed ? *listed_count : (long)nx * ny * ((nz + RUN - 1) / RUN);
    if ((long)blockIdx.x * blockDim.x >= count) return;
    bool valid = slot < count;
    long run = listed && valid ? runs[slot] : slot;
    gather_run(run, valid, coeff, images, out, nviews, nx, ny, nz, nu, nv, z0);
}
"""


def storage_source(*, z_fast: bool) -> str:
    """Kernel source defining the volume storage macros the projector kernels share."""
    header = Path(__file__).with_name("joseph_sampling.cuh").read_text()
    return f"#define Z_FAST {int(z_fast)}\n{header}\n"


@cache
def gather_source(*, z_fast: bool) -> str:
    """Source of the ``joseph_gather`` kernel for one volume storage."""
    source = _GATHER_SOURCE.replace("RUN", str(RUN)).replace("VIEWS", str(_VIEWS_PER_STAGE))
    return storage_source(z_fast=z_fast) + source


@cache
def gather_kernel(*, z_fast: bool = False) -> Any:
    """``joseph_gather``; pass (runs, count, 1) after the output, or (any, any, 0)."""
    import cupy as cp

    return cp.RawModule(code=gather_source(z_fast=z_fast)).get_function("joseph_gather")


@cache
def source(name: str, **constants: int | str) -> str:
    """Specialize a kernel's layout, retaining runtime volume/detector dimensions."""
    root = Path(__file__).parent
    text = (root / "cuda_projector" / (name + ".cu")).read_text()
    if name == "forward":
        constants = {"UNIFORM": 0, "SLAB": 0, **constants}
    for header_name in ("joseph_sampling.cuh", "joseph_gather.cuh", "joseph_tile.cuh"):
        if f'#include "{header_name}"' in text:
            header = root.joinpath(header_name).read_text()
            text = text.replace(f'#include "{header_name}"', header)
    return "\n".join(f"#define {key} {value}" for key, value in constants.items()) + "\n" + text


@cache
def forward_kernel(
    *, narrow: bool, uniform: bool = False, z_fast: bool = False, slab: bool = False
) -> Any:
    import cupy as cp

    text = source(
        "forward",
        INDEX_TYPE="int" if narrow else "long",
        Z_FAST=int(z_fast),
        TILE_V=16 if z_fast else 1,
        BLOCK=THREADS,
        CUBIC=0,
        ABSOLUTE=0,
        UNIFORM=int(uniform),
        SLAB=int(slab),
    )
    return cp.RawModule(code=text, options=("--fmad=false",)).get_function("forward")
