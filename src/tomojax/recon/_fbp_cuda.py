"""Voxel-driven CUDA backprojection of filtered parallel-beam projections."""

from __future__ import annotations

from functools import cache, partial
from typing import Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cuda import launch_kernel
from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin

_SOURCE = r"""
__device__ __forceinline__ float sample(
    const float* image, int u, int v, int nu, int nv) {
    return (unsigned)u < (unsigned)nu && (unsigned)v < (unsigned)nv
        ? image[(long)u * nv + v] : 0.f;
}

extern "C" __global__ void backproject(
    const float* __restrict__ poses, const float* __restrict__ images,
    const float* initial, float* out, int nviews)
{
    __shared__ float cf[32 * 16];
    // Match the reference's rounded reciprocal before the coordinate multiply.
    long count = (long)nx * ny * nz;
    long first = (long)blockIdx.x * blockDim.x * RUN + threadIdx.x;
    float x[RUN], y[RUN], z[RUN], total[RUN], fixed_wv[RUN];
    int fixed_iv[RUN];
    #pragma unroll
    for (int j = 0; j < RUN; ++j) {
        long index = first + j * blockDim.x;
        int iz = index % nz, iy = (index / nz) % ny, ix = index / ((long)ny * nz);
        x[j] = (float)ix * vx + ox;
        y[j] = (float)iy * vy + oy;
        z[j] = (float)iz * vz + oz;
        total[j] = 0.f;
        if (MODE == 0) {
            float v = (z[j] - cv) * (1.f / dv) + (float)((nv - 1) % 2) * 0.5f;
            fixed_iv[j] = (int)floorf(v + 0.5f) + (nv - 1) / 2;
        } else if (MODE == 1) {
            float v = ((poses[10] * z[j] + poses[11]) - cv) / dv
                + (float)((nv - 1) % 2) * 0.5f;
            float lower = floorf(v);
            fixed_wv[j] = v - lower;
            fixed_iv[j] = (int)lower + (nv - 1) / 2;
        }
    }
    for (int start = 0; start < nviews; start += 32) {
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * 16; i += blockDim.x)
            cf[i] = start * 16 + i < nviews * 16 ? poses[start * 16 + i] : 0.f;
        __syncthreads();
        int stop = min(32, nviews - start);
        for (int view = 0; view < stop; ++view) {
            const float* p = cf + view * 16;
            const float* image = images + (long)(start + view) * nu * nv;
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                float wx = (p[0] * x[j] + p[1] * y[j]) + p[3];
                if (MODE == 2) wx += p[2] * z[j];
                float u = (wx - cu) * (1.f / du) + (float)((nu - 1) % 2) * 0.5f;
                float lower = floorf(u), wu = u - lower;
                int iu = (int)lower + (nu - 1) / 2;
                int iv = MODE == 2 ? 0 : fixed_iv[j];
                float wv = MODE == 1 ? fixed_wv[j] : 0.f;
                if (MODE == 2) {
                    float wz = ((p[8] * x[j] + p[9] * y[j]) + p[10] * z[j]) + p[11];
                    float v = (wz - cv) * (1.f / dv) + (float)((nv - 1) % 2) * 0.5f;
                    float lv = floorf(v);
                    wv = v - lv;
                    iv = (int)lv + (nv - 1) / 2;
                }
                float value = sample(image, iu, iv, nu, nv) * (1.f - wu)
                    + sample(image, iu + 1, iv, nu, nv) * wu;
                if (MODE != 0) {
                    float row1 = sample(image, iu, iv + 1, nu, nv) * (1.f - wu)
                        + sample(image, iu + 1, iv + 1, nu, nv) * wu;
                    value = value * (1.f - wv) + row1 * wv;
                }
                total[j] += value;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < RUN; ++j) {
        long index = first + j * blockDim.x;
        if (index < count) out[index] = total[j] + (ACCUMULATE ? initial[index] : 0.f);
    }
}
"""

_VECTOR_SOURCE = r"""
extern "C" __global__ void backproject_vec4(
    const float* __restrict__ poses, const float4* __restrict__ images,
    const float4* initial, float4* out, int nviews)
{
    __shared__ float cf[32 * 16];
    constexpr int R = (RUN+3)/4;
    long count = (long)nx * ny * (nz/4);
    long first = (long)blockIdx.x * blockDim.x * R + threadIdx.x;
    float x[R], y[R], total[R][4] = {};
    int iv[R];
    #pragma unroll
    for (int j = 0; j < R; ++j) {
        long index = first + j * blockDim.x;
        int iz = index % (nz/4), iy = (index / (nz/4)) % ny, ix = index / ((long)ny * (nz/4));
        x[j] = (float)ix * vx + ox;
        y[j] = (float)iy * vy + oy;
        iv[j] = FIRST_ROW/4 + iz;
    }
    for (int start = 0; start < nviews; start += 32) {
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * 16; i += blockDim.x)
            cf[i] = start * 16 + i < nviews * 16 ? poses[start * 16 + i] : 0.f;
        __syncthreads();
        int stop = min(32, nviews-start);
        for (int view = 0; view < stop; ++view) {
            const float* p = cf + view * 16;
            const float4* image = images + (long)(start+view)*nu*(nv/4);
            #pragma unroll
            for (int j = 0; j < R; ++j) {
                float wx = (p[0]*x[j] + p[1]*y[j]) + p[3];
                float u = (wx-cu)*(1.f/du) + (float)((nu-1)%2)*.5f;
                float lower=floorf(u), wu=u-lower;
                int iu=(int)lower+(nu-1)/2, v=iv[j];
                float4 a = {}, b = {};
                if ((unsigned)v < nv/4) {
                    if ((unsigned)iu < nu) a=image[(long)iu*(nv/4)+v];
                    if ((unsigned)(iu+1) < nu) b=image[(long)(iu+1)*(nv/4)+v];
                }
                total[j][0] += a.x*(1.f-wu)+b.x*wu;
                total[j][1] += a.y*(1.f-wu)+b.y*wu;
                total[j][2] += a.z*(1.f-wu)+b.z*wu;
                total[j][3] += a.w*(1.f-wu)+b.w*wu;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < R; ++j) {
        long index=first+j*blockDim.x;
        if (index<count) {
            float4 a = {};
            if (ACCUMULATE) a=initial[index];
            out[index]=make_float4(total[j][0]+a.x,total[j][1]+a.y,
                                   total[j][2]+a.z,total[j][3]+a.w);
        }
    }
}
"""


@cache
def _source(
    grid: Grid, detector: Detector, mode: int, run: int, *, accumulate: bool, vector: bool
) -> str:
    integers = {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz, "nu": detector.nu, "nv": detector.nv}
    ox, oy, oz = grid_volume_origin(grid)
    floats = {
        "ox": ox,
        "oy": oy,
        "oz": oz,
        "vx": grid.vx,
        "vy": grid.vy,
        "vz": grid.vz,
        "du": detector.du,
        "dv": detector.dv,
        "cu": detector.center[0],
        "cv": detector.center[1],
    }
    constants = "".join(f"constexpr int {key} = {value};\n" for key, value in integers.items())
    constants += "".join(
        f"constexpr float {key} = {np.float32(value):.9e}f;\n" for key, value in floats.items()
    )
    first_row = round((oz - detector.center[1]) / detector.dv + (detector.nv - 1) / 2)
    return (
        constants
        + f"#define MODE {mode}\n#define RUN {run}\n#define ACCUMULATE {int(accumulate)}\n"
        + f"#define FIRST_ROW {first_row}\n"
        + (_VECTOR_SOURCE if vector else _SOURCE)
    )


def _launch(context, out, poses, images, *initial: Any, grid, detector, mode, run, vector):
    count = grid.nx * grid.ny * grid.nz
    launch_kernel(
        _source(grid, detector, mode, run, accumulate=bool(initial), vector=vector),
        "backproject_vec4" if vector else "backproject",
        options=("--fmad=false",),
        stream=context.stream,
        grid=(-(-count // (128 * (4 * -(-run // 4) if vector else run))),),
        block=(128,),
        arguments=(poses, images, initial[0] if initial else out, out, np.int32(poses.shape[0])),
    )


def backproject_filtered_cuda(
    poses: jax.Array,
    filtered: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    z_mode: str,
    accumulate: jax.Array | None = None,
    run: int | None = None,
) -> jax.Array:
    """Sum filtered views at voxel centres, adding to an optional accumulator."""
    mode = {"integer": 0, "fixed": 1, "general": 2}[z_mode]
    first = (grid_volume_origin(grid)[2] - detector.center[1]) / detector.dv
    first += (detector.nv - 1) / 2
    vector = (
        mode == 0
        and grid.nz % 4 == 0
        and detector.nv % 4 == 0
        and grid.vz == detector.dv
        and abs(first - round(first)) <= 1e-5
        and round(first) % 4 == 0
    )
    if run is None:
        run = 16 if mode == 0 and grid.nx * grid.ny * grid.nz >= 16 * 1024**2 else 4
    shape = (grid.nx, grid.ny, grid.nz)
    call = buffer_callback(
        partial(_launch, grid=grid, detector=detector, mode=mode, run=run, vector=vector),
        jax.ShapeDtypeStruct(shape, jnp.float32),
        input_output_aliases={2: 0} if accumulate is not None else {},
        vmap_method="sequential",
    )
    operands = [poses, filtered.transpose(0, 2, 1)]
    if accumulate is not None:
        operands.append(accumulate)
    return call(*operands)
