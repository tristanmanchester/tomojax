"""Bounded CUDA volume permutation at resident-solver boundaries and in projector bands."""

from __future__ import annotations

from typing import Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cuda import launch_kernel

_REVERSE = r"""
// Planes z0..z0+depth-1 of a C-order (nz, ny, nx) array, axes reversed into (nx, ny, depth);
// planes beyond nz are zero. A shared tile keeps reads and writes coalesced.
extern "C" __global__ void reverse_axes(
    const float* __restrict__ src,float* __restrict__ dst,int nx,int ny,int nz,int z0,int depth)
{
    __shared__ float tile[32][33];
    int x=blockIdx.x*32+threadIdx.x, z=blockIdx.y*32+threadIdx.y, y=blockIdx.z;
    #pragma unroll
    for(int j=0;j<32;j+=8) if(x<nx && z+j<depth)
        tile[threadIdx.y+j][threadIdx.x]=z0+z+j<nz?src[((long)(z0+z+j)*ny+y)*nx+x]:0.f;
    __syncthreads();
    x=blockIdx.x*32+threadIdx.y; z=blockIdx.y*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8) if(x+j<nx && z<depth)
        dst[((long)(x+j)*ny+y)*depth+z]=tile[threadIdx.x][threadIdx.y+j];
}
"""


_SWAP = r"""
extern "C" __global__ void swap_axes(const float* input,float* output,int nu,int nv) {
 __shared__ float tile[32][33];
 int u=blockIdx.x*32+threadIdx.x,v=blockIdx.y*32+threadIdx.y;
 long base=(long)blockIdx.z*nu*nv;
 #pragma unroll
 for(int j=0;j<32;j+=8)
  if(u<nu && v+j<nv) tile[threadIdx.y+j][threadIdx.x]=input[base+(long)(v+j)*nu+u];
 __syncthreads();
 v=blockIdx.y*32+threadIdx.x;u=blockIdx.x*32+threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8)
  if(u+j<nu && v<nv) output[base+(long)(u+j)*nv+v]=tile[threadIdx.x][threadIdx.y+j];
}
"""


def swap_last_axes(source: Any, target: Any, shape: tuple[int, int, int], *, stream: int) -> None:
    """Launch the transpose of each (rows, columns) matrix of a C-order (n, rows, columns) array."""
    n, rows, columns = shape
    launch_kernel(
        _SWAP,
        "swap_axes",
        stream=stream,
        grid=(-(-columns // 32), -(-rows // 32), n),
        block=(32, 8),
        arguments=(source, target, np.int32(columns), np.int32(rows)),
    )


def reverse_axes(
    source: Any, target: Any, shape: tuple[int, int, int], *, stream: int, z0: int = 0, depth: int
) -> None:
    """Launch the permutation of ``depth`` planes from ``z0`` of a C-order ``shape`` array."""
    nz, ny, nx = shape
    launch_kernel(
        _REVERSE,
        "reverse_axes",
        stream=stream,
        grid=(-(-nx // 32), -(-depth // 32), ny),
        block=(32, 8),
        arguments=(source, target, *map(np.int32, (nx, ny, nz, z0, depth))),
    )


def swap_private_axes(volume: jax.Array, *, z_fast: bool) -> jax.Array:
    """Exchange public and private JAX orders, or retain the public-only order."""
    return jnp.transpose(volume, (2, 1, 0)) if z_fast else volume


@jax.custom_jvp
@jax.jit
def transpose_volume(volume: jax.Array) -> jax.Array:
    """Reverse the volume axes using one output buffer and a bounded shared tile.

    This explicit permutation avoids compiler autotuning with volume-sized
    trial buffers at the boundary of a memory-limited solve. It is compiled
    once per shape, and the launch keeps no reference to its input.
    """
    volume = jnp.asarray(volume, jnp.float32)
    nz, ny, nx = volume.shape

    def launch(context: Any, out: Any, x: Any) -> None:
        reverse_axes(x, out, (nz, ny, nx), stream=context.stream, depth=nz)

    call = buffer_callback(
        launch,
        jax.ShapeDtypeStruct(volume.shape[::-1], jnp.float32),
        vmap_method="sequential",
    )
    # Off a GPU the same permutation is an ordinary transpose.
    return jax.lax.platform_dependent(
        volume, cuda=call, default=lambda v: jnp.transpose(v, (2, 1, 0))
    )


@transpose_volume.defjvp
def _transpose_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    # Keep the linear tangent in JAX, including its exact reverse-mode rule.
    return transpose_volume(primals[0]), jnp.transpose(
        jnp.asarray(tangents[0], jnp.float32), (2, 1, 0)
    )
