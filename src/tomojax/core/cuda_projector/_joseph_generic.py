"""Exact CUDA plane sampling and voxel gathers for arbitrary parallel-ray poses."""

from __future__ import annotations

from functools import cache
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cuda import broadcast_offset, launch_kernel

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


@cache
def source(name: str, **constants: int | str) -> str:
    """Specialize a kernel's layout, retaining runtime volume/detector dimensions."""
    text = Path(__file__).with_name(name + ".cu").read_text()
    for header_name in ("joseph_sampling.cuh", "joseph_gather.cuh"):
        if f'#include "{header_name}"' in text:
            header = Path(__file__).parents[1].joinpath(header_name).read_text()
            text = text.replace(f'#include "{header_name}"', header)
    return "\n".join(f"#define {key} {value}" for key, value in constants.items()) + "\n" + text


def forward(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str,
    absolute_weights: bool,
) -> jax.Array:
    """Sum bilinear or cubic plane samples in the reference plane order."""
    threads = 32 if detector.nv == 1 else 128
    tile_v = 1
    tile_u = threads // tile_v
    tiles = ((detector.nu + tile_u - 1) // tile_u) * ((detector.nv + tile_v - 1) // tile_v)
    views = coeff.shape[0]

    def launch(context: Any, out: Any, cf: Any, x: Any) -> None:
        batch_shape = tuple(out.shape[:-3])
        batches = math.prod(batch_shape)
        if not batches:
            return
        text = source(
            "forward",
            INDEX_TYPE="int" if grid.nx * grid.ny * grid.nz < 2**31 else "long",
            TILE_V=tile_v,
            BLOCK=threads,
            CUBIC=int(interpolation == "cubic"),
            ABSOLUTE=int(absolute_weights),
            COEFF_OFFSET=broadcast_offset(
                batch_shape, tuple(cf.shape[:-2]), item_size=math.prod(cf.shape[-2:])
            ),
            VOLUME_OFFSET=broadcast_offset(
                batch_shape, tuple(x.shape[:-3]), item_size=math.prod(x.shape[-3:])
            ),
        )
        launch_kernel(
            text,
            "forward",
            options=("--fmad=false",),
            stream=context.stream,
            grid=(batches * views * tiles,),
            block=(threads,),
            arguments=(
                cf,
                x,
                out,
                *map(np.int32, (grid.nx, grid.ny, grid.nz, detector.nu, detector.nv, views)),
            ),
        )

    return buffer_callback(
        launch,
        jax.ShapeDtypeStruct((views, detector.nv, detector.nu), jnp.float32),
        command_buffer_compatible=True,
        vmap_method="expand_dims",
    )(coeff, volume)


def adjoint(
    coeff: jax.Array,
    images: jax.Array,
    initial: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str,
    absolute_weights: bool,
) -> jax.Array:
    """Gather the exact plane weights at each voxel and add an initial volume."""
    shape = (grid.nx, grid.ny, grid.nz)
    run = min(4, grid.nz)
    text = source(
        "adjoint_unit",
        RUN=run,
        CUBIC=int(interpolation == "cubic"),
        ABSOLUTE=int(absolute_weights),
    )
    runs = grid.nx * grid.ny * ((grid.nz + run - 1) // run)
    views = coeff.shape[0]

    def launch(context: Any, out: Any, cf: Any, y: Any, x: Any) -> None:
        launch_kernel(
            text,
            "joseph_gather",
            options=("--fmad=false",),
            stream=context.stream,
            grid=((runs + 127) // 128,),
            block=(128,),
            arguments=(cf, y, out, x, *map(np.int32, (views, *shape, detector.nu, detector.nv))),
        )

    return buffer_callback(
        launch,
        jax.ShapeDtypeStruct(shape[::-1], jnp.float32),
        input_output_aliases={2: 0},
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff, images, initial)


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


def swap_detector_axes(images: jax.Array) -> jax.Array:
    """Transpose through an opaque call, keeping loop-carried sinograms in VU order."""
    views, nv, nu = images.shape
    if nv == 1 or nu == 1:
        return images.reshape(views, nu, nv)

    def launch(context: Any, out: Any, data: Any) -> None:
        launch_kernel(
            _SWAP,
            "swap_axes",
            stream=context.stream,
            grid=((nu + 31) // 32, (nv + 31) // 32, views),
            block=(32, 8),
            arguments=(data, out, np.int32(nu), np.int32(nv)),
        )

    return buffer_callback(
        launch,
        jax.ShapeDtypeStruct((views, nu, nv), jnp.float32),
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(images)
