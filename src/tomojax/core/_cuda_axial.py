"""Exact separable CUDA Joseph operations for axial views.

The vertical transpose is evaluated once per detector column, then shared by
all horizontal voxels. The existing generic pair handles nonseparable views.
"""

from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core._joseph_cuda_kernels import storage_source
from tomojax.core.cuda import initialize_buffer, launch_kernel
from tomojax.core.volume_layout import reverse_axes

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid

# Elements of the transposes' (views, nu, nz) vertical scratch; longer scans run in view chunks.
_SCRATCH = 32 * 1024**2


def _chunks(
    views: int, nu: int, nz: int, batch: int | None = None
) -> tuple[int, list[tuple[int, int]]]:
    """Scratch views, and the first view and view count of each chunk.

    Chunks never cross a multiple of ``batch`` views, so one call adds the same
    chunk sums as one call per batch.
    """
    batch = batch or views
    size = max(1, min(batch, _SCRATCH // (nu * nz)))
    return size, [
        (first + offset, min(size, count - offset))
        for first in range(0, views, batch)
        for count in [min(batch, views - first)]
        for offset in range(0, count, size)
    ]


class _Offset:
    """A float32 callback buffer's storage from element ``elements`` on, without a copy."""

    def __init__(self, buffer: Any, elements: int) -> None:
        address = buffer.__cuda_array_interface__["data"][0] + 4 * elements
        self.__cuda_array_interface__: dict[str, Any] = {"data": (address, False)}


_FORWARD = r"""
extern "C" __global__ void axial_forward(
    const float* __restrict__ coeff, const float* __restrict__ x, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv)
{
    int view=blockIdx.x,tile=blockIdx.y;
    int tiles_v=(nv+TV-1)/TV;
    int u=(tile/tiles_v)*(128/TV)+threadIdx.x/TV,v=(tile%tiles_v)*TV+threadIdx.x%TV;
    if(u>=nu || v>=nv) return;
    const float* c=coeff+14*view;
    int a=(int)c[0],na=a==0?nx:ny,nh=a==0?ny:nx;
    float p=(a==0?c[1]:c[5])*(float)u;
    float slope=a==0?c[3]:c[7],offset=a==0?c[4]:c[8];
    float qz=(a==0?c[6]:c[2])*(float)v+(a==0?c[8]:c[4]);
    int z0=(int)floorf(qz);float wz=qz-(float)z0;
    bool loz=(unsigned)z0<(unsigned)nz,hiz=(unsigned)(z0+1)<(unsigned)nz;
    float total=0.f;
    if(loz || (hiz && wz!=0.f)) {
        int start=0,stop=na;
        if(slope!=0.f) {
            float l=(-1.f-p-offset)/slope,h=((float)nh-p-offset)/slope;
            // Clamp BEFORE integer conversion: near-axis rays have tiny slopes.
            start=max(0,(int)floorf(fminf(fmaxf(fminf(l,h),-1.f),(float)na+1.f))-1);
            stop=min(na,(int)ceilf(fminf(fmaxf(fmaxf(l,h),-1.f),(float)na+1.f))+2);
        }
        // Steps along the plane axis k and the in-plane axis h, in either storage.
        long sk=a==0?VOXEL(long,1,0,0):VOXEL(long,0,1,0),dz=Z_STRIDE;
        long stride=a==0?VOXEL(long,0,1,0):VOXEL(long,1,0,0);
        for(int k=start;k<stop;++k) {
            float q=p+(slope*(float)k+offset);
            int h0=(int)floorf(q);float wh=q-(float)h0;
            bool loh=(unsigned)h0<(unsigned)nh,hih=(unsigned)(h0+1)<(unsigned)nh;
            long base=(long)k*sk+(long)z0*dz;
            float x00=0.f,x10=0.f;
            if(loh && loz) x00=x[base+(long)h0*stride];
            if(hih && loz) x10=x[base+(long)(h0+1)*stride];
            float value;
            if(wz==0.f) value=x00*(1.f-wh)+x10*wh;
            else {
                float x01=0.f,x11=0.f;
                if(loh && hiz) x01=x[base+(long)h0*stride+dz];
                if(hih && hiz) x11=x[base+(long)(h0+1)*stride+dz];
                // Preserve the forward operator's axis-dependent grouping.
                value=a==0?(x00*(1.f-wz)+x01*wz)*(1.f-wh)
                            +(x10*(1.f-wz)+x11*wz)*wh
                           :(x00*(1.f-wh)+x10*wh)*(1.f-wz)
                            +(x01*(1.f-wh)+x11*wh)*wz;
            }
            total+=value*c[9];
        }
    }
    out[((long)view*nv+v)*nu+u]=total;
}
"""

_FORWARD_VEC4 = r"""
// Zero the four-row groups whose planes lie outside the volume, which no band reaches.
extern "C" __global__ void axial_rows_outside(
    const float* __restrict__ coeff,float* __restrict__ out,int nz,int nu,int nv)
{
    int view=blockIdx.x,v=4*blockIdx.y;
    const float* c=coeff+14*view;
    int z=v+(int)(c[0]==0.f?c[8]:c[4]);
    if(z>=0 && z+3<nz) return;
    for(int j=0;j<4 && v+j<nv;++j)
        for(int u=threadIdx.x;u<nu;u+=blockDim.x) out[((long)view*nv+v+j)*nu+u]=0.f;
}

// Exact specialization for axial rays with unit vertical scale and aligned z. Each launch
// reads zs planes from z0 on, z-contiguous with z stride zs: a staged [x][y][32] band, or a
// whole z-fast volume (z0 0, zs nz). Eight lanes of float4 then cover one 128-byte line.
extern "C" __global__ void axial_forward_vec4(
    const float* __restrict__ coeff,const float* __restrict__ x,float* __restrict__ out,
    int nx,int ny,int nz,int nu,int nv,int first,int z0,int zs)
{
    int view=blockIdx.x+first,tile=blockIdx.y,tiles_v=(zs+4*TV-1)/(4*TV);
    int u=(tile/tiles_v)*(128/TV)+threadIdx.x/TV;
    int z=(tile%tiles_v)*(4*TV)+4*(threadIdx.x%TV);
    const float* c=coeff+14*view;
    int a=(int)c[0],na=a==0?nx:ny,nh=a==0?ny:nx;
    int v=z0+z-(int)(a==0?c[8]:c[4]);
    if(u>=nu || z>=zs || z0+z+3>=nz || v<0 || v>=nv) return;
    float p=(a==0?c[1]:c[5])*(float)u;
    float slope=a==0?c[3]:c[7],offset=a==0?c[4]:c[8];
    float total[4]={};
    int start=0,stop=na;
    if(slope!=0.f) {
        float l=(-1.f-p-offset)/slope,h=((float)nh-p-offset)/slope;
        start=max(0,(int)floorf(fminf(fmaxf(fminf(l,h),-1.f),(float)na+1.f))-1);
        stop=min(na,(int)ceilf(fminf(fmaxf(fmaxf(l,h),-1.f),(float)na+1.f))+2);
    }
    long stride=a==0?zs:(long)ny*zs;
    for(int k=start;k<stop;++k) {
        float q=p+(slope*(float)k+offset);
        int h0=(int)floorf(q);float wh=q-(float)h0;
        bool loh=(unsigned)h0<(unsigned)nh,hih=(unsigned)(h0+1)<(unsigned)nh;
        long base=(a==0?(long)k*ny:(long)k)*zs+z;
        float4 lo={0.f,0.f,0.f,0.f},hi=lo;
        if(loh) lo=reinterpret_cast<const float4*>(x+base+(long)h0*stride)[0];
        if(hih) hi=reinterpret_cast<const float4*>(x+base+(long)(h0+1)*stride)[0];
        total[0]+=(lo.x*(1.f-wh)+hi.x*wh)*c[9];
        total[1]+=(lo.y*(1.f-wh)+hi.y*wh)*c[9];
        total[2]+=(lo.z*(1.f-wh)+hi.z*wh)*c[9];
        total[3]+=(lo.w*(1.f-wh)+hi.w*wh)*c[9];
    }
    #pragma unroll
    for(int j=0;j<4;++j) if(v+j<nv) out[((long)view*nv+v+j)*nu+u]=total[j];
}
"""

_TRANSPOSE = r"""
__device__ __forceinline__ float axial_weight(float distance)
{
    float r = fabsf(distance);
#if CUBIC
    if(r<1.f) return 1.f+r*r*(1.5f*r-2.5f);
    // Factored roots preserve the small negative lobe near the support edge.
    if(r<2.f) return -.5f*(r-1.f)*(r-2.f)*(r-2.f);
    return 0.f;
#else
    return fmaxf(1.f-r,0.f);
#endif
}

__device__ __forceinline__ int2 axial_support(
    float inverse, float coordinate, float offset, int samples)
{
    float radius = (CUBIC ? 2.f : 1.f) * fabsf(inverse);
    float center = inverse * (coordinate - offset);
    // Inverting rounded forward coordinates can otherwise exclude a positive
    // edge weight. Bound cancellation using the operands, not just center.
    // Only candidate indices widen: the physical weights below stay unchanged.
    float magnitude = radius * (fabsf(coordinate) + fabsf(offset)) + radius + 1.f;
    float margin = 8.f * 1.1920928955078125e-7f * magnitude;
    int first = max((int)floorf(center - radius - margin) + 1, 0);
    int stop = min((int)ceilf(center + radius + margin), samples);
    return make_int2(first, stop);
}

extern "C" __global__ void joseph_vertical(
    const float* __restrict__ coeff, const float* __restrict__ images,
    float* __restrict__ weighted, int nviews, int nz, int nu, int nv, int z0)
{
    // weighted[view][u][z] for planes z0 + z of nz from z0 on.
    long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long)nviews * nu * nz) return;
    int z = z0 + id % nz, u = (id / nz) % nu, view = id / ((long)nu * nz);
    const float* c = coeff + view * 14;
    int a = (int)c[0];
    float vz = a == 0 ? c[6] : c[2], cz = a == 0 ? c[8] : c[4];
    float ivz = a == 0 ? c[13] : c[12];
    int2 support = axial_support(ivz, (float)z, cz, nv);
    float sum = 0.f;
    for (int v=support.x;v<support.y;++v) {
        float qz = __fadd_rn(__fmul_rn(vz,(float)v),cz);
        float wz = axial_weight(qz-(float)z);
        sum += images[((long)view*nu+u)*nv+v] * c[9] * wz;
    }
    weighted[id] = sum;
}

extern "C" __global__ void joseph_axial(
    const float* __restrict__ coeff, const float* __restrict__ weighted,
    float* __restrict__ out, int nviews, int nx, int ny, int nz, int nu)
{
    const int R = RUN, VB = 32;
    long run=(long)blockIdx.x*blockDim.x+threadIdx.x;
    int rz=(nz+R-1)/R;
    bool live=run<(long)nx*ny*rz;
    int ix=0,iy=0,z=0;
    if(live) FIRST_VOXEL(run,R,ix,iy,z);
    float total[R]={};
    __shared__ float cf[VB*14];
    for(int first=0;first<nviews;first+=VB) {
        __syncthreads();
        for(int i=threadIdx.x;i<VB*14;i+=blockDim.x)
            cf[i]=first*14+i<nviews*14?coeff[first*14+i]:0.f;
        __syncthreads();
        for(int w=0;w<min(VB,nviews-first);++w) {
            const float* c=cf+w*14;int a=(int)c[0];
            float k=(float)(a==0?ix:iy),row=(float)(a==0?iy:ix);
            float kh=a==0?c[3]:c[7],ch=a==0?c[4]:c[8],uh=a==0?c[1]:c[5];
            float inv=a==0?c[10]:c[11];
            float h=__fadd_rn(__fmul_rn(kh,k),ch);
            int2 support=axial_support(inv,row,h,nu);
            const float* image=weighted+(long)(first+w)*nu*nz;
            for(int u=support.x;u<support.y;++u) {
                float q=__fadd_rn(__fmul_rn(uh,(float)u),h);
                float wh=axial_weight(q-row);
#if VECTOR
                #pragma unroll
                for(int j=0;j<R;j+=4) {
                    float4 value=reinterpret_cast<const float4*>(image+(long)u*nz+z+j)[0];
                    total[j]+=value.x*wh; total[j+1]+=value.y*wh;
                    total[j+2]+=value.z*wh; total[j+3]+=value.w*wh;
                }
#else
                #pragma unroll
                for(int j=0;j<R;++j) if(z+j<nz) total[j]+=image[(long)u*nz+z+j]*wh;
#endif
            }
        }
    }
    if(live) {
        long base=VOXEL(long,ix,iy,z);
        #pragma unroll
        for(int j=0;j<R;++j) if(z+j<nz) out[base+j*Z_STRIDE]+=total[j];
    }
}
"""


def axial_coefficients(coeff: jax.Array) -> jax.Array:
    """Whether all ray planes separate horizontal coordinates from axial z."""
    a = coeff[:, 0]
    x = (a == 0) & (coeff[:, 2] == 0) & (coeff[:, 5] == 0) & (coeff[:, 7] == 0)
    y = (a == 1) & (coeff[:, 1] == 0) & (coeff[:, 6] == 0) & (coeff[:, 3] == 0)
    return jnp.all(x | y)


def aligned_axial(coeff: jax.Array) -> jax.Array:
    """Whether all views are axial with unit vertical scale and rows aligned to four planes."""
    is_x = coeff[:, 0] == 0
    scale = jnp.where(is_x, coeff[:, 6], coeff[:, 2])
    offset = jnp.where(is_x, coeff[:, 8], coeff[:, 4])
    rows = jnp.all((scale == 1) & (offset == 4 * jnp.floor(offset * 0.25)))
    return axial_coefficients(coeff) & rows


@cache
def _transpose_module(*, vector: bool, cubic: bool, z_fast: bool) -> Any:
    import cupy as cp

    return cp.RawModule(code=_transpose_source(vector=vector, cubic=cubic, z_fast=z_fast, vu=False))


def _transpose_launch(
    context: Any,
    out: Any,
    coeff: Any,
    images: Any,
    initial: Any,
    *,
    shape: tuple[int, int, int],
    cubic: bool,
    z_fast: bool,
) -> None:
    import cupy as cp

    from tomojax.core._cuda_joseph import xla_stream

    nx, ny, nz = shape
    views, nu, nv = images.shape
    with xla_stream(context, out):
        target, previous = cp.asarray(out), cp.asarray(initial)
        if target.data.ptr != previous.data.ptr:
            target[...] = previous
        size, chunks = _chunks(views, nu, nz)
        weighted = cp.empty((size, nu, nz), cp.float32)
        # Aligned float4 loads amortize global-load instructions. Odd axial
        # extents retain guarded scalar reads, including the last partial run.
        vector = nz % 8 == 0
        run, threads = (8, 256) if vector else (4, 128)
        module = _transpose_module(vector=vector, cubic=cubic, z_fast=z_fast)
        runs = nx * ny * -(-nz // run)
        for first, count in chunks:
            c = cp.asarray(coeff)[first : first + count]
            y = cp.asarray(images)[first : first + count]
            module.get_function("joseph_vertical")(
                (-(-(count * nu * nz) // 256),),
                (256,),
                (c, y, weighted, *map(np.int32, (count, nz, nu, nv, 0))),
            )
            module.get_function("joseph_axial")(
                (-(-runs // threads),),
                (threads,),
                (c, weighted, target, *map(np.int32, (count, nx, ny, nz, nu))),
            )


def transpose_axial_cuda(
    coeff: jax.Array,
    images_uv: jax.Array,
    initial: jax.Array,
    *,
    grid: Grid,
    interpolation: str = "linear",
    z_fast: bool = False,
) -> jax.Array:
    """Add the exact linear or Keys-cubic axial transpose to a flat initial volume.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    """
    shape = (grid.nx, grid.ny, grid.nz)
    cubic = interpolation == "cubic"
    call = buffer_callback(
        partial(_transpose_launch, shape=shape, cubic=cubic, z_fast=z_fast),
        jax.ShapeDtypeStruct(initial.shape, jnp.float32),
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    return call(coeff, images_uv, initial)


def _banded_forward(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    text: str,
    tile: int,
    *,
    z_fast: bool,
) -> jax.Array:
    """Four-row axial FP, 32 planes at a time from a (z, y, x) volume.

    The volume is staged one z-contiguous band at a time, so every ray sums the
    same values in the same order as from z-fast storage, with one band of
    workspace. A z-fast volume is read in place in one pass.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    nu, nv = detector.nu, detector.nv
    depth = nz if z_fast else 32
    tiles = -(-nu // (128 // tile)) * -(-depth // (4 * tile))
    # Views 16 a launch share their reads in L2; narrow scans take more, so
    # that each launch has about a wave of blocks.
    chunk = max(16, -(-512 // tiles))
    staged = () if z_fast else (jax.ShapeDtypeStruct((nx, ny, 32), jnp.float32),)

    def launch(context: Any, outputs: Any, cf: Any, x: Any) -> None:
        out, *band = outputs if staged else (outputs,)
        views, stream, options = cf.shape[0], context.stream, ("--fmad=false",)
        launch_kernel(
            text,
            "axial_rows_outside",
            options=options,
            stream=stream,
            grid=(views, -(-nv // 4)),
            block=(128,),
            arguments=(cf, out, *map(np.int32, (nz, nu, nv))),
        )
        for z0 in range(0, nz, depth):
            if band:
                reverse_axes(x, band[0], (nz, ny, nx), stream=stream, z0=z0, depth=32)
            for first in range(0, views, chunk):
                launch_kernel(
                    text,
                    "axial_forward_vec4",
                    options=options,
                    stream=stream,
                    grid=(min(chunk, views - first), tiles),
                    block=(128,),
                    arguments=(
                        cf,
                        band[0] if band else x,
                        out,
                        *map(np.int32, (nx, ny, nz, nu, nv, first, z0, depth)),
                    ),
                )

    shape = jax.ShapeDtypeStruct((coeff.shape[0], nv, nu), jnp.float32)
    result = buffer_callback(
        launch,
        (shape, *staged) if staged else shape,
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff, volume)
    return result[0] if staged else result


def forward_dispatch_cuda(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    generic_source: str,
    tile_v: int,
    z_fast: bool = False,
) -> jax.Array:
    """Select generic, scalar axial or four-row axial FP without extra compilation.

    ``generic_source`` defines the volume storage (Z_FAST) for all three kernels.
    """
    nv, nu = detector.nv, detector.nu
    threads = 32 if nv == 1 else 128
    scalar_tile = min(32 if grid.nz >= 512 else 16, 1 << (nv.bit_length() - 1)) if z_fast else 1
    vector = grid.nz % 4 == 0 and nv >= 4
    vector_tile = min(8, 1 << ((nv // 4).bit_length() - 1)) if vector else 0
    kernels = [generic_source, _FORWARD.replace("TV", str(scalar_tile))]
    if vector:
        kernels.append(_FORWARD_VEC4.replace("TV", str(vector_tile)))
    text = "\n".join(kernels)
    shape = jax.ShapeDtypeStruct((coeff.shape[0], nv, nu), jnp.float32)

    def call(c: jax.Array, x: jax.Array, mode: int) -> jax.Array:
        def launch(context: Any, out: Any, cf: Any, image: Any) -> None:
            dims = (grid.nx, grid.ny, grid.nz, nu, nv)
            if mode == 0:
                tiles = -(-nu // (threads // tile_v)) * -(-nv // tile_v)
                launch_kernel(
                    text,
                    "forward",
                    options=("--fmad=false",),
                    stream=context.stream,
                    grid=(tiles * cf.shape[0],),
                    block=(threads,),
                    arguments=(cf, image, out, *map(np.int32, (*dims, cf.shape[0]))),
                )
            else:
                launch_kernel(
                    text,
                    "axial_forward",
                    options=("--fmad=false",),
                    stream=context.stream,
                    grid=(cf.shape[0], -(-nu // (128 // scalar_tile)) * -(-nv // scalar_tile)),
                    block=(128,),
                    arguments=(cf, image, out, *map(np.int32, dims)),
                )

        return buffer_callback(
            launch, shape, command_buffer_compatible=True, vmap_method="sequential"
        )(c, x)

    def axial(values: tuple[jax.Array, jax.Array]) -> jax.Array:
        c, x = values
        if not vector:
            return call(c, x, 1)
        return jax.lax.cond(
            aligned_axial(c),
            lambda args: _banded_forward(
                args[0], args[1], grid, detector, text, vector_tile, z_fast=z_fast
            ),
            lambda args: call(args[0], args[1], 1),
            (c, x),
        )

    return jax.lax.cond(
        axial_coefficients(coeff),
        axial,
        lambda args: call(args[0], args[1], 0),
        (coeff, volume),
    )


@cache
def _transpose_source(*, vector: bool, cubic: bool, z_fast: bool, vu: bool) -> str:
    text = _TRANSPOSE.replace("RUN", "8" if vector else "4")
    text = text.replace("VECTOR", str(int(vector))).replace("CUBIC", str(int(cubic)))
    text = storage_source(z_fast=z_fast) + text
    if not vu:
        return text
    return text.replace("images[((long)view*nu+u)*nv+v]", "images[((long)view*nv+v)*nu+u]")


def transpose_axial_vu_cuda(
    coeff: jax.Array,
    images: jax.Array,
    initial: jax.Array | None,
    *,
    grid: Grid,
    interpolation: str,
    z_fast: bool = False,
    batch: int | None = None,
) -> jax.Array:
    """Add the exact axial transpose using direct VU input and JAX-owned scratch.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    A (z, y, x) volume is added to 32 planes at a time in a z-fast band, so
    each voxel adds the same chunk sums in the same order in either storage.
    ``batch`` views at a time make the chunks of one call per batch. With no
    ``initial`` the output is new, and each band starts from zeros.
    """
    views, nv, nu = images.shape
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    depth = nz if z_fast else min(nz, 32)
    size, chunks = _chunks(views, nu, nz, batch)

    def add(stream: int, c: Any, y: Any, weighted: Any, target: Any, z0: int, planes: int) -> None:
        # Aligned float4 loads amortize global-load instructions. Odd axial
        # extents retain guarded scalar reads, including the last partial run.
        vector = planes % 8 == 0
        run, threads = (8, 256) if vector else (4, 128)
        text = _transpose_source(
            vector=vector, cubic=interpolation == "cubic", z_fast=True, vu=True
        )
        runs = nx * ny * -(-planes // run)
        for first, count in chunks:
            part = _Offset(c, 14 * first)
            launch_kernel(
                text,
                "joseph_vertical",
                stream=stream,
                grid=(-(-(count * nu * planes) // 256),),
                block=(256,),
                arguments=(
                    part,
                    _Offset(y, first * nv * nu),
                    weighted,
                    *map(np.int32, (count, planes, nu, nv, z0)),
                ),
            )
            launch_kernel(
                text,
                "joseph_axial",
                stream=stream,
                grid=(-(-runs // threads),),
                block=(threads,),
                arguments=(part, weighted, target, *map(np.int32, (count, nx, ny, planes, nu))),
            )

    def launch(context: Any, outputs: Any, c: Any, y: Any, *x: Any) -> None:
        out, weighted, *band = outputs
        stream = context.stream
        if x or not band:
            initialize_buffer(out, stream=stream, source=x[0] if x else None)
        if not band:
            add(stream, c, y, weighted, out, 0, nz)
            return
        for z0 in range(0, nz, depth):
            planes = min(depth, nz - z0)
            if x:
                reverse_axes(out, band[0], (nz, ny, nx), stream=stream, z0=z0, depth=planes)
            else:
                initialize_buffer(band[0], stream=stream)
            add(stream, c, y, weighted, band[0], z0, planes)
            reverse_axes(
                band[0], _Offset(out, z0 * ny * nx), (nx, ny, planes), stream=stream, depth=nx
            )

    staged = () if z_fast else (jax.ShapeDtypeStruct((nx, ny, depth), jnp.float32),)
    shape = (nx, ny, nz) if z_fast else (nz, ny, nx)
    result, *_ = buffer_callback(
        launch,
        (
            jax.ShapeDtypeStruct(shape if initial is None else initial.shape, jnp.float32),
            jax.ShapeDtypeStruct((size, nu, depth), jnp.float32),
            *staged,
        ),
        input_output_aliases={} if initial is None else {2: 0},
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff, images, *(() if initial is None else (initial,)))
    return result.reshape(shape)
