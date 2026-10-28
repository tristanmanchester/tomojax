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

from tomojax.core.cuda import initialize_buffer, launch_kernel

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid

# Elements of the transposes' (views, nu, nz) vertical scratch; longer scans run in view chunks.
_SCRATCH = 32 * 1024**2


def _chunks(views: int, nu: int, nz: int) -> tuple[int, list[tuple[int, int]]]:
    """Scratch views, and the first view and view count of each chunk."""
    size = max(1, min(views, _SCRATCH // (nu * nz)))
    return size, [(first, min(size, views - first)) for first in range(0, views, size)]


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
        long stride=a==0?nz:(long)ny*nz;
        for(int k=start;k<stop;++k) {
            float q=p+(slope*(float)k+offset);
            int h0=(int)floorf(q);float wh=q-(float)h0;
            bool loh=(unsigned)h0<(unsigned)nh,hih=(unsigned)(h0+1)<(unsigned)nh;
            long base=(a==0?(long)k*ny:(long)k)*nz+z0;
            float x00=0.f,x10=0.f;
            if(loh && loz) x00=x[base+(long)h0*stride];
            if(hih && loz) x10=x[base+(long)(h0+1)*stride];
            float value;
            if(wz==0.f) value=x00*(1.f-wh)+x10*wh;
            else {
                float x01=0.f,x11=0.f;
                if(loh && hiz) x01=x[base+(long)h0*stride+1];
                if(hih && hiz) x11=x[base+(long)(h0+1)*stride+1];
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
// Exact specialization for axial rays with unit vertical scale and aligned z.
extern "C" __global__ void axial_forward_vec4(
    const float* __restrict__ coeff,const float* __restrict__ x,float* __restrict__ out,
    int nx,int ny,int nz,int nu,int nv)
{
    int view=blockIdx.x,tile=blockIdx.y,tiles_v=(nv+4*TV-1)/(4*TV);
    int u=(tile/tiles_v)*(128/TV)+threadIdx.x/TV;
    int v=(tile%tiles_v)*(4*TV)+4*(threadIdx.x%TV);
    if(u>=nu || v>=nv) return;
    const float* c=coeff+14*view;
    int a=(int)c[0],na=a==0?nx:ny,nh=a==0?ny:nx;
    float p=(a==0?c[1]:c[5])*(float)u;
    float slope=a==0?c[3]:c[7],offset=a==0?c[4]:c[8];
    int z=v+(int)(a==0?c[8]:c[4]);
    float total[4]={};
    if(z>=0 && z+3<nz) {
        int start=0,stop=na;
        if(slope!=0.f) {
            float l=(-1.f-p-offset)/slope,h=((float)nh-p-offset)/slope;
            start=max(0,(int)floorf(fminf(fmaxf(fminf(l,h),-1.f),(float)na+1.f))-1);
            stop=min(na,(int)ceilf(fminf(fmaxf(fmaxf(l,h),-1.f),(float)na+1.f))+2);
        }
        long stride=a==0?nz:(long)ny*nz;
        for(int k=start;k<stop;++k) {
            float q=p+(slope*(float)k+offset);
            int h0=(int)floorf(q);float wh=q-(float)h0;
            bool loh=(unsigned)h0<(unsigned)nh,hih=(unsigned)(h0+1)<(unsigned)nh;
            long base=(a==0?(long)k*ny:(long)k)*nz+z;
            float4 lo={0.f,0.f,0.f,0.f},hi=lo;
            if(loh) lo=reinterpret_cast<const float4*>(x+base+(long)h0*stride)[0];
            if(hih) hi=reinterpret_cast<const float4*>(x+base+(long)(h0+1)*stride)[0];
            total[0]+=(lo.x*(1.f-wh)+hi.x*wh)*c[9];
            total[1]+=(lo.y*(1.f-wh)+hi.y*wh)*c[9];
            total[2]+=(lo.z*(1.f-wh)+hi.z*wh)*c[9];
            total[3]+=(lo.w*(1.f-wh)+hi.w*wh)*c[9];
        }
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
    float* __restrict__ weighted, int nviews, int nz, int nu, int nv)
{
    long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long)nviews * nu * nz) return;
    int z = id % nz, u = (id / nz) % nu, view = id / ((long)nu * nz);
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
    int ix=live?run/((long)ny*rz):0,iy=live?(run/rz)%ny:0,z=live?R*(run%rz):0;
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
        long base=((long)ix*ny+iy)*nz+z;
        #pragma unroll
        for(int j=0;j<R;++j) if(z+j<nz) out[base+j]+=total[j];
    }
}
"""


def axial_coefficients(coeff: jax.Array) -> jax.Array:
    """Whether all ray planes separate horizontal coordinates from axial z."""
    a = coeff[:, 0]
    x = (a == 0) & (coeff[:, 2] == 0) & (coeff[:, 5] == 0) & (coeff[:, 7] == 0)
    y = (a == 1) & (coeff[:, 1] == 0) & (coeff[:, 6] == 0) & (coeff[:, 3] == 0)
    return jnp.all(x | y)


@cache
def _transpose_module(*, vector: bool, cubic: bool) -> Any:
    import cupy as cp

    source = (
        _TRANSPOSE.replace("RUN", "8" if vector else "4")
        .replace("VECTOR", str(int(vector)))
        .replace("CUBIC", str(int(cubic)))
    )
    return cp.RawModule(code=source)


def _transpose_launch(
    context: Any,
    out: Any,
    coeff: Any,
    images: Any,
    initial: Any,
    *,
    shape: tuple[int, int, int],
    cubic: bool,
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
        module = _transpose_module(vector=vector, cubic=cubic)
        runs = nx * ny * -(-nz // run)
        for first, count in chunks:
            c = cp.asarray(coeff)[first : first + count]
            y = cp.asarray(images)[first : first + count]
            module.get_function("joseph_vertical")(
                (-(-(count * nu * nz) // 256),),
                (256,),
                (c, y, weighted, *map(np.int32, (count, nz, nu, nv))),
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
) -> jax.Array:
    """Add the exact linear or Keys-cubic axial transpose to a flat initial volume."""
    shape = (grid.nx, grid.ny, grid.nz)
    call = buffer_callback(
        partial(_transpose_launch, shape=shape, cubic=interpolation == "cubic"),
        jax.ShapeDtypeStruct(initial.shape, jnp.float32),
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    return call(coeff, images_uv, initial)


@cache
def _forward_source(generic: str, scalar_tile: int, vector_tile: int) -> str:
    parts = [generic]
    for kernel, tile in (
        (_FORWARD, scalar_tile),
        (_FORWARD_VEC4, vector_tile),
    ):
        if not tile:
            continue
        text = kernel.replace("TV", str(tile))
        text = text.replace("int nu, int nv)", "int nu, int nv, int first)")
        text = text.replace("int nu,int nv)", "int nu,int nv,int first)")
        parts.append(text.replace("int view=blockIdx.x,", "int view=blockIdx.x+first,"))
    return "\n".join(parts)


def forward_dispatch_cuda(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    generic_source: str,
) -> jax.Array:
    """Select generic, scalar axial or four-row axial FP without extra compilation."""
    nv, nu = detector.nv, detector.nu
    threads = 32 if nv == 1 else 128
    tile_v = min(16, 1 << (nv.bit_length() - 1))
    scalar_tile = min(32 if grid.nz >= 512 else 16, 1 << (nv.bit_length() - 1))
    vector = grid.nz % 4 == 0 and nv >= 4
    vector_tile = min(8, 1 << ((nv // 4).bit_length() - 1)) if vector else 0
    text = _forward_source(generic_source, scalar_tile, vector_tile)
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
                rows, tile = (4, vector_tile) if mode == 2 else (1, scalar_tile)
                tiles = -(-nu // (128 // tile)) * -(-nv // (rows * tile))
                batch = 16 if mode == 2 else cf.shape[0]
                for first in range(0, cf.shape[0], batch):
                    launch_kernel(
                        text,
                        "axial_forward_vec4" if mode == 2 else "axial_forward",
                        options=("--fmad=false",),
                        stream=context.stream,
                        grid=(min(batch, cf.shape[0] - first), tiles),
                        block=(128,),
                        arguments=(cf, image, out, *map(np.int32, (*dims, first))),
                    )

        return buffer_callback(
            launch, shape, command_buffer_compatible=True, vmap_method="sequential"
        )(c, x)

    def axial(values: tuple[jax.Array, jax.Array]) -> jax.Array:
        c, x = values
        if not vector:
            return call(c, x, 1)
        is_x = c[:, 0] == 0
        scale = jnp.where(is_x, c[:, 6], c[:, 2])
        offset = jnp.where(is_x, c[:, 8], c[:, 4])
        aligned = jnp.all((scale == 1) & (offset == 4 * jnp.floor(offset * 0.25)))
        return jax.lax.cond(
            aligned,
            lambda args: call(args[0], args[1], 2),
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
def _transpose_source(*, vector: bool, cubic: bool) -> str:
    text = _TRANSPOSE.replace("RUN", "8" if vector else "4")
    text = text.replace("VECTOR", str(int(vector))).replace("CUBIC", str(int(cubic)))
    return text.replace("images[((long)view*nu+u)*nv+v]", "images[((long)view*nv+v)*nu+u]")


def transpose_axial_vu_cuda(
    coeff: jax.Array, images: jax.Array, initial: jax.Array, *, grid: Grid, interpolation: str
) -> jax.Array:
    """Add the exact axial transpose using direct VU input and JAX-owned scratch."""
    views, nv, nu = images.shape
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    vector = nz % 8 == 0
    run, threads = (8, 256) if vector else (4, 128)
    text = _transpose_source(vector=vector, cubic=interpolation == "cubic")

    size, chunks = _chunks(views, nu, nz)
    runs = nx * ny * -(-nz // run)

    def launch(context: Any, outputs: Any, c: Any, y: Any, x: Any) -> None:
        out, weighted = outputs
        initialize_buffer(out, stream=context.stream, source=x)
        for first, count in chunks:
            part = _Offset(c, 14 * first)
            launch_kernel(
                text,
                "joseph_vertical",
                stream=context.stream,
                grid=(-(-(count * nu * nz) // 256),),
                block=(256,),
                arguments=(
                    part,
                    _Offset(y, first * nv * nu),
                    weighted,
                    *map(np.int32, (count, nz, nu, nv)),
                ),
            )
            launch_kernel(
                text,
                "joseph_axial",
                stream=context.stream,
                grid=(-(-runs // threads),),
                block=(threads,),
                arguments=(part, weighted, out, *map(np.int32, (count, nx, ny, nz, nu))),
            )

    result, _ = buffer_callback(
        launch,
        (
            jax.ShapeDtypeStruct(initial.shape, jnp.float32),
            jax.ShapeDtypeStruct((size, nu, nz), jnp.float32),
        ),
        input_output_aliases={2: 0},
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coeff, images, initial)
    return result.reshape((nx, ny, nz))
