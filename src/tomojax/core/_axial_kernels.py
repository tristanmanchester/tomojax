"""Separable Joseph CUDA kernels shared by the native and JAX projectors."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from ._joseph_cuda_kernels import raw_kernel, storage_source
from .cuda_types import CuPyValue

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray


# Elements of the transposes' (views, nu, nz) vertical scratch; longer scans run in view chunks.
_SCRATCH = 32 * 1024**2


def view_chunks(
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


FORWARD = r"""
// x holds planes zb..zb+zs-1 of the storage (all nz of them, or a z-fast band). A row is
// owned by one staged band when adjacent bands overlap by a halo plane. Disjoint host slabs
// can both contribute to a row, each reading only the interpolation taps it owns.
extern "C" __global__ void axial_forward(
    const float* __restrict__ coeff, const float* __restrict__ x, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, int zb, int zs, int zhi, int first_row_plane)
{
    int view=blockIdx.x,tile=blockIdx.y+gridDim.y*blockIdx.z;
    int tiles_v=(nv+TV-1)/TV;
    int u=(tile/tiles_v)*(128/TV)+threadIdx.x/TV,v=(tile%tiles_v)*TV+threadIdx.x%TV;
    if(u>=nu || v>=nv) return;
    const float* c=coeff+14*view;
    int a=(int)c[0],na=a==0?nx:ny,nh=a==0?ny:nx;
    float p=(a==0?c[1]:c[5])*(float)u;
    float slope=a==0?c[3]:c[7],offset=a==0?c[4]:c[8];
    float qz=(a==0?c[6]:c[2])*(float)v+(a==0?c[8]:c[4]);
    int z0=(int)floorf(qz);float wz=qz-(float)z0;
    bool loz=(unsigned)(z0-zb)<(unsigned)min(zs,nz-zb);
    bool hiz=(unsigned)(z0+1-zb)<(unsigned)min(zs,nz-zb);
    if((zb>0 && z0<first_row_plane) || (zhi<nz && z0>=zhi)) return;
    x-=zb;
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
        if(zs!=nz) { sk=a==0?(long)ny*zs:zs; stride=a==0?zs:(long)ny*zs; }  // a z-fast band
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

FORWARD_VEC4 = r"""
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
    int view=blockIdx.x+first,tile=blockIdx.y+gridDim.y*blockIdx.z,tiles_v=(zs+4*TV-1)/(4*TV);
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
    float* __restrict__ out, int nviews, int nx, int ny, int nz, int nu, int fresh, int out_depth)
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
        long base=Z_FAST ? ((long)ix*ny+iy)*out_depth+z : VOXEL(long,ix,iy,z);
        // Keep the same +0 addition as a zero-initialized output.
        #pragma unroll
        for(int j=0;j<R;++j) if(z+j<nz)
            out[base+j*Z_STRIDE]=(fresh ? 0.f : out[base+j*Z_STRIDE])+total[j];
    }
}
"""


def tile_grid(views: int, tiles: int) -> tuple[int, int, int]:
    """Spread large detectors over grid.z once grid.y reaches its CUDA limit."""
    width = min(tiles, 65535)
    return views, width, -(-tiles // width)


def axial_forward(
    cp: CuPyValue,
    coeff: CuPyValue,
    volume: CuPyValue,
    *,
    dims: tuple[int, int, int, int, int],
    views: int,
    z_fast: bool,
    aligned: bool,
    z0: int,
    depth: int,
) -> CuPyValue:
    """Project the stored planes, retaining the coordinates of the full volume."""
    nx, ny, nz, nu, nv = dims
    args = tuple(map(np.int32, dims))
    vector = aligned and z_fast and z0 % 4 == 0 and depth % 4 == 0 and nv >= 4
    # A slab only contributes rows whose support reaches it.
    out = cp.zeros((views, nv, nu), cp.float32)
    if vector:
        tile = min(8, 1 << ((nv // 4).bit_length() - 1))
        tiles = -(-nu // (128 // tile)) * -(-depth // (4 * tile))
        code = storage_source(z_fast=True) + FORWARD_VEC4.replace("TV", str(tile))
        kernel = raw_kernel(code, "axial_forward_vec4", ("--fmad=false",))
        chunk = max(16, -(-512 // tiles))
        for first in range(0, views, chunk):
            kernel(
                tile_grid(min(chunk, views - first), tiles),
                (128,),
                (coeff, volume, out, *args, *map(np.int32, (first, z0, depth))),
            )
    else:
        tile = min(16, 1 << (nv.bit_length() - 1)) if z_fast else 1
        tiles = -(-nu // (128 // tile)) * -(-nv // tile)
        code = storage_source(z_fast=z_fast) + FORWARD.replace("TV", str(tile))
        kernel = raw_kernel(code, "axial_forward", ("--fmad=false",))
        kernel(
            tile_grid(views, tiles),
            (128,),
            (coeff, volume, out, *args, *map(np.int32, (z0, depth, z0 + depth, z0 - 1))),
        )
    return out


def axial_coefficients(coeff: jax.Array | NDArray[np.float32]) -> jax.Array | np.bool_:
    """Whether all ray planes separate horizontal coordinates from axial z."""
    a = coeff[:, 0]
    x = (a == 0) & (coeff[:, 2] == 0) & (coeff[:, 5] == 0) & (coeff[:, 7] == 0)
    y = (a == 1) & (coeff[:, 1] == 0) & (coeff[:, 6] == 0) & (coeff[:, 3] == 0)
    return (x | y).all()


def aligned_axial(coeff: jax.Array | NDArray[np.float32]) -> jax.Array | np.bool_:
    """Whether all views are axial with unit vertical scale and rows aligned to four planes."""
    x = (coeff[:, 0] == 0) & (coeff[:, 6] == 1) & (coeff[:, 8] % 4 == 0)
    y = (coeff[:, 0] == 1) & (coeff[:, 2] == 1) & (coeff[:, 4] % 4 == 0)
    return axial_coefficients(coeff) & (x | y).all()


@cache
def transpose_source(*, vector: bool, cubic: bool, z_fast: bool, vu: bool) -> str:
    text = _TRANSPOSE.replace("RUN", "8" if vector else "4")
    text = text.replace("VECTOR", str(int(vector))).replace("CUBIC", str(int(cubic)))
    text = storage_source(z_fast=z_fast) + text
    if not vu:
        return text
    return text.replace("images[((long)view*nu+u)*nv+v]", "images[((long)view*nv+v)*nu+u]")


def transpose_into(
    cp: CuPyValue,
    coeff: CuPyValue,
    images: CuPyValue,
    target: CuPyValue,
    *,
    shape: tuple[int, int, int],
    image_shape: tuple[int, int, int],
    z0: int = 0,
    cubic: bool,
    z_fast: bool,
    vu: bool,
    batch: int | None = None,
) -> None:
    """Accumulate the separable transpose on the current CuPy stream."""
    nx, ny, nz = shape
    views, nu, nv = image_shape
    if vu:
        nu, nv = nv, nu
    size, chunks = view_chunks(views, nu, nz, batch)
    # The resident and slab budgets allow one detector batch of scratch. Resample
    # tall volumes a band at a time when there are fewer detector rows than planes.
    depth = min(nv, nz) if vu else nz
    if 8 <= depth < nz:
        depth -= depth % 8
    weighted = cp.empty((size, nu, depth), cp.float32)
    for first_z in range(0, nz, depth):
        planes = min(depth, nz - first_z)
        band = target
        if first_z:
            band = target[:, :, first_z:] if z_fast else target[first_z:]
        # Aligned float4 loads amortize global-load instructions. Odd axial
        # extents retain guarded scalar reads, including the last partial run.
        vector = planes % 8 == 0
        run, threads = (8, 256) if vector else (4, 128)
        text = transpose_source(vector=vector, cubic=cubic, z_fast=z_fast, vu=vu)
        vertical = raw_kernel(text, "joseph_vertical")
        horizontal = raw_kernel(text, "joseph_axial")
        runs = nx * ny * -(-planes // run)
        for first, count in chunks:
            c = coeff[first : first + count]
            y = images[first : first + count]
            vertical(
                (-(-(count * nu * planes) // 256),),
                (256,),
                (c, y, weighted, *map(np.int32, (count, planes, nu, nv, z0 + first_z))),
            )
            horizontal(
                (-(-runs // threads),),
                (threads,),
                (
                    c,
                    weighted,
                    band,
                    *map(np.int32, (count, nx, ny, planes, nu, 0, nz)),
                ),
            )
