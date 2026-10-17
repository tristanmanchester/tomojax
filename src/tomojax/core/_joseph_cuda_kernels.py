"""JAX-independent CUDA kernels for the exact linear Joseph projector pair."""

from functools import cache
from typing import Any

RUN = 4
_VIEWS_PER_STAGE = 32
THREADS = 128

_GATHER_SOURCE = r"""
template<int A, int R>
__device__ __forceinline__ void gather_view(
    const float* c, const float* image, int ix, int iy, int iz0,
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
    float lo_u = 3.4e38f, hi_u = -3.4e38f, lo_v = 3.4e38f, hi_v = -3.4e38f;
    #pragma unroll
    for (int j = 0; j < R; ++j) {
        float k = (float)(k0 + j * dk);
        hb[j] = __fadd_rn(__fmul_rn(kb, k), cb);
        hc[j] = __fadd_rn(__fmul_rn(kc, k), cc);
        tb[j] = (float)(b0 + j * db);
        tc[j] = (float)(c0 + j * dc);
        float cu = iub * (tb[j] - hb[j]) + iuc * (tc[j] - hc[j]);
        float cv = ivb * (tb[j] - hb[j]) + ivc * (tc[j] - hc[j]);
        lo_u = fminf(lo_u, cu); hi_u = fmaxf(hi_u, cu);
        lo_v = fminf(lo_v, cv); hi_v = fmaxf(hi_v, cv);
    }
    float ru = fabsf(iub) + fabsf(iuc), rv = fabsf(ivb) + fabsf(ivc);
    // Bound inverse-coordinate roundoff without expanding the physical weights.
    float eb = fmaxf(fabsf(tb[0]), fabsf(tb[R-1])) + fmaxf(fabsf(hb[0]), fabsf(hb[R-1]));
    float ec = fmaxf(fabsf(tc[0]), fabsf(tc[R-1])) + fmaxf(fabsf(hc[0]), fabsf(hc[R-1]));
    float mu = 8*1.1920928955078125e-7f*(fabsf(iub)*eb + fabsf(iuc)*ec + ru + 1);
    float mv = 8*1.1920928955078125e-7f*(fabsf(ivb)*eb + fabsf(ivc)*ec + rv + 1);
    int u_start = max((int)floorf(lo_u - ru - mu) + 1, 0);
    int u_stop = min((int)ceilf(hi_u + ru + mu), nu);
    int v_start = max((int)floorf(lo_v - rv - mv) + 1, 0);
    int v_stop = min((int)ceilf(hi_v + rv + mv), nv);
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
                total[j] += value * (fmaxf(1.f - fabsf(qb - tb[j]), 0.f)
                                     * fmaxf(1.f - fabsf(qc - tc[j]), 0.f));
            }
        }
    }
}

extern "C" __global__ void joseph_gather(
    const float* __restrict__ coeff, const float* __restrict__ images,
    float* __restrict__ out, int nviews, int nx, int ny, int nz, int nu, int nv)
{
    const int R = RUN, VB = VIEWS;
    __shared__ float cf[VB * 14];
    long run = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int runs_z = (nz + R - 1) / R;
    bool valid = run < (long)nx * ny * runs_z;
    int ix = valid ? run / ((long)ny * runs_z) : 0;
    int iy = valid ? (run / runs_z) % ny : 0;
    int iz0 = valid ? R * (int)(run % runs_z) : 0;
    float total[R];
    #pragma unroll
    for (int j = 0; j < R; ++j) total[j] = 0.f;
    for (int first = 0; first < nviews; first += VB) {
        __syncthreads();
        for (int i = threadIdx.x; i < VB * 14; i += blockDim.x)
            cf[i] = first * 14 + i < nviews * 14 ? coeff[first * 14 + i] : 0.f;
        __syncthreads();
        int count = min(VB, nviews - first);
        for (int view = 0; view < count; ++view) {
            const float* c = cf + view * 14;
            const float* image = images + (long)(first + view) * nu * nv;
            switch ((int)c[0]) {
                case 0: gather_view<0,R>(c, image, ix, iy, iz0, nu, nv, total); break;
                case 1: gather_view<1,R>(c, image, ix, iy, iz0, nu, nv, total); break;
                case 2: gather_view<2,R>(c, image, ix, iy, iz0, nu, nv, total); break;
            }
        }
    }
    if (valid) {
        long base = ((long)ix * ny + iy) * nz + iz0;
        #pragma unroll
        for (int j = 0; j < R; ++j)
            if (iz0 + j < nz) out[base + j] += total[j];
    }
}
"""


@cache
def gather_kernel() -> Any:
    import cupy as cp

    source = _GATHER_SOURCE.replace("RUN", str(RUN)).replace("VIEWS", str(_VIEWS_PER_STAGE))
    return cp.RawModule(code=source).get_function("joseph_gather")


_FORWARD_SOURCE = r"""
__device__ float sample(const float* x, int a, int k, int b, int c,
                       int nx, int ny, int nz, int nb, int nc) {
    if (b<0 || b>=nb || c<0 || c>=nc) return 0.f;
    if (UNIFORM) return 1.f;
    int ix=a==0?k:(a==1?c:b), iy=a==0?b:(a==1?k:c), iz=a==0?c:(a==1?b:k);
    return x[((INDEX)ix*ny+iy)*nz+iz];
}
extern "C" __global__ void forward(const float* coeff, const float* x, float* out,
        int nx,int ny,int nz,int nu,int nv,int nviews) {
    int tiles_v=(nv+15)/16;
    int tile=blockIdx.x/nviews, view=blockIdx.x%nviews;
    int u=(tile/tiles_v)*8+threadIdx.x/16;
    int v=(tile%tiles_v)*16+threadIdx.x%16;
    bool valid=(u<nu && v<nv);
    const float* c=coeff+14*view;
    int a=(int)c[0];
    int na=a==0?nx:(a==1?ny:nz),nb=a==0?ny:(a==1?nz:nx),nc=a==0?nz:(a==1?nx:ny);
    float pb=c[1]*u+c[2]*v,pc=c[5]*u+c[6]*v;
    float total=0.f;
    float low=0.f, high=na-1.f;
    for (int d=0;d<2;d++) {
        float p=d?pc:pb, slope=d?c[7]:c[3], offset=d?c[8]:c[4];
        float size=d?nc:nb, start=p+offset;
        // Bound coordinate roundoff before computing a support interval.
        float margin=8*1.1920928955078125e-7f*(fabsf(p)+fabsf(offset)+fabsf(slope)*na+1.f);
        if (slope==0.f) {
            if (start < -1.f-margin || start > size+margin) { low=na; high=-1.f; }
        } else {
            float a=(-1.f-margin-start)/slope, b=(size+margin-start)/slope;
            low=fmaxf(low,fminf(a,b)); high=fminf(high,fmaxf(a,b));
        }
    }
    int first=(int)fminf((float)na,fmaxf(0.f,floorf(low)-1.f));
    int stop=(int)fminf((float)na,fmaxf(0.f,ceilf(high)+2.f));
    if (!valid) { first=na; stop=0; }
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    for(int delta=16;delta>0;delta/=2) {
        first=min(first,__shfl_down_sync(0xffffffff,first,delta));
        stop=max(stop,__shfl_down_sync(0xffffffff,stop,delta));
    }
    __shared__ int starts[4], stops[4];
    if(lane==0) { starts[warp]=first; stops[warp]=stop; }
    __syncthreads();
    first=min(min(starts[0],starts[1]),min(starts[2],starts[3]));
    stop=max(max(stops[0],stops[1]),max(stops[2],stops[3]));
    for(int k=first;k<stop;k++) {
        if(!valid) continue;
        float qb=pb+(c[3]*k+c[4]),qc=pc+(c[7]*k+c[8]);
        float val;
        if (UNIFORM && qb>=0.f && qb<(float)(nb-1) && qc>=0.f && qc<(float)(nc-1)) {
            val=1.f;
        } else {
            int ib=(int)floorf(qb),ic=(int)floorf(qc);
            float wb=qb-ib,wc=qc-ic;
            val=(sample(x,a,k,ib,ic,nx,ny,nz,nb,nc)*(1-wc)
                  +sample(x,a,k,ib,ic+1,nx,ny,nz,nb,nc)*wc)*(1-wb)
                 +(sample(x,a,k,ib+1,ic,nx,ny,nz,nb,nc)*(1-wc)+sample(x,a,k,ib+1,ic+1,nx,ny,nz,nb,nc)*wc)*wb;
        }
        // Keep the original sequential additions, even for uniform support.
        total+=val*c[9];
    }
    if(valid) out[((long)view*nv+v)*nu+u]=total;
}
"""


@cache
def forward_kernel(*, narrow: bool, uniform: bool = False) -> Any:
    import cupy as cp

    source = _FORWARD_SOURCE.replace("INDEX", "int" if narrow else "long").replace(
        "UNIFORM", str(int(uniform))
    )
    return cp.RawModule(code=source, options=("--fmad=false",)).get_function("forward")
