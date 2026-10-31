#include "joseph_sampling.cuh"
#ifndef COEFF_OFFSET
#define COEFF_OFFSET 0
#define VOLUME_OFFSET 0
#endif
#define RADIUS (CUBIC ? 2.f : 1.f)

__device__ float sample(const float* x, int a, int k, int b, int c,
                       int nx, int ny, int nz, int nb, int nc) {
    if (b<0 || b>=nb || c<0 || c>=nc) return 0.f;
    int ix=a==0?k:(a==1?c:b), iy=a==0?b:(a==1?k:c), iz=a==0?c:(a==1?b:k);
    return x[VOXEL(INDEX_TYPE, ix, iy, iz)];
}
extern "C" __global__ void forward(const float* coeff, const float* x, float* out,
        int nx,int ny,int nz,int nu,int nv,int nviews) {
    int tiles_v=(nv+TILE_V-1)/TILE_V;
    int tiles=((nu+BLOCK/TILE_V-1)/(BLOCK/TILE_V))*tiles_v;
    long batch=blockIdx.x/(nviews*tiles);
    coeff+=COEFF_OFFSET;x+=VOLUME_OFFSET;out+=batch*nviews*nv*nu;
    int tile=(blockIdx.x/nviews)%tiles, view=blockIdx.x%nviews;
    int u=(tile/tiles_v)*(BLOCK/TILE_V)+threadIdx.x/TILE_V;
    int v=(tile%tiles_v)*TILE_V+threadIdx.x%TILE_V;
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
            if (start < -RADIUS-margin || start > size+RADIUS-1.f+margin) { low=na; high=-1.f; }
        } else {
            float a=(-RADIUS-margin-start)/slope, b=(size+RADIUS-1.f+margin-start)/slope;
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
    __shared__ int starts[BLOCK/32], stops[BLOCK/32];
    if(lane==0) { starts[warp]=first; stops[warp]=stop; }
    __syncthreads();
    first=starts[0];stop=stops[0];
    for(int w=1;w<BLOCK/32;w++){first=min(first,starts[w]);stop=max(stop,stops[w]);}
    for(int k=first;k<stop;k++) {
        if(!valid) continue;
        float qb=pb+(c[3]*k+c[4]),qc=pc+(c[7]*k+c[8]);
        float val;
#if CUBIC
#if ABSOLUTE
        int ib=(int)floorf(qb),ic=(int)floorf(qc);
        float wb[4],wc[4],db[4],dc[4];
        joseph_cubic_weights(qb-ib,wb,db);joseph_cubic_weights(qc-ic,wc,dc);
        val=0.f;
        #pragma unroll
        for(int b=0;b<4;++b) {
            float row=0.f;
            #pragma unroll
            for(int c=0;c<4;++c)
                row+=sample(x,a,k,ib+b-1,ic+c-1,nx,ny,nz,nb,nc)*fabsf(wc[c]);
            val+=row*fabsf(wb[b]);
        }
#else
        float db,dc;
        joseph_value_derivatives<true>(x,a,k,qb,qc,nx,ny,nz,nb,nc,val,db,dc);
#endif
#else
        int ib=(int)floorf(qb),ic=(int)floorf(qc);
        float wb=qb-ib,wc=qc-ic;
        val=(sample(x,a,k,ib,ic,nx,ny,nz,nb,nc)*(1-wc)
             +sample(x,a,k,ib,ic+1,nx,ny,nz,nb,nc)*wc)*(1-wb)
            +(sample(x,a,k,ib+1,ic,nx,ny,nz,nb,nc)*(1-wc)
             +sample(x,a,k,ib+1,ic+1,nx,ny,nz,nb,nc)*wc)*wb;
#endif
        // Keep the original sequential additions, even for uniform support.
        total+=val*c[9];
    }
    if(valid) out[((long)view*nv+v)*nu+u]=total;
}
