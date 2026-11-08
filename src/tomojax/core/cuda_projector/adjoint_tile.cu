#include "joseph_sampling.cuh"
#include "joseph_gather.cuh"
#include "joseph_tile.cuh"

extern "C" __global__ void joseph_gather_tile(
    const float* __restrict__ coeff,const float* __restrict__ images,
    float* out,const float* initial,int nviews,int nx,int ny,int nz,int nu,int nv) {
    const int A=AXIS,B=4,C=4,VB=32;
    int na=A==0?nx:(A==1?ny:nz),nb=A==0?ny:(A==1?nz:nx),nc=A==0?nz:(A==1?nx:ny);
    int tiles_b=(nb+B-1)/B,tiles_c=(nc+C-1)/C;
    long tile=(long)blockIdx.x*blockDim.x+threadIdx.x;
    bool valid=tile<(long)na*tiles_b*tiles_c;
    int k=tile/((long)tiles_b*tiles_c);
    int b0=B*(A==1?tile%tiles_b:(tile/tiles_c)%tiles_b);
    int c0=C*(A==1?(tile/tiles_b)%tiles_c:tile%tiles_c);
    float total[B*C]={};
    bool active=false;
    __shared__ float cf[VB*14];
    for(int first=0;first<nviews;first+=VB) {
        __syncthreads();
        for(int i=threadIdx.x;i<VB*14;i+=blockDim.x)
            cf[i]=first*14+i<nviews*14?coeff[first*14+i]:0.f;
        __syncthreads();
        int count=min(VB,nviews-first);
        for(int view=0;view<count;++view) {
            const float* c=cf+view*14;
            if((int)c[0]!=A) continue;
            active=true;
            if(valid) {
                DetectorColumns image{images+(long)(first+view)*nu*nv,nv};
                gather_tile<B,C>(c,image,k,b0,c0,nu,nv,total);
            }
        }
    }
    // Empty axes need no stores when the callback aliases its initializer.
    if(!valid || (!active && out==initial)) return;
    #pragma unroll
    for(int b=0;b<B;++b) {
        #pragma unroll
        for(int c=0;c<C;++c) if(b0+b<nb && c0+c<nc) {
            int ix=A==0?k:(A==1?c0+c:b0+b);
            int iy=A==0?b0+b:(A==1?k:c0+c);
            int iz=A==0?c0+c:(A==1?b0+b:k);
            long index=VOXEL(long,ix,iy,iz);
            out[index]=initial[index]+total[b*C+c];
        }
    }
}
