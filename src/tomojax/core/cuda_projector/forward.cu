#include "joseph_sampling.cuh"

#ifndef COEFF_OFFSET
#define COEFF_OFFSET 0
#define VOLUME_OFFSET 0
#endif
#define RADIUS (CUBIC ? 2.f : 1.f)

// Intersect the plane interval with lower <= start + k * slope <= upper.
__device__ __forceinline__ void clip_planes(float start, float slope, float lower, float upper,
                                             float& first, float& last) {
    if (lower > upper || (slope == 0.f && (start < lower || start > upper))) {
        first = 3.4e38f; last = -3.4e38f;
    } else if (slope != 0.f) {
        float a = (lower-start)/slope, b = (upper-start)/slope;
        first = fmaxf(first, fminf(a,b)); last = fminf(last, fmaxf(a,b));
    }
}

// Start neighboring rays on the same plane to keep their volume reads together.
template<int Threads>
__device__ __forceinline__ void block_planes(int& first, int& stop) {
    for (int delta=16; delta>0; delta/=2) {
        first=min(first,__shfl_down_sync(0xffffffff,first,delta));
        stop=max(stop,__shfl_down_sync(0xffffffff,stop,delta));
    }
    __shared__ int starts[Threads/32], stops[Threads/32];
    int warp=threadIdx.x/32;
    if (threadIdx.x%32==0) { starts[warp]=first; stops[warp]=stop; }
    __syncthreads();
    first=starts[0]; stop=stops[0];
    #pragma unroll
    for (int w=1; w<Threads/32; ++w) { first=min(first,starts[w]); stop=max(stop,stops[w]); }
}

// x holds planes z0..z0+zs-1 of z-fast storage (all of them unless BAND). With BAND, a ray
// adds only the samples whose lower z plane lies in [z0, zhi), the first and last bands
// taking the edges, and resumes its sum from out, so bands taken in the ray's z order
// keep the reference addition order (zhi <= nz). Views whose z falls along the ray (descending)
// take the bands top down in their own launches.
extern "C" __global__ void forward(const float* coeff, const float* x, float* out,
        int nx,int ny,int nz,int nu,int nv,int nviews,int z0,int zs,int zhi,int descending) {
    int tiles_v=(nv+TILE_V-1)/TILE_V;
    int tiles=((nu+BLOCK/TILE_V-1)/(BLOCK/TILE_V))*tiles_v;
    long batch=blockIdx.x/(nviews*tiles);
    coeff+=COEFF_OFFSET;x+=VOLUME_OFFSET-(SLAB?0:z0);out+=batch*nviews*nv*nu;
    int tile=(blockIdx.x/nviews)%tiles, view=blockIdx.x%nviews;
    int u=(tile/tiles_v)*(BLOCK/TILE_V)+threadIdx.x/TILE_V;
    int v=(tile%tiles_v)*TILE_V+threadIdx.x%TILE_V;
    bool valid=(u<nu && v<nv);
    const float* c=coeff+14*view;
    int a=(int)c[0];
    int na=a==0?nx:(a==1?ny:nz),nb=a==0?ny:(a==1?nz:nx),nc=a==0?nz:(a==1?nx:ny);
    float pb=c[1]*u+c[2]*v,pc=c[5]*u+c[6]*v;
    float total=0.f;
    float low=SLAB && a==2 ? z0 : 0.f, high=SLAB && a==2 ? z0+zs-1.f : na-1.f;
    float interior_start=low, interior_end=high;
    for (int d=0;d<2;d++) {
        float p=d?pc:pb, slope=d?c[7]:c[3], offset=d?c[8]:c[4];
        float size=d?nc:nb, start=p+offset;
        // Bound coordinate roundoff before computing a support interval.
        float margin=8*1.1920928955078125e-7f*(fabsf(p)+fabsf(offset)+fabsf(slope)*na+1.f);
        bool local_z=SLAB && ((d==0 && a==1) || (d==1 && a==0));
        float lower=local_z?z0:0.f, upper=local_z?z0+zs:size;
        clip_planes(start, slope, lower-RADIUS-margin, upper+RADIUS-1.f+margin, low, high);
        if (UNIFORM) {
            float guard=slope==0.f?0.f:margin;
            clip_planes(start, slope, lower+guard, upper-1.f-guard, interior_start, interior_end);
        }
    }
#if BAND
    int zlo=z0==0?-1:z0;
    float zslope=a==0?c[7]:(a==1?c[3]:1.f);
    if ((zslope<0.f)!=(descending!=0)) valid=false;
    if (a==2) { low=fmaxf(low,(float)zlo); high=fminf(high,zhi-1.f); }
    else {
        float p=a==0?pc:pb, offset=a==0?c[8]:c[4], start=p+offset;
        float margin=8*1.1920928955078125e-7f*(fabsf(p)+fabsf(offset)+fabsf(zslope)*na+1.f);
        clip_planes(start, zslope, zlo-1.f-margin, zhi+1.f+margin, low, high);
    }
    if (valid) total=out[((long)view*nv+v)*nu+u];
#endif
    int first=(int)fminf((float)na,fmaxf(0.f,floorf(low)-1.f));
    int stop=(int)fminf((float)na,fmaxf(0.f,ceilf(high)+2.f));
    if (!valid) { first=na; stop=0; }
    if (!UNIFORM) block_planes<BLOCK>(first, stop);
    int inside=(int)ceilf(fminf((float)stop,fmaxf((float)first,interior_start)));
    int end=(int)floorf(fminf((float)(stop-1),fmaxf((float)(first-1),interior_end)))+1;
    const INDEX_TYPE sx=VOXEL(INDEX_TYPE,1,0,0), sy=VOXEL(INDEX_TYPE,0,1,0), sz=Z_STRIDE;
    const INDEX_TYPE sa=a==0?sx:(a==1?sy:sz), sb=a==0?sy:(a==1?sz:sx), sc=a==0?sz:(a==1?sx:sy);
    for(int k=first;k<stop;k++) {
        if(!valid) continue;
        // A fully covered unit sample contributes exactly one path weight.
        // Preserve the sequential FP32 sum while skipping its interpolation.
        if (UNIFORM && k==inside && k<end) {
            for (; k<end; ++k) total+=c[9];
            --k;
            continue;
        }
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
                row+=joseph_sample<INDEX_TYPE>(x,a,k,ib+b-1,ic+c-1,nx,ny,zs,nb,nc)*fabsf(wc[c]);
            val+=row*fabsf(wb[b]);
        }
#else
        float db,dc;
        joseph_value_derivatives<true>(x,a,k,qb,qc,nx,ny,zs,nb,nc,val,db,dc);
#endif
#else
        int ib=(int)floorf(qb),ic=(int)floorf(qc);
#if BAND
        int iz=a==0?ic:(a==1?ib:k);
        if (iz<zlo || iz>=zhi) continue;
        const bool interior = false;
#else
        const bool interior = !UNIFORM && !SLAB && (unsigned)ib < (unsigned)(nb-1) && (unsigned)ic < (unsigned)(nc-1);
#endif
        float f00, f01, f10, f11;
        if (interior) {
            const float* p = x + k*sa + ib*sb + ic*sc;
            f00=p[0];  f01=p[sc];
            f10=p[sb]; f11=p[sb+sc];
        } else {
            f00=joseph_sample<INDEX_TYPE,UNIFORM,SLAB>(x,a,k,ib,ic,nx,ny,zs,nb,nc,z0);
            f01=joseph_sample<INDEX_TYPE,UNIFORM,SLAB>(x,a,k,ib,ic+1,nx,ny,zs,nb,nc,z0);
            f10=joseph_sample<INDEX_TYPE,UNIFORM,SLAB>(x,a,k,ib+1,ic,nx,ny,zs,nb,nc,z0);
            f11=joseph_sample<INDEX_TYPE,UNIFORM,SLAB>(x,a,k,ib+1,ic+1,nx,ny,zs,nb,nc,z0);
        }
        float wb=qb-ib,wc=qc-ic;
        val=(f00*(1-wc)+f01*wc)*(1-wb)+(f10*(1-wc)+f11*wc)*wb;
#endif
        // Keep the original sequential additions, even for uniform support.
        total+=val*c[9];
    }
    if(valid) out[((long)view*nv+v)*nu+u]=total;
}
