// Sampled parallel-ray pair and fused residual gradient.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

constexpr int sample_unroll = UNROLL;

__device__ float read_sample(const STORAGE* x, int a, int b, int c) {
    if ((unsigned)a>=NX || (unsigned)b>=NY || (unsigned)c>=NZ) return 0.f;
    return (float)x[((long)a*NY+b)*NZ+c];
}
__device__ float sample(const STORAGE* x, float a, float b, float c) {
    int ia=(int)floorf(a),ib=(int)floorf(b),ic=(int)floorf(c);
    float wa=a-ia,wb=b-ib,wc=c-ic;
    if (ZINTEGER) {
        ic=(int)floorf(c+.5f);
        float c0=read_sample(x,ia,ib,ic)*(1-wb)+read_sample(x,ia,ib+1,ic)*wb;
        float c1=read_sample(x,ia+1,ib,ic)*(1-wb)+read_sample(x,ia+1,ib+1,ic)*wb;
        return c0*(1-wa)+c1*wa;
    }
    float c00=read_sample(x,ia,ib,ic)*(1-wc)+read_sample(x,ia,ib,ic+1)*wc;
    float c01=read_sample(x,ia,ib+1,ic)*(1-wc)+read_sample(x,ia,ib+1,ic+1)*wc;
    float c10=read_sample(x,ia+1,ib,ic)*(1-wc)+read_sample(x,ia+1,ib,ic+1)*wc;
    float c11=read_sample(x,ia+1,ib+1,ic)*(1-wc)+read_sample(x,ia+1,ib+1,ic+1)*wc;
    return (c00*(1-wb)+c01*wb)*(1-wa)+(c10*(1-wb)+c11*wb)*wa;
}
__device__ void scatter(float* out,float val,float a,float b,float c) {
    if (ZINTEGER) c=floorf(c+.5f);
    int ia=(int)floorf(a),ib=(int)floorf(b),ic=(int)floorf(c);
    float wa=a-ia,wb=b-ib,wc=c-ic;
    #pragma unroll
    for (int da=0;da<2;da++) {
        #pragma unroll
        for (int db=0;db<2;db++) {
            #pragma unroll
            for (int dc=0;dc<2;dc++) {
                int i=ia+da,j=ib+db,k=ic+dc;
                float weight=((da?wa:1-wa)*(db?wb:1-wb))*(dc?wc:1-wc);
                if ((unsigned)i<NX && (unsigned)j<NY && (unsigned)k<NZ && weight!=0.f)
                    atomicAdd(out+((long)i*NY+j)*NZ+k,val*weight);
            }
        }
    }
}
__device__ int traverse(const float* t,int u,int v,float (&start)[3],float (&delta)[3]) {
    float xr=((float)u-(NU/2.f-.5f))*DU+CU;
    float zr=((float)v-(NV/2.f-.5f))*DV+CV;
    float origin[3]={OX,OY,OZ},voxel[3]={VX,VY,VZ};
    float lower[3]={LOX,LOY,LOZ},upper[3]={HIX,HIY,HIZ};
    float base[3],dir[3],entry=-CUDART_INF_F,exit=CUDART_INF_F;
    #pragma unroll
    for (int a=0;a<3;a++) {
        dir[a]=t[4+a];
        float inverse=-((t[a]*t[3]+t[4+a]*t[7])+t[8+a]*t[11]);
        base[a]=(t[a]*xr+t[8+a]*zr)+inverse;
        float low=lower[a], high=upper[a];
        bool parallel=fabsf(dir[a])<1.e-8f;
        float denom=parallel?1.f:dir[a];
        float p=(low-base[a])/denom,q=(high-base[a])/denom;
        bool inside=base[a]>=low && base[a]<=high;
        entry=fmaxf(entry,parallel?(inside?-CUDART_INF_F:CUDART_INF_F):fminf(p,q));
        exit=fminf(exit,parallel?(inside?CUDART_INF_F:-CUDART_INF_F):fmaxf(p,q));
    }
    float length=fmaxf(0.f,exit-entry), time=length>0.f?entry:0.f;
    #pragma unroll
    for (int a=0;a<3;a++) {
        start[a]=((base[a]+time*dir[a])-origin[a])/voxel[a];
        delta[a]=(STEP*dir[a])/voxel[a];
    }
    return length>0.f ? min((int)ceilf(length/STEP),NSTEPS):0;
}
extern "C" __global__ void integrate(const float* poses,const STORAGE* volume,
    const float* targets,const float* weights,float* out,float* gradient,int views
#if CACHED
    , const float* ix, const float* iy, const float* iz, const int* counts,
    const float* dx, const float* dy, const float* dz
#endif
) {
    int tile=blockIdx.x/views,view=blockIdx.x%views;
    int tiles_u=(NU+TU-1)/TU;
    int u=(tile%tiles_u)*TU+(LAYOUT?threadIdx.x/TV:threadIdx.x%TU);
    int v=(tile/tiles_u)*TV+(LAYOUT?threadIdx.x%TV:threadIdx.x/TU);
    bool valid=threadIdx.x<TU*TV && u<NU && v<NV;
    long ray=((long)view*NV+v)*NU+u;
    float start[3],delta[3];
#if CACHED
    start[0]=valid?ix[ray]:0.f; start[1]=valid?iy[ray]:0.f; start[2]=valid?iz[ray]:0.f;
    delta[0]=dx[view]; delta[1]=dy[view]; delta[2]=dz[view];
    int steps=valid?min(counts[ray],NSTEPS):0;
#else
    int steps=traverse(poses+16*view,u,v,start,delta);
    if (!valid) steps=0;
#endif
    float p[3]={start[0],start[1],start[2]},acc=0.f;
    if (MODE!=1) {
        #pragma unroll sample_unroll
        for(int k=0;k<steps;k++) {
            acc+=sample(volume,p[0],p[1],p[2])*STEP;
            #pragma unroll
            for(int a=0;a<3;a++) p[a]+=delta[a];
        }
    }
    float loss=0.f;
    if (MODE==0) { if(valid) out[ray]=acc; }
    if (MODE>=2) {
        float residual=valid ? acc-targets[ray]:0.f;
        float weight=MODE==2?weights[view]:1.f;
        if(MODE==3 || COMPUTE_LOSS) loss=(residual*weight)*(residual*weight);
        acc=((residual*weight)*weight)*STEP;
    }
    if (MODE==1 || MODE==2) {
        if(MODE==1) acc=valid?targets[ray]*STEP:0.f;
        #pragma unroll
        for(int a=0;a<3;a++) p[a]=start[a];
        #pragma unroll sample_unroll
        for(int k=0;k<steps;k++) {
            scatter(gradient,acc,p[0],p[1],p[2]);
            #pragma unroll
            for(int a=0;a<3;a++) p[a]+=delta[a];
        }
    }
    if(MODE>=2) {
        __shared__ float partial[BLOCK];
        partial[threadIdx.x]=loss;__syncthreads();
        for(int stride=BLOCK/2;stride;stride/=2) {
            if(threadIdx.x<stride) partial[threadIdx.x]+=partial[threadIdx.x+stride];
            __syncthreads();
        }
        if(threadIdx.x==0) out[(long)view*((NU+TU-1)/TU)*((NV+TV-1)/TV)+tile]=(MODE==2?.5f:1.f)*partial[0];
    }
}
