// Exact first-order Joseph derivatives, retaining constant per-ray state.
#include "joseph_sampling.cuh"

__device__ void sample_plane(const float* volume, int axis, int k, float b, float c,
    float &value, float &db, float &dc) {
    int nb=axis==0?NY:(axis==1?NZ:NX),nc=axis==0?NZ:(axis==1?NX:NY);
    joseph_value_derivatives<CUBIC>(volume,axis,k,b,c,NX,NY,NZ,nb,nc,value,db,dc);
}

__device__ void ray_sums(const float* cf, const float* volume, int u, int v,
    float (&sum)[6]) {
    int axis=(int)cf[0],na=axis==0?NX:(axis==1?NY:NZ);
    for (int k=0;k<na;k++) {
        float qb=(cf[1]*u+cf[2]*v)+(cf[3]*k+cf[4]);
        float qc=(cf[5]*u+cf[6]*v)+(cf[7]*k+cf[8]);
        float value,db,dc;
        sample_plane(volume,axis,k,qb,qc,value,db,dc);
        db*=cf[9];dc*=cf[9];
        sum[0]+=db;sum[1]+=k*db;sum[2]+=dc;sum[3]+=k*dc;
        sum[4]+=value;sum[5]+=value*cf[9];
    }
}

__device__ float ray_jvp(const float* cf,const float* volume,const float* dc,
    const float* dvolume,int u,int v) {
    int axis=(int)cf[0],na=axis==0?NX:(axis==1?NY:NZ);
    float total=0.f;
    for(int k=0;k<na;k++) {
        float qb=(cf[1]*u+cf[2]*v)+(cf[3]*k+cf[4]);
        float qc=(cf[5]*u+cf[6]*v)+(cf[7]*k+cf[8]);
        float tangent=0.f,value,db,dq;
#if HAS_COEFF
        sample_plane(volume,axis,k,qb,qc,value,db,dq);
        float tb=(dc[1]*u+dc[2]*v)+(dc[3]*k+dc[4]);
        float tc=(dc[5]*u+dc[6]*v)+(dc[7]*k+dc[8]);
        tangent=(db*tb+dq*tc)*cf[9]+value*dc[9];
#endif
#if HAS_VOLUME
        sample_plane(dvolume,axis,k,qb,qc,value,db,dq);
        tangent+=value*cf[9];
#endif
        total+=tangent;
    }
    return total;
}

extern "C" __global__ void derivatives(const float* coeff,const float* volume,
    const float* data,const float* directions,const float* dvolume,
    float* partials,float* residuals,int views,int tiles) {
    long batch=blockIdx.x/((long)views*tiles), block=blockIdx.x%((long)views*tiles);
    coeff+=COEFF_OFFSET;volume+=VOLUME_OFFSET;data+=DATA_OFFSET;
    directions+=DIRECTIONS_OFFSET;dvolume+=DVOLUME_OFFSET;
#if MODE!=2
    partials+=batch*(long)views*tiles*WIDTH;
#endif
#if MODE!=0
    residuals+=batch*(long)views*NV*NU;
#endif
    int view=block%views,tile=block/views;
    int ray=tile*blockDim.x+threadIdx.x;
    int u=ray/NV,v=ray%NV;
    bool valid=u<NU;
    long pixel=((long)view*NV+v)*NU+u;
    const float* cf=coeff+14*view;
#if MODE==2
    if(valid) residuals[pixel]=ray_jvp(cf,volume,data+14*view,dvolume,u,v);
#else
    float sum[6]={},values[WIDTH]={};
    if(valid) ray_sums(cf,volume,u,v,sum);
    float scalar=valid?data[pixel]:0.f;
#if MODE!=0
    scalar=valid?sum[5]-scalar:0.f;
    if(valid) residuals[pixel]=scalar;
#endif
#if MODE==3
    float columns[PARAMETERS];
    for(int d=0;d<PARAMETERS;d++) {
        const float* dc=directions+(view*14)*PARAMETERS+d;
        float col=(((sum[0]*((dc[PARAMETERS]*u+dc[2*PARAMETERS]*v)+dc[4*PARAMETERS])
                     +sum[1]*dc[3*PARAMETERS])
                     +sum[2]*((dc[5*PARAMETERS]*u+dc[6*PARAMETERS]*v)+dc[8*PARAMETERS]))
                     +sum[3]*dc[7*PARAMETERS])+sum[4]*dc[9*PARAMETERS];
        columns[d]=col;
        values[1+d]=col*scalar;
    }
    values[0]=.5f*scalar*scalar;
    int offset=1+PARAMETERS;
    for(int i=0;i<PARAMETERS;i++) for(int j=i;j<PARAMETERS;j++) values[offset++]=columns[i]*columns[j];
#else
    float gb=sum[0]*scalar,gkb=sum[1]*scalar,gc=sum[2]*scalar,gkc=sum[3]*scalar,gw=sum[4]*scalar;
    values[1]=gb*u;values[2]=gb*v;values[3]=gkb;values[4]=gb;
    values[5]=gc*u;values[6]=gc*v;values[7]=gkc;values[8]=gc;values[9]=gw;
    if(MODE==1) values[15]=.5f*scalar*scalar;
#endif
    __shared__ float warps[4][WIDTH];
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int i=0;i<WIDTH;i++) {
        float value=values[i];
        for(int shift=16;shift;shift/=2) value+=__shfl_down_sync(0xffffffff,value,shift);
        if(lane==0) warps[warp][i]=value;
    }
    __syncthreads();
    for(int i=threadIdx.x;i<WIDTH;i+=blockDim.x)
        partials[((long)view*tiles+tile)*WIDTH+i]=(warps[0][i]+warps[1][i])+(warps[2][i]+warps[3][i]);
#endif
}
