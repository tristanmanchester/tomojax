// Weights use global coordinates; integer offsets address local arrays.
struct RegionVolume {
    const float* data;
    int nx,ny,nz,ox,oy,oz;
};
__device__ __forceinline__ float joseph_sample(RegionVolume x,int a,int k,int b,int c,
        int,int,int,int,int) {
    int ix=(a==0?k:(a==1?c:b))-x.ox;
    int iy=(a==0?b:(a==1?k:c))-x.oy;
    int iz=(a==0?c:(a==1?b:k))-x.oz;
    if(ix<0 || ix>=x.nx || iy<0 || iy>=x.ny || iz<0 || iz>=x.nz) return 0.f;
    return x.data[((long)ix*x.ny+iy)*x.nz+iz];
}
struct DetectorRegion {
    const float* data;
    int nu,nv,u0,v0;
    __device__ __forceinline__ float at(int u,int v) const {
        u-=u0;v-=v0;
        if(u<0 || u>=nu || v<0 || v>=nv) return 0.f;
        return data[(long)v*nu+u];
    }
};
__device__ __forceinline__ float joseph_pixel(DetectorRegion image,int,int u,int v) {
    return image.at(u,v);
}
#include "joseph_sampling.cuh"
#include "joseph_gather.cuh"

__device__ __forceinline__ void clip_region_planes(float pixel,float slope,float offset,
        int lower,int size,int full_planes,int& first,int& last) {
    double base=(double)pixel+offset;
    double error=4.*1.1920928955078125e-7*(fabs((double)pixel)+fabs((double)offset)
        +fabs((double)slope)*(full_planes-1)+1.);
    double low=(double)lower-(CUBIC?2:1)-error;
    double high=(double)lower+size+(CUBIC?2:1)-1.+error;
    if(slope==0.f) { if(base<low || base>high) last=first;return; }
    double lo=(low-base)/slope,hi=(high-base)/slope;
    if(lo>hi) { double tmp=lo;lo=hi;hi=tmp; }
    first=max(first,(int)fmax(0.,fmin((double)full_planes,ceil(lo)-1.)));
    last=min(last,(int)fmax(0.,fmin((double)full_planes,floor(hi)+2.)));
}

extern "C" __global__ void region_columns(const float* coeff,const float* tangent,
        const float* volume,const float* initial,float* out,int nx,int ny,int nz,int sx,int sy,int sz,
        int ox,int oy,int oz,int nu,int nv,int u0,int v0,int views) {
    int tiles_v=(nv+15)/16,tile=blockIdx.x/views,view=blockIdx.x%views;
    int lu=(tile/tiles_v)*8+threadIdx.x/16,lv=(tile%tiles_v)*16+threadIdx.x%16;
    int u=lu+u0,v=lv+v0;bool valid=lu<nu && lv<nv;
    const float* cf=coeff+14*view;
    int a=(int)cf[0];
    int na=a==0?nx:(a==1?ny:nz),nb=a==0?ny:(a==1?nz:nx),nc=a==0?nz:(a==1?nx:ny);
    int oa=a==0?ox:(a==1?oy:oz),ob=a==0?oy:(a==1?oz:ox),oc=a==0?oz:(a==1?ox:oy);
    int sa=a==0?sx:(a==1?sy:sz),sb=a==0?sy:(a==1?sz:sx),sc=a==0?sz:(a==1?sx:sy);
    float pb=cf[1]*u+cf[2]*v,pc=cf[5]*u+cf[6]*v;
    int first=oa,last=oa+sa;
    clip_region_planes(pb,cf[3],cf[4],ob,sb,na,first,last);
    clip_region_planes(pc,cf[7],cf[8],oc,sc,na,first,last);
    if(!valid || first>=last) {first=na;last=0;}
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int delta=16;delta>0;delta/=2) {
        first=min(first,__shfl_down_sync(0xffffffff,first,delta));
        last=max(last,__shfl_down_sync(0xffffffff,last,delta));
    }
    __shared__ int starts[4],stops[4];
    if(lane==0) {starts[warp]=first;stops[warp]=last;}
    __syncthreads();
    first=min(min(starts[0],starts[1]),min(starts[2],starts[3]));
    last=max(max(stops[0],stops[1]),max(stops[2],stops[3]));
    if(!valid)return;
    RegionVolume x={volume,sx,sy,sz,ox,oy,oz};
    float prediction=0.f,columns[DIRECTIONS+1]={};
    for(int k=first;k<last;++k) {
        float qb=pb+(cf[3]*k+cf[4]),qc=pc+(cf[7]*k+cf[8]);
        float value,db,dc;
        joseph_value_derivatives<CUBIC,ABSOLUTE>(x,a,k,qb,qc,nx,ny,nz,nb,nc,value,db,dc);
        prediction+=value*cf[9];
        #pragma unroll
        for(int j=0;j<DIRECTIONS;++j) {
            const float* d=tangent+((long)view*DIRECTIONS+j)*14;
            float tb=(d[1]*u+d[2]*v)+(d[3]*k+d[4]);
            float tc=(d[5]*u+d[6]*v)+(d[7]*k+d[8]);
            columns[j]+=(db*tb+dc*tc)*cf[9]+value*d[9];
        }
    }
    long pixel=(long)lv*nu+lu,pixels=(long)nv*nu;
    long index=(long)view*(DIRECTIONS+1)*pixels+pixel;
    out[index]=prediction+(ACCUMULATE?initial[index]:0.f);
    #pragma unroll
    for(int j=0;j<DIRECTIONS;++j) {
        index=((long)view*(DIRECTIONS+1)+j+1)*pixels+pixel;
        out[index]=columns[j]+(ACCUMULATE?initial[index]:0.f);
    }
}

extern "C" __global__ void region_adjoint(const float* coeff,const float* images,
        const float* initial,float* out,int views,int nx,int ny,int nz,
        int ox,int oy,int oz,int full_nu,int full_nv,int nu,int nv,int u0,int v0) {
    const int R=RUN,VB=32;
    __shared__ float cf[VB*14];
    long run=(long)blockIdx.x*blockDim.x+threadIdx.x;
    int runs_z=(nz+R-1)/R;
    bool valid=run<(long)nx*ny*runs_z;
    int ix=valid?run/((long)ny*runs_z):0,iy=valid?(run/runs_z)%ny:0;
    int iz0=valid?R*(int)(run%runs_z):0;
    float total[R]={};
    for(int first=0;first<views;first+=VB) {
        __syncthreads();
        for(int i=threadIdx.x;i<VB*14;i+=blockDim.x) cf[i]=first*14+i<views*14?coeff[first*14+i]:0.f;
        __syncthreads();
        for(int view=0;view<min(VB,views-first);++view) {
            const float* c=cf+view*14;
            DetectorRegion image={images+(long)(first+view)*nu*nv,nu,nv,u0,v0};
            bool unit=((c[2]==0.f && c[5]==0.f && fabsf(c[6])==1.f)
                ||(c[1]==0.f && c[6]==0.f && fabsf(c[2])==1.f));
            if(unit) gather_unit_rows<R>(c,image,ix+ox,iy+oy,iz0+oz,full_nu,full_nv,total);
            else gather_view<R>(c,image,ix+ox,iy+oy,iz0+oz,full_nu,full_nv,total);
        }
    }
    if(valid) {
        long base=((long)ix*ny+iy)*nz+iz0;
        #pragma unroll
        for(int j=0;j<R;++j) if(iz0+j<nz) out[base+j]=initial[base+j]+total[j];
    }
}
