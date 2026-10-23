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
    return x.data[((long)iz*x.ny+iy)*x.nx+ix];
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

// Round in the forward kernel's order. Sign-selected endpoints enclose every
// sample in the detector/major-plane rectangle, including at rounding knots.
__device__ __forceinline__ void region_coordinate_bounds(
        float u,float v,float k,float offset,int u0,int u1,int v0,int v1,int k0,int k1,
        float& lower,float& upper) {
    float ul=(float)(u<0.f?u1:u0),uh=(float)(u<0.f?u0:u1);
    float vl=(float)(v<0.f?v1:v0),vh=(float)(v<0.f?v0:v1);
    float kl=(float)(k<0.f?k1:k0),kh=(float)(k<0.f?k0:k1);
    lower=__fadd_rn(__fadd_rn(__fmul_rn(u,ul),__fmul_rn(v,vl)),
                    __fadd_rn(__fmul_rn(k,kl),offset));
    upper=__fadd_rn(__fadd_rn(__fmul_rn(u,uh),__fmul_rn(v,vh)),
                    __fadd_rn(__fmul_rn(k,kh),offset));
}

__device__ __forceinline__ bool region_block_visible(
        const float* c,int x0,int x1,int y0,int y1,int z0,int z1,
        int u0,int u1,int v0,int v1) {
    int a=(int)c[0];
    int k0=a==0?x0:(a==1?y0:z0),k1=a==0?x1:(a==1?y1:z1);
    int b0=a==0?y0:(a==1?z0:x0),b1=a==0?y1:(a==1?z1:x1);
    int t0=a==0?z0:(a==1?x0:y0),t1=a==0?z1:(a==1?x1:y1);
    float blo,bhi,clo,chi;
    region_coordinate_bounds(c[1],c[2],c[3],c[4],u0,u1,v0,v1,k0,k1,blo,bhi);
    region_coordinate_bounds(c[5],c[6],c[7],c[8],u0,u1,v0,v1,k0,k1,clo,chi);
    const float radius=CUBIC?2.f:1.f;
    return !(blo>(float)b1+radius || bhi<(float)b0-radius ||
             clo>(float)t1+radius || chi<(float)t0-radius);
}

extern "C" __global__ void region_adjoint(const float* coeff,const float* images,
        const float* initial,float* out,int views,int nx,int ny,int nz,
        int ox,int oy,int oz,int full_nu,int full_nv,int nu,int nv,int u0,int v0) {
    const int R=RUN,VB=32;
    __shared__ float cf[VB*14];
    __shared__ bool visible[VB];
    long run=(long)blockIdx.x*blockDim.x+threadIdx.x;
    int runs_z=(nz+R-1)/R;
    bool valid=run<(long)nx*ny*runs_z;
    int ix=valid?run%nx:0,iy=valid?(run/nx)%ny:0;
    int iz0=valid?R*(int)(run/((long)nx*ny)):0;
    float total[R]={};
    for(int first=0;first<views;first+=VB) {
        __syncthreads();
        for(int i=threadIdx.x;i<VB*14;i+=blockDim.x) cf[i]=first*14+i<views*14?coeff[first*14+i]:0.f;
        __syncthreads();
        if(threadIdx.x<min(VB,views-first)) {
            long begin=(long)blockIdx.x*blockDim.x;
            long end=min(begin+blockDim.x-1,(long)nx*ny*runs_z-1);
            int rz0=begin/((long)nx*ny),rz1=end/((long)nx*ny);
            int z0=R*rz0,z1=min(nz-1,R*rz1+R-1);
            int y0=rz0==rz1?(begin/nx)%ny:0;
            int y1=rz0==rz1?(end/nx)%ny:ny-1;
            int x0=begin/nx==end/nx?begin%nx:0;
            int x1=begin/nx==end/nx?end%nx:nx-1;
            visible[threadIdx.x]=region_block_visible(cf+threadIdx.x*14,
                x0+ox,x1+ox,y0+oy,y1+oy,z0+oz,z1+oz,u0,u0+nu-1,v0,v0+nv-1);
        }
        __syncthreads();
        for(int view=0;view<min(VB,views-first);++view) {
            if(!visible[view]) continue;
            const float* c=cf+view*14;
            DetectorRegion image={images+(long)(first+view)*nu*nv,nu,nv,u0,v0};
            bool unit=((c[2]==0.f && c[5]==0.f && fabsf(c[6])==1.f)
                ||(c[1]==0.f && c[6]==0.f && fabsf(c[2])==1.f));
            if(unit) gather_unit_rows<R>(c,image,ix+ox,iy+oy,iz0+oz,full_nu,full_nv,total);
            else gather_view<R>(c,image,ix+ox,iy+oy,iz0+oz,full_nu,full_nv,total);
        }
    }
    if(valid) {
        long base=((long)iz0*ny+iy)*nx+ix;
        #pragma unroll
        for(int j=0;j<R;++j) if(iz0+j<nz) out[base+(long)j*nx*ny]=initial[base+(long)j*nx*ny]+total[j];
    }
}
