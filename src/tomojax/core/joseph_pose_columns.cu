// One physical ray traversal for its value and all active pose directions.
__device__ __forceinline__ void columns_clip_planes(
    float pixel, float slope, float offset, int side, int planes, int support,
    int& first, int& last) {
    // Bound the exact floating-point expression pixel + (slope*k + offset).
    // The expanded interpolation halo also retains one-sided derivatives at
    // support knots. Outward integer rounding keeps this a conservative skip,
    // not a changed ray-integration rule.
    double base=(double)pixel+offset;
    double error=4.*1.1920928955078125e-7*
        (fabs((double)pixel)+fabs((double)offset)+fabs((double)slope)*(planes-1)+1.);
    double low=-support-error, high=side+support-1.+error;
    if (slope==0.f) {
        if (base<low || base>high) last=first;
        return;
    }
    double lo=(low-base)/slope, hi=(high-base)/slope;
    if (lo>hi) { double temporary=lo; lo=hi; hi=temporary; }
    // Clamp before integer conversion: near-axial rays can have tiny slopes.
    int begin=(int)fmax(0.,fmin((double)planes,ceil(lo)-1.));
    int end=(int)fmax(0.,fmin((double)planes,floor(hi)+2.));
    first=max(first,begin); last=min(last,end);
}

extern "C" __global__ void pose_columns(
    const float* __restrict__ coeff, const float* __restrict__ tangent,
    const float* __restrict__ x, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, int views) {
    const int K = DIRECTIONS;
    int tiles_v=(nv+15)/16, tile=blockIdx.x/views, view=blockIdx.x%views;
    int u=(tile/tiles_v)*8+threadIdx.x/16;
    int v=(tile%tiles_v)*16+threadIdx.x%16;
    bool valid=u<nu && v<nv;
    const float* c=coeff+14*view;
    int a=(int)c[0];
    int na=a==0?nx:(a==1?ny:nz), nb=a==0?ny:(a==1?nz:nx), nc=a==0?nz:(a==1?nx:ny);
    float pb=c[1]*u+c[2]*v, pc=c[5]*u+c[6]*v;
    int first=0,last=na;
    if (na>64) {
        columns_clip_planes(pb,c[3],c[4],nb,na,CUBIC?2:1,first,last);
        columns_clip_planes(pc,c[7],c[8],nc,na,CUBIC?2:1,first,last);
        if (!valid || first>=last) { first=na; last=0; }
        // A tile samples the same plane together, preserving locality rather
        // than dephasing neighboring rays by their individual entry planes.
        int lane=threadIdx.x%32, warp=threadIdx.x/32;
        for (int delta=16;delta>0;delta/=2) {
            first=min(first,__shfl_down_sync(0xffffffff,first,delta));
            last=max(last,__shfl_down_sync(0xffffffff,last,delta));
        }
        __shared__ int starts[4],stops[4];
        if (lane==0) { starts[warp]=first; stops[warp]=last; }
        __syncthreads();
        first=min(min(starts[0],starts[1]),min(starts[2],starts[3]));
        last=max(max(stops[0],stops[1]),max(stops[2],stops[3]));
    }
    if (!valid) return;
    float dbase_b[K], dbase_c[K], slope_b[K], slope_c[K], dw[K], sums[K];
    #pragma unroll
    for (int j=0;j<K;j++) {
        const float* d=tangent+((long)view*K+j)*14;
        dbase_b[j]=d[1]*u+d[2]*v; dbase_c[j]=d[5]*u+d[6]*v;
        slope_b[j]=d[3]; slope_c[j]=d[7]; dw[j]=d[9]; sums[j]=0.f;
    }
    float prediction=0.f;
    for (int k=first;k<last;k++) {
        float qb=pb+(c[3]*k+c[4]), qc=pc+(c[7]*k+c[8]);
        float value, gb, gc;
        joseph_value_derivatives<CUBIC>(x,a,k,qb,qc,nx,ny,nz,nb,nc,value,gb,gc);
        prediction += value*c[9];
        #pragma unroll
        for (int j=0;j<K;j++) {
            const float* d=tangent+((long)view*K+j)*14;
            float db=dbase_b[j]+(slope_b[j]*k+d[4]);
            float dc=dbase_c[j]+(slope_c[j]*k+d[8]);
            sums[j] += (gb*db+gc*dc)*c[9]+value*dw[j];
        }
    }
    long pixel=(long)v*nu+u, pixels=(long)nv*nu;
    out[(long)view*(K+1)*pixels+pixel]=prediction;
    #pragma unroll
    for (int j=0;j<K;j++) out[((long)view*(K+1)+j+1)*pixels+pixel]=sums[j];
}
