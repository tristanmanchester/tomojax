// One physical ray traversal for its value and all active pose directions.
extern "C" __global__ void pose_columns(
    const float* __restrict__ coeff, const float* __restrict__ tangent,
    const float* __restrict__ x, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, int views) {
    const int K = DIRECTIONS;
    int tiles_v=(nv+15)/16, tile=blockIdx.x/views, view=blockIdx.x%views;
    int u=(tile/tiles_v)*8+threadIdx.x/16;
    int v=(tile%tiles_v)*16+threadIdx.x%16;
    if (u>=nu || v>=nv) return;
    const float* c=coeff+14*view;
    int a=(int)c[0];
    int na=a==0?nx:(a==1?ny:nz), nb=a==0?ny:(a==1?nz:nx), nc=a==0?nz:(a==1?nx:ny);
    float pb=c[1]*u+c[2]*v, pc=c[5]*u+c[6]*v;
    float dbase_b[K], dbase_c[K], slope_b[K], slope_c[K], dw[K], sums[K];
    #pragma unroll
    for (int j=0;j<K;j++) {
        const float* d=tangent+((long)view*K+j)*14;
        dbase_b[j]=d[1]*u+d[2]*v; dbase_c[j]=d[5]*u+d[6]*v;
        slope_b[j]=d[3]; slope_c[j]=d[7]; dw[j]=d[9]; sums[j]=0.f;
    }
    float prediction=0.f;
    for (int k=0;k<na;k++) {
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
