// Exact zero-extended trilinear ray integrals and their matched transpose.
// Two Gauss nodes integrate the cubic density polynomial in each voxel cell.
#include <math_constants.h>

__device__ void ray_integral(const float* cf, const float* volume, const float* image,
    float* output, int view, int ray, int nx, int ny, int nz, int nu, int nv,
    float (&sums)[7]) {
    int u = ray % nu, v = ray / nu;
    float q[3], d[3], safe[3], crossing[3], period[3];
    int shape[3] = {nx, ny, nz};
    float entry = -CUDART_INF_F, end_ray = CUDART_INF_F;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        q[a] = (cf[a] + cf[3+a] * (float)u) + cf[6+a] * (float)v;
        d[a] = cf[9+a];
        bool moving = fabsf(d[a]) > 1.e-12f;
        safe[a] = moving ? d[a] : 1.f;
        float lo = (-1.f - q[a]) / safe[a], hi = ((float)shape[a] - q[a]) / safe[a];
        bool inside = q[a] >= -1.f && q[a] <= (float)shape[a];
        entry = fmaxf(entry, moving ? fminf(lo,hi) : (inside ? -CUDART_INF_F : CUDART_INF_F));
        end_ray = fminf(end_ray, moving ? fmaxf(lo,hi) : (inside ? CUDART_INF_F : -CUDART_INF_F));
    }
    if (ray >= nu * nv || !isfinite(entry) || !isfinite(end_ray) || end_ray <= entry) return;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        float start = q[a] + d[a] * entry;
        float plane = d[a] > 0.f ? floorf(start) + 1.f : ceilf(start) - 1.f;
        bool moving = fabsf(d[a]) > 1.e-12f;
        crossing[a] = moving ? (plane - q[a]) / safe[a] : CUDART_INF_F;
        period[a] = moving ? 1.f / fabsf(safe[a]) : CUDART_INF_F;
    }
    float ray_value = MODE == 1 ? image[(long)view * nu * nv + ray] : 0.f;
    float current = entry;
    for (int it = 0; it < nx + ny + nz + 9 && current < end_ray; ++it) {
        float end = fminf(end_ray, fminf(fminf(crossing[0], crossing[1]), crossing[2]));
        float length = fmaxf(end-current, 0.f), midpoint = current + .5f * length;
        float times[2] = {current + 0.21132486540518713f * length,
                          current + 0.7886751345948129f * length};
        int index[3]; float frac[2][3];
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            index[a] = (int)floorf(q[a] + d[a] * midpoint);
            #pragma unroll
            for (int node = 0; node < 2; ++node)
                frac[node][a] = fminf(1.f, fmaxf(0.f, (q[a] + d[a] * times[node]) - (float)index[a]));
        }
        float density = 0.f, gradient[3] = {}, moment[3] = {};
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            #pragma unroll
            for (int dy = 0; dy < 2; ++dy) {
                #pragma unroll
                for (int dz = 0; dz < 2; ++dz) {
                    int offsets[3] = {dx, dy, dz};
                    int x = index[0]+dx, y = index[1]+dy, z = index[2]+dz;
                    bool valid = length > 0.f && (unsigned)x < (unsigned)nx
                        && (unsigned)y < (unsigned)ny && (unsigned)z < (unsigned)nz;
                    float w[2][3];
                    #pragma unroll
                    for (int node = 0; node < 2; ++node) {
                        #pragma unroll
                        for (int a = 0; a < 3; ++a)
                            w[node][a] = offsets[a] ? frac[node][a] : 1.f-frac[node][a];
                    }
                    float weight = .5f * length * ((w[0][0]*w[0][1])*w[0][2]
                                                  + (w[1][0]*w[1][1])*w[1][2]);
                    long position = ((long)x * ny + y) * nz + z;
                    if (MODE == 1) {
                        if (valid && weight != 0.f) atomicAdd(output+position, weight*ray_value);
                    } else {
                        float sample = valid ? volume[position] : 0.f;
                        density += weight * sample;
                        if (MODE >= 2) {
                            #pragma unroll
                            for (int a = 0; a < 3; ++a) {
                                float dw0 = ((float)(2*offsets[a]-1)*w[0][(a+1)%3])*w[0][(a+2)%3];
                                float dw1 = ((float)(2*offsets[a]-1)*w[1][(a+1)%3])*w[1][(a+2)%3];
                                gradient[a] += (.5f*length*(dw0+dw1))*sample;
                                moment[a] += (.5f*length*(times[0]*dw0+times[1]*dw1))*sample;
                            }
                        }
                    }
                }
            }
        }
        sums[0] += density;
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            if (MODE >= 2) { sums[1+a] += gradient[a]; sums[4+a] += moment[a]; }
            if (crossing[a] <= end) crossing[a] += period[a];
        }
        current = end;
    }
}

extern "C" __global__ void integrate(const float* coeff, const float* volume,
    const float* images, float* out, int nx, int ny, int nz, int nu, int nv, int nviews) {
    int view = blockIdx.x % nviews, ray = (blockIdx.x / nviews) * blockDim.x + threadIdx.x;
    float sums[7] = {};
    ray_integral(coeff+12*view, volume, images, out, view, ray, nx,ny,nz,nu,nv,sums);
    if (MODE != 1 && ray < nu*nv) {
        #pragma unroll
        for (int i = 0; i < (MODE == 2 ? 7 : 1); ++i)
            out[((long)i*nviews+view)*nu*nv+ray] = sums[i];
    }
}

extern "C" __global__ void normals(const float* coeff, const float* volume,
    const float* directions, const float* target, const float* weights,
    float* partials, float* residuals, int nx, int ny, int nz, int nu, int nv, int tiles, int nviews) {
    int view = blockIdx.x % nviews, tile = blockIdx.x / nviews;
    int ray = tile * blockDim.x + threadIdx.x;
    float sums[7] = {};
    ray_integral(coeff+12*view,volume,target,partials,view,ray,nx,ny,nz,nu,nv,sums);
    long location = (long)view*nu*nv+ray;
    bool valid = ray < nu*nv;
    float weight = valid ? weights[location] : 0.f;
    float residual = weight*(sums[0]-(valid ? target[location] : 0.f));
    float columns[5], values[21];
    int u = ray%nu, v = ray/nu;
    #pragma unroll
    for (int dof=0; dof<5; ++dof) {
        const float* dc = directions+(view*5+dof)*12;
        float col = 0.f;
        #pragma unroll
        for (int a=0; a<3; ++a)
            col = (col+sums[1+a]*((dc[a]+(float)u*dc[3+a])+(float)v*dc[6+a]))
                + sums[4+a]*dc[9+a];
        columns[dof] = weight*col;
        values[1+dof] = columns[dof]*residual;
    }
    values[0] = .5f*residual*residual;
    int offset=6;
    #pragma unroll
    for (int i=0; i<5; ++i) {
        #pragma unroll
        for (int j=i; j<5; ++j) values[offset++] = columns[i]*columns[j];
    }
    if (valid) residuals[location] = weight*residual;
    __shared__ float warp_sums[4][21];
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    #pragma unroll
    for (int i=0; i<21; ++i) {
        float value=values[i];
        for (int shift=16; shift; shift/=2) value+=__shfl_down_sync(0xffffffff,value,shift);
        if (lane==0) warp_sums[warp][i]=value;
    }
    __syncthreads();
    if (threadIdx.x<21) {
        int i=threadIdx.x;
        partials[((long)view*tiles+tile)*21+i] =
            (warp_sums[0][i]+warp_sums[1][i])+(warp_sums[2][i]+warp_sums[3][i]);
    }
}
