// Exact zero-extended trilinear ray integrals and their matched transpose.
// Two Gauss nodes integrate the cubic density polynomial in each voxel cell.
#define RAY_INFINITY (__int_as_float(0x7f800000))

__device__ float ray_integral(const float* cf, const float* volume, const float* image,
    float* output, int view, int ray, int nx, int ny, int nz, int nu, int nv) {
    int u = ray % nu, v = ray / nu;
    float q[3], d[3], safe[3], crossing[3], period[3];
    int shape[3] = {nx, ny, nz};
    float entry = -RAY_INFINITY, end_ray = RAY_INFINITY;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        q[a] = (cf[a] + cf[3+a] * (float)u) + cf[6+a] * (float)v;
        d[a] = cf[9+a];
        bool moving = fabsf(d[a]) > 1.e-12f;
        safe[a] = moving ? d[a] : 1.f;
        float lo = (-1.f - q[a]) / safe[a], hi = ((float)shape[a] - q[a]) / safe[a];
        bool inside = q[a] >= -1.f && q[a] <= (float)shape[a];
        entry = fmaxf(entry, moving ? fminf(lo,hi) : (inside ? -RAY_INFINITY : RAY_INFINITY));
        end_ray = fminf(end_ray, moving ? fmaxf(lo,hi) : (inside ? RAY_INFINITY : -RAY_INFINITY));
    }
    if (ray >= nu * nv || !isfinite(entry) || !isfinite(end_ray) || end_ray <= entry) return 0.f;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        float start = q[a] + d[a] * entry;
        float plane = d[a] > 0.f ? floorf(start) + 1.f : ceilf(start) - 1.f;
        bool moving = fabsf(d[a]) > 1.e-12f;
        crossing[a] = moving ? (plane - q[a]) / safe[a] : RAY_INFINITY;
        period[a] = moving ? 1.f / fabsf(safe[a]) : RAY_INFINITY;
    }
    float ray_value = MODE == 1 ? image[(long)view * nu * nv + ray] : 0.f;
    float current = entry, sum = 0.f;
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
        float density = 0.f;
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
                    long position = ((long)z * ny + y) * nx + x;
                    if (MODE == 1) {
                        if (valid && weight != 0.f) atomicAdd(output+position, weight*ray_value);
                    } else {
                        density += weight * (valid ? volume[position] : 0.f);
                    }
                }
            }
        }
        sum += density;
        #pragma unroll
        for (int a = 0; a < 3; ++a)
            if (crossing[a] <= end) crossing[a] += period[a];
        current = end;
    }
    return sum;
}

extern "C" __global__ void integrate(const float* coeff, const float* volume,
    const float* images, float* out, int nx, int ny, int nz, int nu, int nv, int nviews) {
    int view = blockIdx.x % nviews, ray = (blockIdx.x / nviews) * blockDim.x + threadIdx.x;
    float sum = ray_integral(coeff+12*view, volume, images, out, view, ray, nx,ny,nz,nu,nv);
    if (MODE == 0 && ray < nu*nv) out[(long)view*nu*nv+ray] = sum;
}
