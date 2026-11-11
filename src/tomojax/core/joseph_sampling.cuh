// Shared zero-padded Joseph samples: exact cell weights and their derivatives.

// Volume storage. With Z_FAST, (nx, ny, nz) C-order with z contiguous: the solver-private
// layout. Otherwise (nz, ny, nx) C-order with x contiguous: the public layout.
#if Z_FAST
#define VOXEL(T, ix, iy, iz) (((T)(ix) * ny + (iy)) * nz + (iz))
#define Z_STRIDE 1L
#else
#define VOXEL(T, ix, iy, iz) (((T)(iz) * ny + (iy)) * nx + (ix))
#define Z_STRIDE ((long)nx * ny)
#endif
// A transpose thread owns a run of R voxels along z. Neighbouring threads take neighbouring
// runs of one column in either storage: their gathers read neighbouring detector rows, which
// matters more than coalescing the single final store.
#define FIRST_VOXEL(run, R, ix, iy, iz0) { int runs_z_ = (nz + (R) - 1) / (R); \
    ix = (int)((run) / ((long)ny * runs_z_)); iy = (int)(((run) / runs_z_) % ny); \
    iz0 = (R) * (int)((run) % runs_z_); }
__device__ __forceinline__ float joseph_cubic_weight(float distance) {
    float r = fabsf(distance);
    if (r < 1.f) return 1.f + r*r * (1.5f*r - 2.5f);
    if (r < 2.f) return -.5f * (r-1.f) * ((r-2.f) * (r-2.f));
    return 0.f;
}

__device__ __forceinline__ float joseph_sample(
    const float* x, int a, int k, int b, int c,
    int nx, int ny, int nz, int nb, int nc) {
    if (b < 0 || b >= nb || c < 0 || c >= nc) return 0.f;
    int ix = a == 0 ? k : (a == 1 ? c : b);
    int iy = a == 0 ? b : (a == 1 ? k : c);
    int iz = a == 0 ? c : (a == 1 ? b : k);
    return x[VOXEL(long, ix, iy, iz)];
}

__device__ __forceinline__ void joseph_cubic_weights(float t, float* w, float* d) {
    w[0] = -.5f * t * ((1-t) * (1-t));
    w[1] = 1 + t*t * (1.5f*t - 2.5f);
    w[2] = t * (.5f + t * (2 - 1.5f*t));
    w[3] = -.5f * t*t * (1-t);
    d[0] = -.5f + t * (2 - 1.5f*t);
    d[1] = t * (4.5f*t - 5);
    d[2] = .5f + t * (4 - 4.5f*t);
    d[3] = t * (1.5f*t - 1);
}

// The 4x4 cubic taps around (ib, ic) on plane k, zero outside the volume.
template<typename Volume>
__device__ __forceinline__ void joseph_cubic_taps(
    Volume x, int a, int k, int ib, int ic,
    int nx, int ny, int nz, int nb, int nc, float (&f)[4][4]) {
    #pragma unroll
    for (int b=0;b<4;b++)
        #pragma unroll
        for (int c=0;c<4;c++) f[b][c] = joseph_sample(x,a,k,ib+b-1,ic+c-1,nx,ny,nz,nb,nc);
}
// Inside the volume, which nearly every sample is, all 16 taps load from one
// base address without per-tap tests.
__device__ __forceinline__ void joseph_cubic_taps(
    const float* x, int a, int k, int ib, int ic,
    int nx, int ny, int nz, int nb, int nc, float (&f)[4][4]) {
    if (nb > 3 && nc > 3 && (unsigned)(ib-1) < (unsigned)(nb-3)
        && (unsigned)(ic-1) < (unsigned)(nc-3)) {
        long sx = VOXEL(long, 1, 0, 0), sy = VOXEL(long, 0, 1, 0), sz = VOXEL(long, 0, 0, 1);
        long sb = a == 0 ? sy : (a == 1 ? sz : sx), sc = a == 0 ? sz : (a == 1 ? sx : sy);
        int b0 = ib - 1, c0 = ic - 1;
        const float* p = x + VOXEL(long, a == 0 ? k : (a == 1 ? c0 : b0),
                                   a == 0 ? b0 : (a == 1 ? k : c0), a == 0 ? c0 : (a == 1 ? b0 : k));
        #pragma unroll
        for (int b=0;b<4;b++)
            #pragma unroll
            for (int c=0;c<4;c++) f[b][c] = p[b * sb + c * sc];
        return;
    }
    joseph_cubic_taps<const float*>(x,a,k,ib,ic,nx,ny,nz,nb,nc,f);
}

template<bool Cubic, bool Absolute = false, typename Volume>
__device__ __forceinline__ void joseph_value_derivatives(
    Volume x, int a, int k, float qb, float qc,
    int nx, int ny, int nz, int nb, int nc,
    float& value, float& db, float& dc) {
    int ib = (int)floorf(qb), ic = (int)floorf(qc);
    float tb = qb - ib, tc = qc - ic;
    if (!Cubic) {
        float f00 = joseph_sample(x,a,k,ib,ic,nx,ny,nz,nb,nc);
        float f01 = joseph_sample(x,a,k,ib,ic+1,nx,ny,nz,nb,nc);
        float f10 = joseph_sample(x,a,k,ib+1,ic,nx,ny,nz,nb,nc);
        float f11 = joseph_sample(x,a,k,ib+1,ic+1,nx,ny,nz,nb,nc);
        float low = f00*(1-tc)+f01*tc, high = f10*(1-tc)+f11*tc;
        value = low*(1-tb)+high*tb;
        db = high-low;
        dc = (f01-f00)*(1-tb)+(f11-f10)*tb;
    } else {
        float wb[4], wc[4], gb[4], gc[4];
        joseph_cubic_weights(tb,wb,gb); joseph_cubic_weights(tc,wc,gc);
        if (Absolute) {
            #pragma unroll
            for (int i=0;i<4;i++) {
                gb[i] *= wb[i] < 0.f ? -1.f : 1.f;
                gc[i] *= wc[i] < 0.f ? -1.f : 1.f;
                wb[i] = fabsf(wb[i]); wc[i] = fabsf(wc[i]);
            }
        }
        float taps[4][4];
        joseph_cubic_taps(x,a,k,ib,ic,nx,ny,nz,nb,nc,taps);
        value = db = dc = 0.f;
        #pragma unroll
        for (int b=0;b<4;b++) {
            float row=0.f, grad=0.f;
            #pragma unroll
            for (int c=0;c<4;c++) {
                float f=taps[b][c];
                row += f*wc[c]; grad += f*gc[c];
            }
            value += row*wb[b]; db += row*gb[b]; dc += grad*wb[b];
        }
    }
}
