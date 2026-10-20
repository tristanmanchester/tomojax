// Shared zero-padded Joseph samples: exact cell weights and their derivatives.
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
    return x[((long)ix * ny + iy) * nz + iz];
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
        value = db = dc = 0.f;
        #pragma unroll
        for (int b=0;b<4;b++) {
            float row=0.f, grad=0.f;
            #pragma unroll
            for (int c=0;c<4;c++) {
                float f=joseph_sample(x,a,k,ib+b-1,ic+c-1,nx,ny,nz,nb,nc);
                row += f*wc[c]; grad += f*gc[c];
            }
            value += row*wb[b]; db += row*gb[b]; dc += grad*wb[b];
        }
    }
}
