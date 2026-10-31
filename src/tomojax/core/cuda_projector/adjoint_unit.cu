#include "joseph_sampling.cuh"
#include "joseph_gather.cuh"

extern "C" __global__ void joseph_gather(
    const float* __restrict__ coeff, const float* __restrict__ images,
    float* out, const float* initial, int nviews, int nx, int ny, int nz, int nu, int nv)
{
    const int R = RUN, VB = 32;
    __shared__ float cf[VB * 14];
    long run = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int runs_z = (nz + R - 1) / R;
    bool valid = run < (long)nx * ny * runs_z;
    int ix = 0, iy = 0, iz0 = 0;
    if (valid) FIRST_VOXEL(run, R, ix, iy, iz0);
    float total[R];
    #pragma unroll
    for (int j = 0; j < R; ++j) total[j] = 0.f;
    for (int first = 0; first < nviews; first += VB) {
        __syncthreads();
        for (int i = threadIdx.x; i < VB * 14; i += blockDim.x)
            cf[i] = first * 14 + i < nviews * 14 ? coeff[first * 14 + i] : 0.f;
        __syncthreads();
        int count = min(VB, nviews - first);
        for (int view = 0; view < count; ++view) {
            const float* c = cf + view * 14;
            const float* image = images + (long)(first + view) * nu * nv;
            bool unit=((c[2]==0.f && c[5]==0.f && fabsf(c[6])==1.f)
                ||(c[1]==0.f && c[6]==0.f && fabsf(c[2])==1.f));
            if(unit) {
            gather_unit_rows<R>(c, image, ix, iy, iz0, nu, nv, total);
            } else {
            gather_view<R>(c, image, ix, iy, iz0, nu, nv, total);
            }
        }
    }
    if (valid) {
        long base = VOXEL(long, ix, iy, iz0);
        #pragma unroll
        for (int j = 0; j < R; ++j)
            if (iz0 + j < nz) out[base + j * Z_STRIDE] = initial[base + j * Z_STRIDE] + total[j];
    }
}
