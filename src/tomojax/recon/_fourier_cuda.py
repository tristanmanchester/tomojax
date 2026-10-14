"""CUDA polar-to-Cartesian interpolation for the Fourier-slice reconstruction."""

from __future__ import annotations

import functools

_SOURCE = r"""
__device__ float radial_weight(float distance, const float* table) {
    distance = fabsf(distance);
    if (distance >= 3) return 0;
    float coordinate = distance * (16384.0f / 3);
    int index = min((int)coordinate, 16383);
    float fraction = coordinate - index;
    return table[index] * (1 - fraction) + table[index + 1] * fraction;
}

__device__ __forceinline__ float2 polar_value(
    const float2* spectrum, const double* angle, const float* radial,
    const float2* phase, const float2* detector_phase, long long m, long long z,
    int nviews, int nsource, const int* groups, int nr, int nfft,
    float qr, float qi, const float* table,
    const signed char* flipped
) {
    // Keep the angular fraction precise even with hundreds of views. Casting
    // the full view coordinate to FP32 would lose bits before subtracting ai.
    double a = angle[m];
    float r = radial[m];
    float2 total = make_float2(0, 0);
    if (r <= nfft * 0.5f) {
        int ai = (int)floor(a), ri = (int)floorf(r);
        float weights[6];
        #pragma unroll
        for (int j = 0; j < 6; j++) weights[j] = radial_weight(r - (ri + j - 2), table);
        float2 dp = detector_phase[m];
        #pragma unroll
        for (int da = 0; da <= 1; da++) {
            int view = ai + da;
            if (view >= 2 * nviews) view -= 2 * nviews;
            bool opposing = view >= nviews;
            if (opposing) view -= nviews;
            float angular = da == 0 ? 1 - (a - ai) : a - ai;
            #ifdef GROUP_REPEATS
            int first = groups[view], stop = groups[view + 1];
            angular /= stop - first;
            #else
            int first = view, stop = view + 1;
            #endif
            for (int source = first; source < stop; source++) {
                bool conjugate = opposing != (flipped[source] != 0);
                float2 row = make_float2(0, 0);
                #pragma unroll
                for (int j = 0; j < 6; j++) {
                    int k = ri + j - 2;
                    int reflected = k < 0 ? -k : (k > nfft / 2 ? nfft - k : k);
                    float2 value = spectrum[(z * nsource + source) * nr + reflected];
                    if (k < 0 || k > nfft / 2) value.y = -value.y;
                    // The centered transform is quasi-periodic for an even detector.
                    if (k > nfft / 2) {
                        float real = value.x * qr - value.y * qi;
                        value.y = value.x * qi + value.y * qr;
                        value.x = real;
                    }
                    row.x += weights[j] * value.x;
                    row.y += weights[j] * value.y;
                }
                float real = row.x * dp.x - row.y * dp.y;
                float imag = row.x * dp.y + row.y * dp.x;
                total.x += angular * real;
                total.y += angular * (conjugate ? -imag : imag);
            }
        }
    }
    float2 p = phase[m];
    return make_float2(total.x * p.x - total.y * p.y, total.x * p.y + total.y * p.x);
}

extern "C" __global__ void polar(
    const float2* spectrum, const double* angle, const float* radial,
    const float2* phase, float2* out, long long count, int layers,
    int nviews, int nsource, const int* groups, int nr, int nfft,
    float qr, float qi, const float* table,
    const float2* detector_phase, const signed char* flipped,
    const double* row_mass, const double* view_weights, double du, int replace_dc
    #ifdef FUSE_X_BOUNDARY
    , const double* mirror_angle, const float* mirror_radial,
    const float2* mirror_phase, const float2* mirror_detector,
    long long row_offset, int stop
    #endif
) {
    long long m = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long z = blockIdx.y + (long long)blockIdx.z * gridDim.y;
    if (z >= layers) return;
    long long i = z * count + m;
    // The unweighted projection mass replaces DC, not the deapodized FFT's DC.
    // All lanes participate before the bounds guard, even for fewer than 32 sites.
    if (replace_dc && blockIdx.x == 0 && threadIdx.x < 32) {
        double mass = 0;
        for (int view = threadIdx.x; view < nsource; view += 32)
            mass += row_mass[z * nsource + view] * view_weights[view];
        for (int shift = 16; shift > 0; shift >>= 1)
            mass += __shfl_down_sync(0xffffffff, mass, shift);
        if (threadIdx.x == 0) {
            out[i] = make_float2((float)(mass * du), 0.0f);
            return;
        }
    }
    if (m >= count) return;
    #ifdef FUSE_X_BOUNDARY
    // The x Nyquist bin also represents -kx. Average the missing opposite-y
    // value before storing it, preserving real(ifft2)'s Hermitian projection.
    // One non-unrolled loop shares the interpolation body between both samples.
    long long y = m - row_offset;
    int samples = y > 0 && y < stop ? 2 : 1;
    float2 value = make_float2(0, 0);
    #pragma unroll 1
    for (int pass = 0; pass < samples; pass++) {
        bool opposite = pass != 0;
        float2 current = polar_value(
            spectrum, opposite ? mirror_angle : angle, opposite ? mirror_radial : radial,
            opposite ? mirror_phase : phase, opposite ? mirror_detector : detector_phase,
            opposite ? y - 1 : m, z, nviews, nsource, groups, nr, nfft, qr, qi, table, flipped
        );
        if (pass == 0) value = current;
        else value = make_float2(0.5f * (value.x + current.x), 0.5f * (value.y - current.y));
    }
    out[i] = value;
    #else
    out[i] = polar_value(
        spectrum, angle, radial, phase, detector_phase, m, z,
        nviews, nsource, groups, nr, nfft, qr, qi, table, flipped
    );
    #endif

}
"""

_MASS_SOURCE = r"""
extern "C" __global__ void mass_deapod(
    float* raw, const float* deapod, double* row_mass, int nu, long long nrows
) {
    long long row = (long long)blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    double total = 0.0;
    if (row < nrows) {
        for (int u = lane; u < nu; u += 32) {
            long long index = row * nu + u;
            float value = raw[index];
            total += (double)value;
            raw[index] = value * deapod[u];
        }
    }
    for (int shift = 16; shift > 0; shift >>= 1)
        total += __shfl_down_sync(0xffffffff, total, shift);
    if (lane == 0 && row < nrows) row_mass[row] = total;
}
"""


@functools.lru_cache(maxsize=4)
def interpolation_kernel(*, x_boundary=False, repeats=False):
    import cupy

    options = (
        ("--fmad=false",)
        + (("-DFUSE_X_BOUNDARY",) if x_boundary else ())
        + (("-DGROUP_REPEATS",) if repeats else ())
    )
    return cupy.RawKernel(_SOURCE, "polar", options=options)


@functools.lru_cache(maxsize=1)
def mass_deapod_kernel():
    """Sum each measured profile in FP64 while deapodizing its owned FP32 copy."""
    import cupy

    return cupy.RawKernel(_MASS_SOURCE, "mass_deapod", options=("--fmad=false",))
