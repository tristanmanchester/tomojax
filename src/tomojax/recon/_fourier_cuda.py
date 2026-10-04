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

extern "C" __global__ void polar(
    const float2* spectrum, const double* angle, const float* radial,
    const float2* phase, float2* out, long long count, int layers,
    int nviews, int nr, int nfft, float qr, float qi, const float* table,
    const float2* detector_phase, const signed char* flipped
) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count * layers) return;
    long long m = i % count, z = i / count;
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
            bool conjugate = view >= nviews;
            if (conjugate) view -= nviews;
            conjugate = conjugate != (flipped[view] != 0);
            float angular = da == 0 ? 1 - (a - ai) : a - ai;
            float2 row = make_float2(0, 0);
            #pragma unroll
            for (int j = 0; j < 6; j++) {
                int k = ri + j - 2;
                int reflected = k < 0 ? -k : (k > nfft / 2 ? nfft - k : k);
                float2 value = spectrum[(z * nviews + view) * nr + reflected];
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
    float2 p = phase[m];
    out[i] = make_float2(total.x * p.x - total.y * p.y, total.x * p.y + total.y * p.x);
}
"""


@functools.lru_cache(maxsize=1)
def interpolation_kernel():
    import cupy

    return cupy.RawKernel(_SOURCE, "polar", options=("--fmad=false",))
