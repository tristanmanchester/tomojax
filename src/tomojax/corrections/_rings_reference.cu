
/* LOCAL_MEDIAN */
__device__ float block_sum(float x, float *work) {
    int t = threadIdx.x, lane = t % 32, warp = t / 32;
    for (int d = 16; d; d /= 2)
        x += __shfl_down_sync(0xffffffff, x, d);
    if (lane == 0)
        work[warp] = x;
    __syncthreads();
    x = t < B / 32 ? work[t] : 0.0f;
    if (warp == 0) {
        for (int d = 16; d; d /= 2)
            x += __shfl_down_sync(0xffffffff, x, d);
        if (lane == 0)
            work[0] = x;
    }
    __syncthreads();
    float result = work[0];
    __syncthreads();
    return result;
}

// Nonnegative IEEE float bits sort in numerical order. Four byte histograms
// select the upper median without a Count-sized shared array. Only lane zero
// returns the median; all other lanes return zero after the last barrier.
template <int Count> __device__ float positive_median(const unsigned (&keys)[(Count + B - 1) / B]) {
    __shared__ unsigned hist[256], selected, target, counts[B / 32], maxima[B / 32];
    int t = threadIdx.x, lane = t % 32, warp = t / 32;
    if (t == 0) {
        selected = 0;
        target = Count / 2;
    }
    unsigned mask = 0;
#pragma unroll
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (t < 256)
            hist[t] = 0;
        __syncthreads();
        unsigned prefix = selected;
#pragma unroll
        for (int c = 0; c < (Count + B - 1) / B; c++) {
            bool valid = t + c * B < Count && (keys[c] & mask) == prefix;
            unsigned active = __ballot_sync(0xffffffff, valid);
            if (valid) {
                unsigned bucket = (keys[c] >> shift) & 255;
#if __CUDA_ARCH__ >= 700
                unsigned peers = __match_any_sync(active, bucket);
                if (lane == __ffs(peers) - 1)
                    atomicAdd(hist + bucket, (unsigned)__popc(peers));
#else
                atomicAdd(hist + bucket, 1u);
#endif
            }
        }
        __syncthreads();
        if (t == 0) {
            unsigned count = 0;
            for (unsigned bucket = 0; bucket < 256; bucket++) {
                unsigned next = count + hist[bucket];
                if (next > target) {
                    selected |= bucket << shift;
                    target -= count;
                    break;
                }
                count = next;
            }
        }
        __syncthreads();
        mask |= 255u << shift;
    }
    // For even counts, the lower median is either the same tied value or the
    // largest strictly smaller key when exactly half of the keys precede upper.
    unsigned count = 0, lower = 0, upper = selected;
#pragma unroll
    for (int c = 0; c < (Count + B - 1) / B; c++)
        if (t + c * B < Count && keys[c] < upper) {
            count++;
            lower = max(lower, keys[c]);
        }
    for (int d = 16; d; d /= 2) {
        count += __shfl_down_sync(0xffffffff, count, d);
        lower = max(lower, __shfl_down_sync(0xffffffff, lower, d));
    }
    if (lane == 0) {
        counts[warp] = count;
        maxima[warp] = lower;
    }
    __syncthreads();
    if (t != 0)
        return 0.0f;
    unsigned total = 0, largest = 0;
    for (int j = 0; j < B / 32; j++) {
        total += counts[j];
        largest = max(largest, maxima[j]);
    }
    float hi = __uint_as_float(upper), lo = total == Count / 2 ? __uint_as_float(largest) : hi;
    return Count % 2 ? hi : 0.5f * (lo + hi);
}
extern "C" __global__ void residual_mad(const float *values, float *residual, float *sigma) {
    const float *v = values + blockIdx.x * N;
    unsigned keys[C];
    int row = blockIdx.x, t = threadIdx.x;

#pragma unroll
    for (int c = 0; c < C; c++) {
        int x = t + c * B;
        if (x < N) { /* NARROW */
            float r = v[x] - fit;
            residual[row * N + x] = r;
            keys[c] = __float_as_uint(fabsf(r));
        } else
            keys[c] = 0;
    }
    float median = positive_median<N>(keys);
    if (t == 0)
        sigma[row] = 1.4826f * median;
}

extern "C" __global__ void sigma_floor(const float *sigma, float *floor) {
    int t = threadIdx.x;
    unsigned keys[(V + B - 1) / B];
#pragma unroll
    for (int c = 0; c < (V + B - 1) / B; c++) {
        int x = t + c * B;
        keys[c] = x < V ? __float_as_uint(sigma[x]) : 0;
    }
    float median = positive_median<V>(keys);
    if (t == 0)
        *floor = 1e-3f * median + 1e-12f;
}

// The halo serves every support, including repeated detector-edge samples.
// The excluded centre and its immediate neighbours never enter a fit.
extern "C" __global__ void weighted_tiles(const float *values, const float *residual,
                                          const float *sigma, const float *floor, float *fits,
                                          float *r) {
    __shared__ float v[T + 60], w[T + 60];
    int row = blockIdx.y, t = threadIdx.x, x = blockIdx.x * T + t;
    float scale = 4.685f * fmaxf(sigma[row], *floor);
    for (int k = t; k < T + 60; k += T) {
        int at = max(0, min(N - 1, int(blockIdx.x) * T + k - 30));
        v[k] = values[row * N + at];
        float u = residual[row * N + at] / scale, q = 1.0f - u * u;
        w[k] = 1e-3f + (fabsf(u) < 1.0f ? q * q : 0.0f);
    }
    __syncthreads();
    if (x >= N)
        return;

    // Derive a weighted polynomial basis on the narrowest support. Its full
    // Gram matrix (including cross terms) is retained for every wider fit.
    float center = 0.5f * (v[t + 30 - 2] + v[t + 30 + 2]), s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
    for (int j = 2; j <= 6; j++) {
        float wl = w[t + 30 - j], wr = w[t + 30 + j], z = float(j) / 6.0f;
        s0 += wl + wr;
        s1 += z * (wr - wl);
        s2 += z * z * (wl + wr);
        s3 += z * z * z * (wr - wl);
    }
    float inv0 = 1.0f / s0, mu = s1 * inv0, second = s2 * inv0, variance = second - mu * mu,
          skew = (s3 * inv0 - 3.0f * mu * second + 2.0f * mu * mu * mu) / variance;
    float g00 = 0, g01 = 0, g02 = 0, g11 = 0, g12 = 0, g22 = 0, b0 = 0, b1 = 0, b2 = 0;
#pragma unroll
    for (int j = 2; j <= 30; j++) {
        float xl = -float(j) / 6.0f - mu, xr = float(j) / 6.0f - mu;
        float ql = xl * xl - skew * xl - variance, qr = xr * xr - skew * xr - variance;
        int il = t + 30 - j, ir = t + 30 + j;
        float wl = w[il], wr = w[ir];
        float yl = wl * (v[il] - center), yr = wr * (v[ir] - center);
        g00 += wl + wr;
        g01 += wl * xl + wr * xr;
        g02 += wl * ql + wr * qr;
        g11 += wl * xl * xl + wr * xr * xr;
        g12 += wl * xl * ql + wr * xr * qr;
        g22 += wl * ql * ql + wr * qr * qr;
        b0 += yl + yr;
        b1 += yl * xl + yr * xr;
        b2 += yl * ql + yr * qr;
        if (j == 6 || j == 9 || j == 14 || j == 20 || j == 30) {
            // Rescale columns to this support, eliminate the intercept, then solve
            // the centred two-by-two system and evaluate the polynomial at x=0.
            float scale = 6.0f / float(j), scale2 = scale * scale, a1 = g01 * scale,
                  a2 = g02 * scale2;
            float a11 = g11 * scale2, a12 = g12 * scale2 * scale, a22 = g22 * scale2 * scale2;
            float inv00 = 1.0f / g00, mean = b0 * inv00, m1 = a1 * inv00, m2 = a2 * inv00,
                  c11 = a11 - a1 * m1, c12 = a12 - a1 * m2, c22 = a22 - a2 * m2;
            float r1 = b1 * scale - a1 * mean, r2 = b2 * scale2 - a2 * mean, inv11 = 1.0f / c11,
                  l = c12 * inv11, p2 = (r2 - l * r1) / (c22 - l * c12),
                  p1 = (r1 - c12 * p2) * inv11;
            float p0 = mean - m1 * p1 - m2 * p2;
            int k = j == 30 ? 0 : j == 20 ? 1 : j == 14 ? 2 : j == 9 ? 3 : 4;
            float fit =
                center + p0 - mu * scale * p1 + (mu * mu + skew * mu - variance) * scale2 * p2;
            fits[(k * V + row) * N + x] = fit;
            if (j == 6)
                r[row * N + x] = v[t + 30] - fit;
        }
    }
}
extern "C" __global__ void row_correlations(const float *residual, float *rho) {
    __shared__ float work[B / 32];
    int row = blockIdx.x, t = threadIdx.x;
    const float *r = residual + row * N;
    float sum = 0;
    for (int x = t; x < N; x += B)
        sum += r[x];
    float mean = block_sum(sum, work) / float(N);
    float vtotal = 0, p1 = 0, p2 = 0;
    for (int x = t; x < N; x += B) {
        float a = r[x] - mean;
        vtotal += a * a;
        if (x >= 1)
            p1 += a * (r[x - 1] - mean);
        if (x >= 2)
            p2 += a * (r[x - 2] - mean);
    }
    float variance = fmaxf(block_sum(vtotal, work) / float(N), 1.1754943508222875e-38f);
    float rho1 = fminf(fmaxf((block_sum(p1, work) / float(N - 1)) / variance, 0.0f), 1.0f);
    float rho2 = fminf(fmaxf((block_sum(p2, work) / float(N - 2)) / variance, 0.0f), 1.0f);
    if (t == 0) {
        rho[row * 2] = rho1;
        rho[row * 2 + 1] = rho2;
    }
}
extern "C" __global__ void blend_tiles(const float *fits, const float *residual, const float *rho,
                                       float *output) {
    int row = blockIdx.y, x = blockIdx.x * T + threadIdx.x;
    if (x >= N)
        return;
    const float *r = residual + row * N;
    float rho1 = rho[row * 2], rho2 = rho[row * 2 + 1];
    float scatter = (1.4826f * local_median(r, x)) * SCATTER_NORMALIZATION;
    float out = fits[row * N + x]; /* BLEND */
    output[row * N + x] = out;
}
