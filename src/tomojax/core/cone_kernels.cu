// CUDA kernels for cone-beam Joseph projection; see tomojax/core/cone.py.
#define NC 25
#define GU 32
#define GV 16
#define TB 64
#define TC 64
#define ULEN 112
#define PB 32
#define PC 32
#define VRUN 8
#define SEL(i, x0, x1, x2) ((i) == 0 ? (x0) : ((i) == 1 ? (x1) : (x2)))
__device__ __forceinline__ float trif(float f, float j) { return fmaxf(1.f - fabsf(f - j), 0.f); }
__device__ __forceinline__ bool separable(const float* c) { return c[12] > 0.5f; }
// A projected interval has no finite corner bound when its denominator crosses zero.
__device__ __forceinline__ bool crosses_zero(float lo, float hi) {
    return (lo <= 0.f && hi >= 0.f) || (hi <= 0.f && lo >= 0.f);
}
// Clamp in floating point before converting footprint bounds to integers. The
// two out-of-detector sentinels also leave room for the separable path's padding.
__device__ __forceinline__ float pixel_bound(float x, int size) {
    return fminf(fmaxf(x, -2.f), (float)size + 1.f);
}
// Whether any of the block's nviews (<= blockDim.x) views is (non-)separable and has rays
// sampled along axis a; every thread of the block must call it.
__device__ __forceinline__ bool any_view(const float* coeff, const int* axis_box, int nviews,
                                         int a, bool sep) {
    int i = threadIdx.x;
    bool hit = i < nviews && separable(coeff + (long)i * NC) == sep
        && axis_box[((long)i * 3 + a) * 4 + 1] >= axis_box[((long)i * 3 + a) * 4];
    return __syncthreads_or(hit);
}
// Component i of the 3-vector at p.
__device__ __forceinline__ float comp(const float* p, int i) { return SEL(i, p[0], p[1], p[2]); }
// The ray's plane axis: its largest index-space component, ties to the lower axis.
__device__ __forceinline__ int ray_axis(float r0, float r1, float r2) {
    int a = 0; float m = fabsf(r0);
    if (fabsf(r1) > m) { a = 1; m = fabsf(r1); }
    if (fabsf(r2) > m) a = 2;
    return a;
}
__device__ __forceinline__ void ray_of(const float* c, float fu, float fv,
                                       float& r0, float& r1, float& r2) {
    // Explicit rounding locks the sampling model across NVRTC optimization contexts.
    // In particular, subtracting the source before adding v can change an axis tie.
    r0 = __fsub_rn(__fmaf_rn(fv, c[9], __fmaf_rn(fu, c[6], c[3])), c[0]);
    r1 = __fsub_rn(__fmaf_rn(fv, c[10], __fmaf_rn(fu, c[7], c[4])), c[1]);
    r2 = __fsub_rn(__fmaf_rn(fv, c[11], __fmaf_rn(fu, c[8], c[5])), c[2]);
}
__device__ __forceinline__ float path_w(float r0, float r1, float r2, float ra,
                                        float sx, float sy, float sz) {
    float px = r0 * sx, py = r1 * sy, pz = r2 * sz;
    return sqrtf(px * px + py * py + pz * pz) / fabsf(ra);
}

// Per-ray Joseph sum for one pixel, any view.
// The smallest and largest of a value over the (whole) warp.
__device__ __forceinline__ int warp_min(int x) {
    for (int o = 16; o; o >>= 1) x = min(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}
__device__ __forceinline__ int warp_max(int x) {
    for (int o = 16; o; o >>= 1) x = max(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}

// Planes k0..k1 of a ray, stepped by the warp through first..last (see ray_sum): plane
// k lies sa apart, in-plane axes b and c sb and sc (SC1: c is the contiguous axis).
template <int SC1>
__device__ __forceinline__ float plane_sum(
    const float* __restrict__ vol, int k0, int k1, int first, int last, float Sa, float Sb,
    float Sc, float inv, float rb, float rc, int nb, int nc, long sa, long sb, long sc = 1)
{
    if (SC1) sc = 1;
    float sum = 0.f;
    const float* plane = vol + (long)first * sa;
    for (int k = first; k <= last; ++k, plane += sa) {
        if (k < k0 || k > k1) continue;
        // Same reciprocal, multiply and FMA sequence as plane_adjoint.
        float t = __fmul_rn(__fsub_rn((float)k, Sa), inv);
        float fb = __fmaf_rn(t, rb, Sb), fc = __fmaf_rn(t, rc, Sc);
        float fb0 = floorf(fb), fc0 = floorf(fc);
        int b0 = (int)fb0, c0 = (int)fc0;
        // The same weights as plane_adjoint's.
        float wb1 = fb - fb0, wc1 = fc - fc0, wb0 = 1.f - wb1, wc0 = 1.f - wc1;
        bool b0in = (unsigned)b0 < (unsigned)nb, b1in = (unsigned)(b0 + 1) < (unsigned)nb;
        bool c0in = (unsigned)c0 < (unsigned)nc, c1in = (unsigned)(c0 + 1) < (unsigned)nc;
        const float* p = plane + (long)b0 * sb + (long)c0 * sc;
        float s0 = 0.f, s1 = 0.f;
        if (b0in && c0in) s0 = __ldg(p) * wc0;
        if (b0in && c1in) s0 += __ldg(p + sc) * wc1;
        if (b1in && c0in) s1 = __ldg(p + sb) * wc0;
        if (b1in && c1in) s1 += __ldg(p + sb + sc) * wc1;
        sum += wb0 * s0 + wb1 * s1;
    }
    return sum;
}

// Called by every lane of a warp (``live`` false for lanes without a ray). The lanes
// step through the planes together, each sampling only its own ray's planes: rays
// entering through the volume's top or bottom start at different planes, and lanes
// left on different planes would not share cache lines.
__device__ float ray_sum(const float* __restrict__ vol, const float* c, float fu, float fv,
                         int nx, int ny, int nz, bool live)
{
    float r0, r1, r2; ray_of(c, fu, fv, r0, r1, r2);
    int a = ray_axis(r0, r1, r2), b = a == 0 ? 1 : 0, cc = a == 2 ? 1 : 2;
    int na = SEL(a, nx, ny, nz), nb = SEL(b, nx, ny, nz), nc = SEL(cc, nx, ny, nz);
    long s0 = (long)ny * nz, s1 = nz;
    long sa = SEL(a, s0, s1, 1L), sb = SEL(b, s0, s1, 1L), sc = SEL(cc, s0, s1, 1L);
    float ra = SEL(a, r0, r1, r2), rb = SEL(b, r0, r1, r2), rc = SEL(cc, r0, r1, r2);
    float Sa = c[a], Sb = c[b], Sc = c[cc];
    float inv = __frcp_rn(ra);
    float lo = -1e30f, hi = 1e30f;
    float slope = rb * inv, base = Sb - Sa * slope;
    if (fabsf(slope) < 1e-12f) { if (base <= -1.f || base >= (float)nb) live = false; }
    else { float k1 = (-1.f - base) / slope, k2 = ((float)nb - base) / slope;
           lo = fmaxf(lo, fminf(k1, k2)); hi = fminf(hi, fmaxf(k1, k2)); }
    slope = rc * inv; base = Sc - Sa * slope;
    if (fabsf(slope) < 1e-12f) { if (base <= -1.f || base >= (float)nc) live = false; }
    else { float k1 = (-1.f - base) / slope, k2 = ((float)nc - base) / slope;
           lo = fmaxf(lo, fminf(k1, k2)); hi = fminf(hi, fmaxf(k1, k2)); }
    int k0 = max((int)floorf(lo), 0), k1 = min((int)ceilf(hi), na - 1);
    if (!live) { k0 = na; k1 = -1; }
    int first = warp_min(k0), last = warp_max(k1);
    // Rays of one warp nearly always share a plane axis; z (contiguous) is then the
    // in-plane c axis at compile time.
    unsigned busy = __ballot_sync(0xffffffffu, k0 <= k1), x = __ballot_sync(0xffffffffu, a == 0);
    if ((busy & ~x) == 0)
        return plane_sum<1>(vol, k0, k1, first, last, Sa, Sb, Sc, inv, rb, rc, nb, nc,
                            (long)ny * nz, nz);
    if ((busy & x) == 0 && a != 2)
        return plane_sum<1>(vol, k0, k1, first, last, Sa, Sb, Sc, inv, rb, rc, nb, nc,
                            nz, (long)ny * nz);
    return plane_sum<0>(vol, k0, k1, first, last, Sa, Sb, Sc, inv, rb, rc, nb, nc, sa, sb, sc);
}

// One ray per thread: grid (views, u tiles, v tiles); out (view, v, u).
extern "C" __global__ void __launch_bounds__(GU * GV) cone_forward(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz)
{
    int view = blockIdx.x;
    const float* c = coeff + (long)view * NC;
    int u = blockIdx.y * GU + threadIdx.x / GV, v = blockIdx.z * GV + threadIdx.x % GV;
    bool live = u < nu && v < nv;
    float fu = (float)u, fv = (float)v;
    float sum = ray_sum(vol, c, fu, fv, nx, ny, nz, live);
    if (!live) return;
    float r0, r1, r2; ray_of(c, fu, fv, r0, r1, r2);
    float ra = SEL(ray_axis(r0, r1, r2), r0, r1, r2);
    // Written (view, v, u), as JAX holds projections: one store per ray, so the
    // stride costs little, and no transposed copy of the projections is needed.
    out[((long)view * nv + v) * nu + u] = sum * path_w(r0, r1, r2, ra, sx, sy, sz);
}

// Path-weighted images for the transposes, reordered from JAX's (view, v, u) to the
// (view, u, v) they read column by column.
extern "C" __global__ void weight_images(
    const float* __restrict__ coeff, const float* __restrict__ img, float* __restrict__ out,
    int nviews, int nu, int nv, float sx, float sy, float sz)
{
    long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long)nviews * nu * nv) return;
    int v = id % nv; int u = (id / nv) % nu; int view = id / ((long)nu * nv);
    const float* c = coeff + (long)view * NC;
    float r0, r1, r2; ray_of(c, (float)u, (float)v, r0, r1, r2);
    float ra = SEL(ray_axis(r0, r1, r2), r0, r1, r2);
    out[id] = img[((long)view * nv + v) * nu + u] * path_w(r0, r1, r2, ra, sx, sy, sz);
}

// Separable views, the columns whose plane axis is a: grid (z tiles, row tiles, planes).
// img is path-weighted; axis_box is from axis_boxes.
extern "C" __global__ void sep_adjoint(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol,
    int nx, int ny, int nz, int nu, int nv)
{
    __shared__ float R[ULEN][TC + 1];
    __shared__ float s_fb[ULEN], s_t[ULEN], s_rc0[ULEN], s_f0[ULEN], s_idf[ULEN];
    __shared__ int s_taps[ULEN];
    __shared__ float cf[NC];
    __shared__ int skip;
    int b = 1 - a;
    int nb = a == 0 ? ny : nx;
    int c0 = blockIdx.x * TC, b0 = blockIdx.y * TB, k = blockIdx.z;
    int tid = threadIdx.x;
    int my_b = b0 + tid / 4, my_c = (tid % 4) * (TC / 4);
    float my_bf = (float)my_b;
    if (!any_view(coeff, axis_box, nviews, a, true)) return;
    float acc[TC / 4];
    #pragma unroll
    for (int j = 0; j < TC / 4; ++j) acc[j] = 0.f;
    for (int w = 0; w < nviews; ++w) {
        __syncthreads();
        if (tid < NC) cf[tid] = coeff[(long)w * NC + tid];
        __syncthreads();
        const int* ab = axis_box + ((long)w * 3 + a) * 4;
        if (tid == 0) skip = !separable(cf) || ab[1] < ab[0];
        __syncthreads();
        if (skip) continue;
        const float* c = cf;
        float Sa = c[a], Sb = c[b], Sc = c[2], DVc = c[11];
        float K = (float)k - Sa;
        float ra0 = c[3 + a] - c[a], rb0 = c[3 + b] - c[b], DUa = c[6 + a], DUb = c[6 + b];
        // The tile's rows inside the volume: a far edge beyond it can project through
        // infinity when the source is close, giving the wrong column range.
        float g1 = (float)b0 - 1.f - Sb, g2 = (float)min(b0 + TB, nb) - Sb;
        float den1 = g1 * DUa - K * DUb, den2 = g2 * DUa - K * DUb;
        int ulo_all = ab[0], uhi = ab[1];
        if (!crosses_zero(den1, den2)) {
            float u1 = (K * rb0 - g1 * ra0) / den1;
            float u2 = (K * rb0 - g2 * ra0) / den2;
            ulo_all = max((int)floorf(pixel_bound(fminf(u1, u2), nu)) - 1, ab[0]);
            uhi = min((int)ceilf(pixel_bound(fmaxf(u1, u2), nu)) + 1, ab[1]);
        }
        const float* image = img + (long)w * nu * nv;
        for (int ulo = ulo_all; ulo <= uhi; ulo += ULEN) {
            int ulen = min(ULEN, uhi - ulo + 1);
            __syncthreads();
            for (int i = tid; i < ulen; i += blockDim.x) {
                float fu = (float)(ulo + i);
                float r0 = (c[3] + fu * c[6]) - c[0], r1 = (c[4] + fu * c[7]) - c[1];
                int col_axis = fabsf(r1) > fabsf(r0) ? 1 : 0;
                float ra = a == 0 ? r0 : r1, rb = a == 0 ? r1 : r0;
                float t = K * (1.0f / ra);
                float rc0 = c[5] + fu * c[8];
                // Columns sampled along the other axis contribute nothing here.
                s_t[i] = t; s_fb[i] = col_axis == a ? Sb + t * rb : -1e30f; s_rc0[i] = rc0;
                // z cells step with v at 1/idf per row (either sign). Cell j takes
                // the rows strictly between (j - 1 - f0) idf and (j + 1 - f0) idf,
                // at most taps of them.
                float idf = 1.f / (t * DVc);
                s_f0[i] = Sc + t * (rc0 - Sc); s_idf[i] = idf;
                // A footprint wider than the image still needs at most nv rows.
                float taps = fminf(2.f * fabsf(idf), (float)nv);
                s_taps[i] = col_axis == a ? min((int)floorf(taps) + 1, nv) : 0;
            }
            __syncthreads();
            for (int i = tid / TC, j = tid % TC; i < ulen; i += blockDim.x / TC) {
                float jcf = (float)(c0 + j);
                float t = s_t[i], rc0 = s_rc0[i];
                float v1 = (jcf - 1.f - s_f0[i]) * s_idf[i];
                float v2 = (jcf + 1.f - s_f0[i]) * s_idf[i];
                float lower = fminf(v1, v2);
                int va = max((int)floorf(pixel_bound(lower, nv)) + 1, 0);
                int taps = s_taps[i];   // the same for the whole warp: no divergence
                const float* col = image + (long)(ulo + i) * nv;
                float s = 0.f;
                #pragma unroll 4
                for (int q = 0; q < taps; ++q) {
                    int vv = va + q;
                    if (vv < 0 || vv >= nv) continue;
                    s += __ldg(col + vv) * trif(Sc + t * ((rc0 + (float)vv * DVc) - Sc), jcf);
                }
                R[i][j] = s;
            }
            __syncthreads();
            if (my_b < nb) {
                float h1 = my_bf - 1.f - Sb, h2 = my_bf + 1.f - Sb;
                float den1 = h1 * DUa - K * DUb, den2 = h2 * DUa - K * DUb;
                int ia = 0, ib = ulen - 1;
                if (!crosses_zero(den1, den2)) {
                    float w1 = (K * rb0 - h1 * ra0) / den1;
                    float w2 = (K * rb0 - h2 * ra0) / den2;
                    ia = max((int)floorf(pixel_bound(fminf(w1, w2), nu)) - 1 - ulo, 0);
                    ib = min((int)ceilf(pixel_bound(fmaxf(w1, w2), nu)) + 1 - ulo, ulen - 1);
                }
                for (int i = ia; i <= ib; ++i) {
                    float wb = trif(s_fb[i], my_bf);
                    if (wb == 0.f) continue;
                    #pragma unroll
                    for (int j = 0; j < TC / 4; ++j) acc[j] += wb * R[i][my_c + j];
                }
            }
        }
    }
    int na = a == 0 ? nx : ny;
    if (my_b < nb && k < na) {
        int ix = a == 0 ? k : my_b, iy = a == 0 ? my_b : k;
        long base = ((long)ix * ny + iy) * nz + c0 + my_c;
        #pragma unroll
        for (int j = 0; j < TC / 4; ++j) if (c0 + my_c + j < nz) vol[base + j] += acc[j];
    }
}

// Per view and plane axis, the bounding box (u0, u1, v0, v1) of the pixels whose rays
// are sampled along that axis; an empty box has u1 < u0. grid (views,).
extern "C" __global__ void axis_boxes(
    const float* __restrict__ coeff, int* __restrict__ boxes, int nu, int nv)
{
    __shared__ int box[3][4];
    const float* c = coeff + (long)blockIdx.x * NC;
    if (threadIdx.x < 3) {
        box[threadIdx.x][0] = nu; box[threadIdx.x][1] = -1;
        box[threadIdx.x][2] = nv; box[threadIdx.x][3] = -1;
    }
    __syncthreads();
    int lo[3][2], hi[3][2];
    #pragma unroll
    for (int a = 0; a < 3; ++a) { lo[a][0] = nu; hi[a][0] = -1; lo[a][1] = nv; hi[a][1] = -1; }
    for (int p = threadIdx.x; p < nu * nv; p += blockDim.x) {
        int u = p / nv, v = p % nv;
        float r0, r1, r2; ray_of(c, (float)u, (float)v, r0, r1, r2);
        int a = ray_axis(r0, r1, r2);
        #pragma unroll
        for (int q = 0; q < 3; ++q) if (q == a) {
            lo[q][0] = min(lo[q][0], u); hi[q][0] = max(hi[q][0], u);
            lo[q][1] = min(lo[q][1], v); hi[q][1] = max(hi[q][1], v);
        }
    }
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        atomicMin(&box[a][0], lo[a][0]); atomicMax(&box[a][1], hi[a][0]);
        atomicMin(&box[a][2], lo[a][1]); atomicMax(&box[a][3], hi[a][1]);
    }
    __syncthreads();
    if (threadIdx.x < 12) boxes[(long)blockIdx.x * 12 + threadIdx.x] = (&box[0][0])[threadIdx.x];
}

// Adds x0..x3 to four cells of shared memory (padded: no bounds to test); zeros are skipped.
__device__ __forceinline__ void add4(float* p0, float x0, float* p1, float x1, float* p2,
                                     float x2, float* p3, float x3)
{
    if (x0 != 0.f) atomicAdd(p0, x0);
    if (x1 != 0.f) atomicAdd(p1, x1);
    if (x2 != 0.f) atomicAdd(p2, x2);
    if (x3 != 0.f) atomicAdd(p3, x3);
}

// Non-separable views, rays sampled along axis a: grid (c tiles, b tiles, planes k).
// Each block owns a PB x PC tile of plane k. Each of its views' footprints on the tile is
// found by a thread of its own, and the views' runs of VRUN rows of one detector column
// are then walked as one list, without a barrier per view. Each run's crossings of plane k
// step monotonically through the tile, so its thread keeps the 2 x 2 cells around the
// current crossing in registers and adds them to the tile only as it moves on. img is
// path-weighted; nviews <= VMAX.
#define VMAX 32
extern "C" __global__ void __launch_bounds__(128, 8) plane_adjoint(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol, int nx, int ny, int nz, int nu, int nv)
{
    __shared__ float acc[PB + 2][PC + 3];  // a cell of padding round the tile: no bounds tests
    __shared__ float cf[VMAX][12];  // source, detector origin, u and v axes
    __shared__ int foot[VMAX][4];   // first column and row, rows, runs per column
    __shared__ int first[VMAX + 1]; // each view's first run in the block's list
    int b = a == 0 ? 1 : 0, cc = a == 2 ? 1 : 2;
    int nb = SEL(b, nx, ny, nz), nc = SEL(cc, nx, ny, nz);
    int c0 = blockIdx.x * PC, b0 = blockIdx.y * PB, k = blockIdx.z;
    int nb0 = min(PB, nb - b0), nc0 = min(PC, nc - c0);
    if (!any_view(coeff, axis_box, nviews, a, false)) return;
    for (int i = threadIdx.x; i < (PB + 2) * (PC + 3); i += blockDim.x) (&acc[0][0])[i] = 0.f;
    int t = threadIdx.x, runs_here = 0;
    if (t < nviews) {
        const float* c = coeff + (long)t * NC;
        const int* ab = axis_box + ((long)t * 3 + a) * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            cf[t][3*i] = comp(c + 3*i, a);
            cf[t][3*i + 1] = comp(c + 3*i, b);
            cf[t][3*i + 2] = comp(c + 3*i, cc);
        }
        if (!separable(c) && ab[1] >= ab[0]) {
            float umin = 1e30f, umax = -1e30f, vmin = 1e30f, vmax = -1e30f;
            float denmin = 1e30f, denmax = -1e30f;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                // Corners of the tile's part inside the volume (see sep_adjoint).
                float pb = q & 1 ? (float)(b0 + nb0) : (float)(b0 - 1);
                float pc = q & 2 ? (float)(c0 + nc0) : (float)(c0 - 1);
                float pk = (float)k;
                float d0 = (a == 0 ? pk : pb) - c[0];
                float d1 = (a == 1 ? pk : (a == 0 ? pb : pc)) - c[1];
                float d2 = (a == 2 ? pk : pc) - c[2];
                float den = c[13] * d0 + c[14] * d1 + c[15] * d2;
                denmin = fminf(denmin, den); denmax = fmaxf(denmax, den);
                float lam = c[16] / den;
                float uu = c[23] + lam * (c[17] * d0 + c[18] * d1 + c[19] * d2);
                float vv = c[24] + lam * (c[20] * d0 + c[21] * d1 + c[22] * d2);
                umin = fminf(umin, uu); umax = fmaxf(umax, uu);
                vmin = fminf(vmin, vv); vmax = fmaxf(vmax, vv);
            }
            // The denominator is affine over the tile. If it crosses zero, its
            // footprint passes through infinity: the corners cannot bound it.
            bool unbounded = crosses_zero(denmin, denmax);
            int ulo = unbounded ? ab[0] : max((int)ceilf(pixel_bound(umin, nu)), ab[0]);
            int uhi = unbounded ? ab[1] : min((int)floorf(pixel_bound(umax, nu)), ab[1]);
            int vlo = unbounded ? ab[2] : max((int)ceilf(pixel_bound(vmin, nv)), ab[2]);
            int vhi = unbounded ? ab[3] : min((int)floorf(pixel_bound(vmax, nv)), ab[3]);
            int wu = uhi - ulo + 1, wv = vhi - vlo + 1;
            if (wu > 0 && wv > 0) {
                int runs = (wv + VRUN - 1) / VRUN;
                runs_here = wu * runs;
                foot[t][0] = ulo; foot[t][1] = vlo; foot[t][2] = wv; foot[t][3] = runs;
            }
        }
    }
    if (t < 32) {  // each view's first run: a prefix sum over the (<= 32) views
        int x = runs_here;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, x, o);
            if (t >= o) x += y;
        }
        first[t + 1] = x;
        if (t == 0) first[0] = 0;
    }
    __syncthreads();
    int total = first[nviews];
    if (total == 0) return;
    int w = 0, cw = -1, ulo = 0, vlo = 0, wv = 0, runs = 1;
    const float* image = img;
    float Sa = 0.f, Sb = 0.f, Sc = 0.f, DVa = 0.f, DVb = 0.f, DVc = 0.f, K = 0.f;
    for (int p = t; p < total; p += blockDim.x) {
        while (p >= first[w + 1]) ++w;
        const float* c = cf[w];
        if (w != cw) {  // the view's constants, while this thread stays on it
            cw = w;
            ulo = foot[w][0]; vlo = foot[w][1]; wv = foot[w][2]; runs = foot[w][3];
            image = img + (long)w * nu * nv;
            Sa = c[0]; Sb = c[1]; Sc = c[2];
            DVa = c[9]; DVb = c[10]; DVc = c[11];
            K = __fsub_rn((float)k, Sa);
        }
        int q = p - first[w];
        int u = ulo + q / runs, v0 = vlo + VRUN * (q % runs), v1 = min(v0 + VRUN, vlo + wv);
        float fu = (float)u;
        // Keep the detector-column position, not position minus source: ray_of
        // adds the v displacement first, then subtracts the source.
        float ra = __fmaf_rn(fu, c[6], c[3]);
        float rb = __fmaf_rn(fu, c[7], c[4]);
        float rc = __fmaf_rn(fu, c[8], c[5]);
        const float* col = image + (long)u * nv;
        int cb = -100, ccj = -100;
        float x00 = 0.f, x01 = 0.f, x10 = 0.f, x11 = 0.f;
        // Load the run's pixels together: one wait for memory instead of VRUN.
        float values[VRUN];
        #pragma unroll
        for (int i = 0; i < VRUN; ++i) values[i] = v0 + i < v1 ? __ldg(col + v0 + i) : 0.f;
        // The run's crossings of plane k first, independent of each other, so their
        // divisions overlap; then their cells, in order.
        float fbs[VRUN], fcs[VRUN];
        #pragma unroll
        for (int i = 0; i < VRUN; ++i) {
            float fv = (float)(v0 + i);
            float r_a = __fsub_rn(__fmaf_rn(fv, DVa, ra), Sa);
            float r_b = __fsub_rn(__fmaf_rn(fv, DVb, rb), Sb);
            float r_c = __fsub_rn(__fmaf_rn(fv, DVc, rc), Sc);
            float aa = fabsf(r_a), ab = fabsf(r_b), ac = fabsf(r_c);
            float t_ = __fmul_rn(K, __frcp_rn(r_a));
            bool hit = v0 + i < v1
                && (a == 0 ? aa >= ab : aa > ab)
                && (a == 2 ? aa > ac : aa >= ac);
            fbs[i] = hit ? __fmaf_rn(t_, r_b, Sb) : -1e9f;  // off the tile: skipped below
            fcs[i] = __fmaf_rn(t_, r_c, Sc);
        }
        #pragma unroll
        for (int i = 0; i < VRUN; ++i) {
            float fb = fbs[i], fc = fcs[i];
            float fb0 = floorf(fb), fc0 = floorf(fc);
            int jb = (int)fb0 - b0, jc = (int)fc0 - c0;
            if (jb < -1 || jb >= PB || jc < -1 || jc >= PC) continue;
            float value = values[i];
            float wb1 = fb - fb0, wc1 = fc - fc0, wb0 = 1.f - wb1, wc0 = 1.f - wc1;
            if (jb != cb || jc != ccj) {
                float* cell = &acc[cb + 1][ccj + 1];
                bool next = jb == cb && jc == ccj + 1;
                add4(cell, x00, cell + 1, next ? 0.f : x01,
                     cell + (PC + 3), x10, cell + (PC + 4), next ? 0.f : x11);
                x00 = next ? x01 : 0.f; x10 = next ? x11 : 0.f;
                x01 = x11 = 0.f;
                cb = jb; ccj = jc;
            }
            x00 += value * wb0 * wc0; x01 += value * wb0 * wc1;
            x10 += value * wb1 * wc0; x11 += value * wb1 * wc1;
        }
        if (cb > -100) {
            float* cell = &acc[cb + 1][ccj + 1];
            add4(cell, x00, cell + 1, x01, cell + (PC + 3), x10, cell + (PC + 4), x11);
        }
    }
    __syncthreads();
    long s0 = (long)ny * nz, s1 = nz;
    long sa = SEL(a, s0, s1, 1L), sb = SEL(b, s0, s1, 1L), sc = SEL(cc, s0, s1, 1L);
    for (int i = threadIdx.x; i < PB * PC; i += blockDim.x) {
        int jb = i / PC, jc = i % PC;
        if (jb < nb0 && jc < nc0)
            vol[(long)k * sa + (long)(b0 + jb) * sb + (long)(c0 + jc) * sc] += acc[jb + 1][jc + 1];
    }
}
