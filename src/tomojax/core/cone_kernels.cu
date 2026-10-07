// CUDA kernels for cone-beam Joseph projection; see tomojax/core/cone.py.
#define NC 25
#define TU 8
#define TV 128
#define GU 4
#define GV 128
#define QLEN 224
#define TB 64
#define TC 64
#define ULEN 112
#define PB 32
#define PC 32
#define VRUN 8
#define SEL(i, x0, x1, x2) ((i) == 0 ? (x0) : ((i) == 1 ? (x1) : (x2)))
__device__ __forceinline__ float trif(float f, float j) { return fmaxf(1.f - fabsf(f - j), 0.f); }
__device__ __forceinline__ bool separable(const float* c) { return c[12] > 0.5f; }
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
    r0 = (c[3] + fu * c[6] + fv * c[9]) - c[0];
    r1 = (c[4] + fu * c[7] + fv * c[10]) - c[1];
    r2 = (c[5] + fu * c[8] + fv * c[11]) - c[2];
}
__device__ __forceinline__ float path_w(float r0, float r1, float r2, float ra,
                                        float sx, float sy, float sz) {
    float px = r0 * sx, py = r1 * sy, pz = r2 * sz;
    return sqrtf(px * px + py * py + pz * pz) / fabsf(ra);
}

// Per-ray Joseph sum for one pixel, any view.
__device__ float ray_sum(const float* __restrict__ vol, const float* c, float fu, float fv,
                         int nx, int ny, int nz)
{
    float r0, r1, r2; ray_of(c, fu, fv, r0, r1, r2);
    int a = ray_axis(r0, r1, r2), b = a == 0 ? 1 : 0, cc = a == 2 ? 1 : 2;
    int na = SEL(a, nx, ny, nz), nb = SEL(b, nx, ny, nz), nc = SEL(cc, nx, ny, nz);
    long s0 = (long)ny * nz, s1 = nz;
    long sa = SEL(a, s0, s1, 1L), sb = SEL(b, s0, s1, 1L), sc = SEL(cc, s0, s1, 1L);
    float ra = SEL(a, r0, r1, r2), rb = SEL(b, r0, r1, r2), rc = SEL(cc, r0, r1, r2);
    float Sa = c[a], Sb = c[b], Sc = c[cc];
    float inv = 1.0f / ra;
    float lo = -1e30f, hi = 1e30f;
    float slope = rb * inv, base = Sb - Sa * slope;
    if (fabsf(slope) < 1e-12f) { if (base <= -1.f || base >= (float)nb) return 0.f; }
    else { float k1 = (-1.f - base) / slope, k2 = ((float)nb - base) / slope;
           lo = fmaxf(lo, fminf(k1, k2)); hi = fminf(hi, fmaxf(k1, k2)); }
    slope = rc * inv; base = Sc - Sa * slope;
    if (fabsf(slope) < 1e-12f) { if (base <= -1.f || base >= (float)nc) return 0.f; }
    else { float k1 = (-1.f - base) / slope, k2 = ((float)nc - base) / slope;
           lo = fmaxf(lo, fminf(k1, k2)); hi = fminf(hi, fmaxf(k1, k2)); }
    int k0 = max((int)floorf(lo), 0), k1 = min((int)ceilf(hi), na - 1);
    float sum = 0.f;
    for (int k = k0; k <= k1; ++k) {
        float t = ((float)k - Sa) * inv;
        float fb = Sb + t * rb, fc = Sc + t * rc;
        float fb0 = floorf(fb), fc0 = floorf(fc);
        int b0 = (int)fb0, c0 = (int)fc0;
        float wb0 = trif(fb, fb0), wb1 = trif(fb, fb0 + 1.f);
        float wc0 = trif(fc, fc0), wc1 = trif(fc, fc0 + 1.f);
        bool b0in = b0 >= 0 && b0 < nb, b1in = b0 + 1 >= 0 && b0 + 1 < nb;
        bool c0in = c0 >= 0 && c0 < nc, c1in = c0 + 1 >= 0 && c0 + 1 < nc;
        const float* p = vol + (long)k * sa + (long)b0 * sb + (long)c0 * sc;
        if (b0in && c0in) sum += __ldg(p) * (wb0 * wc0);
        if (b0in && c1in) sum += __ldg(p + sc) * (wb0 * wc1);
        if (b1in && c0in) sum += __ldg(p + sb) * (wb1 * wc0);
        if (b1in && c1in) sum += __ldg(p + sb + sc) * (wb1 * wc1);
    }
    return sum;
}

// Separable view, columns u0 .. u0 + TU - 1 of one view. AX is the plane axis shared by
// every column of the block, or -1 when columns differ (rays near 45 degrees).
struct SepShared {
    float Q[TU][QLEN];
    float w0[TU], w1[TU];
    int b0[TU], lo[TU], len[TU], ax[TU];
};

template <int AX>
__device__ __forceinline__ void sep_forward(
    const float* __restrict__ vol, const float* c, float (&acc)[TU], SepShared& sh, int u0,
    int v0, int tid, int nx, int ny, int nz, int nu, int nv)
{
    float (&Q)[TU][QLEN] = sh.Q;
    float *s_w0 = sh.w0, *s_w1 = sh.w1;
    int *s_b0 = sh.b0, *clo = sh.lo, *clen = sh.len, *s_ax = sh.ax;
    float Sc = c[2], DVc = c[11], fv = (float)(v0 + tid);
    float inv[TU], rcv[TU], Sa_[TU];
    int ax_t = 0;
    float rb_t = 0.f, inv_t = 0.f, rc0_t = 0.f, Sa_t = 0.f, Sb_t = 0.f;
    #pragma unroll
    for (int i = 0; i < TU; ++i) {
        float fu = (float)min(u0 + i, nu - 1);
        float r0 = (c[3] + fu * c[6]) - c[0], r1 = (c[4] + fu * c[7]) - c[1];
        int a = AX >= 0 ? AX : (fabsf(r1) > fabsf(r0) ? 1 : 0);
        float ra = a == 0 ? r0 : r1;
        inv[i] = 1.0f / ra;
        Sa_[i] = c[a];
        rcv[i] = ((c[5] + fu * c[8]) + fv * DVc) - Sc;
        if (i == tid) {
            ax_t = a; inv_t = inv[i]; Sa_t = c[a]; Sb_t = c[1 - a];
            rb_t = a == 0 ? r1 : r0;
            rc0_t = c[5] + fu * c[8];
            s_ax[i] = a;
        }
    }
    int vlast = min(v0 + TV, nv) - 1;
    long sxl = (long)ny * nz;
    int kmax = AX == 0 ? nx : (AX == 1 ? ny : max(nx, ny));
    for (int k = 0; k < kmax; ++k) {
        __syncthreads();
        if (tid < TU) {
            int na = ax_t == 0 ? nx : ny;
            float t = ((float)k - Sa_t) * inv_t;
            float fb = Sb_t + t * rb_t, fl = floorf(fb);
            s_b0[tid] = (int)fl; s_w0[tid] = trif(fb, fl); s_w1[tid] = trif(fb, fl + 1.f);
            float f1 = Sc + t * ((rc0_t + (float)v0 * DVc) - Sc);
            float f2 = Sc + t * ((rc0_t + (float)vlast * DVc) - Sc);
            int lo = (int)floorf(fminf(f1, f2));
            clo[tid] = lo;
            clen[tid] = k < na ? (int)floorf(fmaxf(f1, f2)) + 2 - lo : 0;
        }
        __syncthreads();
        {
            int i = tid / (TV / TU), j0 = tid % (TV / TU);
            int a = AX >= 0 ? AX : s_ax[i], nb = a == 0 ? ny : nx;
            long sa = a == 0 ? sxl : (long)nz, sb = a == 0 ? (long)nz : sxl;
            const float* plane = vol + (long)k * sa;
            int len = min(clen[i], QLEN), lo = clo[i], b0 = s_b0[i];
            float w0 = (b0 >= 0 && b0 < nb) ? s_w0[i] : 0.f;
            float w1 = (b0 + 1 >= 0 && b0 + 1 < nb) ? s_w1[i] : 0.f;
            const float* p0 = plane + (long)min(max(b0, 0), nb - 1) * sb;
            const float* p1 = plane + (long)min(max(b0 + 1, 0), nb - 1) * sb;
            for (int j = j0; j < len; j += TV / TU) {
                int jc = lo + j;
                bool in = jc >= 0 && jc < nz;
                Q[i][j] = in ? __ldg(p0 + jc) * w0 + __ldg(p1 + jc) * w1 : 0.f;
            }
        }
        __syncthreads();
        if (v0 + tid < nv) {
            #pragma unroll
            for (int i = 0; i < TU; ++i) {
                if (AX < 0 && clen[i] == 0) continue;
                float t = ((float)k - Sa_[i]) * inv[i];
                float fc = Sc + t * rcv[i], fl = floorf(fc);
                int j = (int)fl - clo[i];
                float w0 = trif(fc, fl), w1 = trif(fc, fl + 1.f);
                if (clen[i] <= QLEN) {
                    if (j >= 0 && j + 1 < clen[i]) acc[i] += Q[i][j] * w0 + Q[i][j + 1] * w1;
                } else {
                    // Column range too long for Q: read the two rows directly.
                    int a = AX >= 0 ? AX : s_ax[i], nb = a == 0 ? ny : nx;
                    long sa = a == 0 ? sxl : (long)nz, sb = a == 0 ? (long)nz : sxl;
                    const float* plane = vol + (long)k * sa;
                    int b0 = s_b0[i], c0 = (int)fl;
                    float wb0 = (b0 >= 0 && b0 < nb) ? s_w0[i] : 0.f;
                    float wb1 = (b0 + 1 >= 0 && b0 + 1 < nb) ? s_w1[i] : 0.f;
                    const float* p0 = plane + (long)min(max(b0, 0), nb - 1) * sb;
                    const float* p1 = plane + (long)min(max(b0 + 1, 0), nb - 1) * sb;
                    if (c0 >= 0 && c0 < nz)
                        acc[i] += (__ldg(p0 + c0) * wb0 + __ldg(p1 + c0) * wb1) * w0;
                    if (c0 + 1 >= 0 && c0 + 1 < nz)
                        acc[i] += (__ldg(p0 + c0 + 1) * wb0 + __ldg(p1 + c0 + 1) * wb1) * w1;
                }
            }
        }
    }
}

__device__ __forceinline__ int column_axis(const float* c, int u) {
    float fu = (float)u;
    return fabsf((c[4] + fu * c[7]) - c[1]) > fabsf((c[3] + fu * c[6]) - c[0]) ? 1 : 0;
}

// Separable views: grid (u tiles, v tiles, views); out (view, u, v). They share one plane
// axis per detector column (a in {x, y}), so a column's rays share their in-plane row.
extern "C" __global__ void __launch_bounds__(TV, 7) cone_forward(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz)
{
    int view = blockIdx.z;
    const float* c = coeff + (long)view * NC;
    int u0 = blockIdx.x * TU, v0 = blockIdx.y * TV, tid = threadIdx.x, v = v0 + tid;
    float fv = (float)v;
    if (!separable(c)) return;  // general_forward
    float acc[TU];
    #pragma unroll
    for (int i = 0; i < TU; ++i) acc[i] = 0.f;
    {
        __shared__ SepShared sh;
        int first = column_axis(c, u0);
        bool mixed = false;
        for (int i = 1; i < TU; ++i) mixed |= column_axis(c, min(u0 + i, nu - 1)) != first;
        if (mixed) sep_forward<-1>(vol, c, acc, sh, u0, v0, tid, nx, ny, nz, nu, nv);
        else if (first == 0) sep_forward<0>(vol, c, acc, sh, u0, v0, tid, nx, ny, nz, nu, nv);
        else sep_forward<1>(vol, c, acc, sh, u0, v0, tid, nx, ny, nz, nu, nv);
    }
    if (v < nv) {
        #pragma unroll
        for (int i = 0; i < TU; ++i) {
            int u = u0 + i;
            if (u < nu) {
                float r0, r1, r2; ray_of(c, (float)u, fv, r0, r1, r2);
                float ra = SEL(ray_axis(r0, r1, r2), r0, r1, r2);
                out[((long)view * nu + u) * nv + v] = acc[i] * path_w(r0, r1, r2, ra, sx, sy, sz);
            }
        }
    }
}

// Non-separable views, one ray per thread: grid (u tiles, v tiles, views); out (view, u, v).
extern "C" __global__ void __launch_bounds__(GU * GV) general_forward(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz)
{
    int view = blockIdx.z;
    const float* c = coeff + (long)view * NC;
    if (separable(c)) return;  // cone_forward
    int u = blockIdx.x * GU + threadIdx.x / GV, v = blockIdx.y * GV + threadIdx.x % GV;
    if (u >= nu || v >= nv) return;
    float fu = (float)u, fv = (float)v;
    float sum = ray_sum(vol, c, fu, fv, nx, ny, nz);
    float r0, r1, r2; ray_of(c, fu, fv, r0, r1, r2);
    float ra = SEL(ray_axis(r0, r1, r2), r0, r1, r2);
    out[((long)view * nu + u) * nv + v] = sum * path_w(r0, r1, r2, ra, sx, sy, sz);
}

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
    out[id] = img[id] * path_w(r0, r1, r2, ra, sx, sy, sz);
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
        float u1 = (K * rb0 - g1 * ra0) / (g1 * DUa - K * DUb);
        float u2 = (K * rb0 - g2 * ra0) / (g2 * DUa - K * DUb);
        int ulo_all = max((int)floorf(fminf(u1, u2)) - 1, ab[0]);
        int uhi = min((int)ceilf(fmaxf(u1, u2)) + 1, ab[1]);
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
                // z cells rise with v at 1/idf per row: cell j takes the rows strictly
                // inside ((j - 1 - f0) idf, (j + 1 - f0) idf), at most taps of them.
                float idf = 1.f / (t * DVc);
                s_f0[i] = Sc + t * (rc0 - Sc); s_idf[i] = idf;
                s_taps[i] = col_axis == a ? (int)floorf(2.f * fabsf(idf)) + 1 : 0;
            }
            __syncthreads();
            for (int i = tid / TC, j = tid % TC; i < ulen; i += blockDim.x / TC) {
                float jcf = (float)(c0 + j);
                float t = s_t[i], rc0 = s_rc0[i];
                int va = (int)floorf((jcf - 1.f - s_f0[i]) * s_idf[i]) + 1;
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
                float w1 = (K * rb0 - h1 * ra0) / (h1 * DUa - K * DUb);
                float w2 = (K * rb0 - h2 * ra0) / (h2 * DUa - K * DUb);
                int ia = max((int)floorf(fminf(w1, w2)) - 1 - ulo, 0);
                int ib = min((int)ceilf(fmaxf(w1, w2)) + 1 - ulo, ulen - 1);
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

// Adds a crossing's pending 2 x 2 cells at (jb, jc) of a tile with nb0 x nc0 valid cells.
__device__ __forceinline__ void flush(float (*acc)[PC + 1], int jb, int jc, float x00, float x01,
                                      float x10, float x11, int nb0, int nc0)
{
    bool b0in = jb >= 0 && jb < nb0, b1in = jb + 1 >= 0 && jb + 1 < nb0;
    bool c0in = jc >= 0 && jc < nc0, c1in = jc + 1 >= 0 && jc + 1 < nc0;
    if (b0in && c0in && x00 != 0.f) atomicAdd(&acc[jb][jc], x00);
    if (b0in && c1in && x01 != 0.f) atomicAdd(&acc[jb][jc + 1], x01);
    if (b1in && c0in && x10 != 0.f) atomicAdd(&acc[jb + 1][jc], x10);
    if (b1in && c1in && x11 != 0.f) atomicAdd(&acc[jb + 1][jc + 1], x11);
}

// Non-separable views, rays sampled along axis a: grid (c tiles, b tiles, planes k).
// Each block owns a PB x PC tile of plane k. Per view each thread walks VRUN rows of one
// detector column in the tile's footprint; their crossings of plane k step monotonically
// through the tile, so it keeps the 2 x 2 cells around the current crossing in registers
// and adds them to the tile only as it moves on. img is path-weighted.
extern "C" __global__ void plane_adjoint(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol, int nx, int ny, int nz, int nu, int nv)
{
    __shared__ float acc[PB][PC + 1];
    __shared__ float cf[NC];
    __shared__ int box[4];
    int b = a == 0 ? 1 : 0, cc = a == 2 ? 1 : 2;
    int nb = SEL(b, nx, ny, nz), nc = SEL(cc, nx, ny, nz);
    int c0 = blockIdx.x * PC, b0 = blockIdx.y * PB, k = blockIdx.z;
    int nb0 = min(PB, nb - b0), nc0 = min(PC, nc - c0);
    if (!any_view(coeff, axis_box, nviews, a, false)) return;
    for (int i = threadIdx.x; i < PB * (PC + 1); i += blockDim.x) (&acc[0][0])[i] = 0.f;
    for (int w = 0; w < nviews; ++w) {
        __syncthreads();
        if (threadIdx.x < NC) cf[threadIdx.x] = coeff[(long)w * NC + threadIdx.x];
        __syncthreads();
        const float* c = cf;
        const int* ab = axis_box + ((long)w * 3 + a) * 4;
        if (separable(c) || ab[1] < ab[0]) continue;
        if (threadIdx.x == 0) {
            float umin = 1e30f, umax = -1e30f, vmin = 1e30f, vmax = -1e30f;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                // Corners of the tile's part inside the volume (see sep_adjoint).
                float pb = q & 1 ? (float)(b0 + nb0) : (float)(b0 - 1);
                float pc = q & 2 ? (float)(c0 + nc0) : (float)(c0 - 1);
                float pk = (float)k;
                float d0 = (a == 0 ? pk : pb) - c[0];
                float d1 = (a == 1 ? pk : (a == 0 ? pb : pc)) - c[1];
                float d2 = (a == 2 ? pk : pc) - c[2];
                float lam = c[16] / (c[13] * d0 + c[14] * d1 + c[15] * d2);
                float uu = c[23] + lam * (c[17] * d0 + c[18] * d1 + c[19] * d2);
                float vv = c[24] + lam * (c[20] * d0 + c[21] * d1 + c[22] * d2);
                umin = fminf(umin, uu); umax = fmaxf(umax, uu);
                vmin = fminf(vmin, vv); vmax = fmaxf(vmax, vv);
            }
            box[0] = max((int)ceilf(umin), ab[0]); box[1] = min((int)floorf(umax), ab[1]);
            box[2] = max((int)ceilf(vmin), ab[2]); box[3] = min((int)floorf(vmax), ab[3]);
        }
        __syncthreads();
        int ulo = box[0], vlo = box[2], wu = box[1] - ulo + 1, wv = box[3] - vlo + 1;
        if (wu <= 0 || wv <= 0) continue;
        const float* image = img + (long)w * nu * nv;
        float Sa = comp(c, a), Sb = comp(c, b), Sc = comp(c, cc);
        float DVa = comp(c + 9, a), DVb = comp(c + 9, b), DVc = comp(c + 9, cc);
        float K = (float)k - Sa;
        int runs = (wv + VRUN - 1) / VRUN;
        for (int p = threadIdx.x; p < wu * runs; p += blockDim.x) {
            int u = ulo + p / runs, v0 = vlo + VRUN * (p % runs), v1 = min(v0 + VRUN, vlo + wv);
            float fu = (float)u;
            float ra = comp(c + 3, a) + fu * comp(c + 6, a) - Sa;
            float rb = comp(c + 3, b) + fu * comp(c + 6, b) - Sb;
            float rc = comp(c + 3, cc) + fu * comp(c + 6, cc) - Sc;
            const float* col = image + (long)u * nv;
            int cb = -100, ccj = -100;
            float x00 = 0.f, x01 = 0.f, x10 = 0.f, x11 = 0.f;
            for (int v = v0; v < v1; ++v) {
                float fv = (float)v;
                float r_a = ra + fv * DVa, r_b = rb + fv * DVb, r_c = rc + fv * DVc;
                float r0 = a == 0 ? r_a : r_b, r1 = a == 1 ? r_a : (a == 0 ? r_b : r_c);
                float r2 = a == 2 ? r_a : r_c;
                if (ray_axis(r0, r1, r2) != a) continue;
                float t = K / r_a;
                float fb = Sb + t * r_b, fc = Sc + t * r_c;
                float fb0 = floorf(fb), fc0 = floorf(fc);
                int jb = (int)fb0 - b0, jc = (int)fc0 - c0;
                if (jb < -1 || jb >= PB || jc < -1 || jc >= PC) continue;
                float value = __ldg(col + v);
                float wb1 = fb - fb0, wc1 = fc - fc0, wb0 = 1.f - wb1, wc0 = 1.f - wc1;
                if (jb != cb || jc != ccj) {
                    if (jb == cb && jc == ccj + 1) {
                        // Step one cell along c: the trailing cells are final.
                        flush(acc, cb, ccj, x00, 0.f, x10, 0.f, nb0, nc0);
                        x00 = x01; x10 = x11; x01 = 0.f; x11 = 0.f;
                    } else {
                        flush(acc, cb, ccj, x00, x01, x10, x11, nb0, nc0);
                        x00 = x01 = x10 = x11 = 0.f;
                    }
                    cb = jb; ccj = jc;
                }
                x00 += value * wb0 * wc0; x01 += value * wb0 * wc1;
                x10 += value * wb1 * wc0; x11 += value * wb1 * wc1;
            }
            flush(acc, cb, ccj, x00, x01, x10, x11, nb0, nc0);
        }
    }
    __syncthreads();
    long s0 = (long)ny * nz, s1 = nz;
    long sa = SEL(a, s0, s1, 1L), sb = SEL(b, s0, s1, 1L), sc = SEL(cc, s0, s1, 1L);
    for (int i = threadIdx.x; i < PB * PC; i += blockDim.x) {
        int jb = i / PC, jc = i % PC;
        if (jb < nb0 && jc < nc0)
            vol[(long)k * sa + (long)(b0 + jb) * sb + (long)(c0 + jc) * sc] += acc[jb][jc];
    }
}
