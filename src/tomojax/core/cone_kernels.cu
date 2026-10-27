#ifndef VOLUME_Z_FAST
#define VOLUME_Z_FAST 0
#endif
// Solver-private axis order; public buffers remain (z, y, x).
__device__ __forceinline__ long volume_index(int x,int y,int z,int nx,int ny,int nz) {
    return VOLUME_Z_FAST ? ((long)x * ny + y) * nz + z
                         : ((long)z * ny + y) * nx + x;
}
// CUDA kernels for cone-beam Joseph projection; see tomojax/core/cone.py.
#define NC 25
#define TB 64
#define TC 64
#define ULEN 112
#define PB 32
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
// Bound roundoff when inverting the vertical ray coordinate. Cancellation can
// leave a positive forward weight outside the nominal integer support. The
// margin changes candidates only; every weight retains the forward arithmetic.
__device__ __forceinline__ float vertical_support_margin(
    float idf, float Sc, float t, float rc0, float DVc, int nz, int nv)
{
    return 8.f * 1.1920928955078125e-7f * fabsf(idf) *
        ((float)nz + fabsf(Sc) + fabsf(t) *
         (fabsf(rc0) + fabsf(Sc) + (float)nv * fabsf(DVc)) + 1.f);
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

__device__ __forceinline__ int warp_min(int x) {
    for (int o = 16; o; o >>= 1) x = min(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}
__device__ __forceinline__ int warp_max(int x) {
    for (int o = 16; o; o >>= 1) x = max(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}


// Source directions are already expressed in voxel-index coordinates. This
// direction-spread heuristic selects scheduling, not any ray or interpolation.
__device__ __forceinline__ float packet_dot(float3 a, float3 b) {
    return __fadd_rn(__fadd_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y)),
                     __fmul_rn(a.z, b.z));
}
__device__ __forceinline__ float3 packet_source(
    const float* c, float3 centre)
{
    return make_float3(__fsub_rn(c[0], centre.x), __fsub_rn(c[1], centre.y),
                       __fsub_rn(c[2], centre.z));
}
__device__ __forceinline__ bool dense_view_packet(
    const float* coeff, int first, int nviews, int nx, int ny, int nz)
{
    int last = min(first + 8, nviews);
    // A mostly empty cross-view warp cannot amortize its unused view lanes.
    if (last - first < 4) return false;
    float3 centre = make_float3(__fmul_rn((float)(nx - 1), .5f),
                               __fmul_rn((float)(ny - 1), .5f),
                               __fmul_rn((float)(nz - 1), .5f));
    float extent = (float)max(nx, max(ny, nz));
    float extent2 = __fmul_rn(extent, extent);
    float3 before = packet_source(coeff + (long)first * NC, centre);
    float before2 = packet_dot(before, before);
    for (int view = first + 1; view < last; ++view) {
        float3 next = packet_source(coeff + (long)view * NC, centre);
        float next2 = packet_dot(next, next);
        float3 cross = make_float3(
            __fsub_rn(__fmul_rn(before.y, next.z), __fmul_rn(before.z, next.y)),
            __fsub_rn(__fmul_rn(before.z, next.x), __fmul_rn(before.x, next.z)),
            __fsub_rn(__fmul_rn(before.x, next.y), __fmul_rn(before.y, next.x)));
        // Adjacent directions drift at most eight voxels over the grid extent.
        // Explicit rounding makes this SAME predicate identical in both grids.
        float spread = __fmul_rn(packet_dot(cross, cross), extent2);
        float limit = __fmul_rn(64.f, __fmul_rn(before2, next2));
        if (!(packet_dot(before, next) > 0.f && spread <= limit)) return false;
        before = next;
        before2 = next2;
    }
    return true;
}

#if VOLUME_Z_FAST
namespace axis_packets {
// A warp follows one detector column along contiguous z samples.
constexpr int GU=8, GV=32;
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
extern "C" __global__ void __launch_bounds__(GU * GV) cone_forward_z_fast(
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

} // namespace axis_packets
#else
namespace dense_packets {
constexpr int GU = 8, GV = 4;
// Per-ray Joseph sum for one pixel, any view.
// The smallest and largest of a value over the (whole) warp.
// Planes k0..k1 of a ray, stepped by the warp through first..last (see ray_sum): plane
// k lies sa apart, in-plane axes b and c sb and sc. Axis 0 marches contiguously;
// the other axes have contiguous in-plane x taps.
template <typename Offset, bool AXIS_X>
__device__ __forceinline__ float plane_sum(
    const float* __restrict__ vol, int k0, int k1, int first, int last, float Sa, float Sb,
    float Sc, float inv, float rb, float rc, int nb, int nc, long sa, long sb, long sc = 1)
{
    if constexpr (AXIS_X) sa = 1;
    else sb = 1;
    float sum = 0.f;
    int pm = AXIS_X ? min(8, (int)sb & -(int)sb) - 1 : 0;
    int phase = threadIdx.x & pm;
    Offset plane = (Offset)(first - phase) * (Offset)sa;
    for (int step = first; step <= last + pm; ++step, plane += (Offset)sa) {
        int k = step - phase;
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
        Offset p = plane + (Offset)b0 * (Offset)sb + (Offset)c0 * (Offset)sc;
        float s0 = 0.f, s1 = 0.f;
        if (b0in && c0in) s0 = __ldg(vol + p) * wc0;
        if (b0in && c1in) s0 += __ldg(vol + (Offset)(p + (Offset)sc)) * wc1;
        if (b1in && c0in) s1 = __ldg(vol + (Offset)(p + (Offset)sb)) * wc0;
        if (b1in && c1in) s1 += __ldg(vol + (Offset)(p + (Offset)sb + (Offset)sc)) * wc1;
        sum += wb0 * s0 + wb1 * s1;
    }
    return sum;
}

// Called by every lane of a warp (``live`` false for lanes without a ray). The lanes
// step through the planes together, each sampling only its own ray's planes: rays
// entering through the volume's top or bottom start at different planes, and lanes
// left on different planes would not share cache lines.
template <typename Offset>
__device__ float ray_sum(const float* __restrict__ vol, const float* c, float fu, float fv,
                         int nx, int ny, int nz, bool live)
{
    float r0, r1, r2; ray_of(c, fu, fv, r0, r1, r2);
    int a = ray_axis(r0, r1, r2), b = a == 0 ? 1 : 0, cc = a == 2 ? 1 : 2;
    int na = SEL(a, nx, ny, nz), nb = SEL(b, nx, ny, nz), nc = SEL(cc, nx, ny, nz);
    long s0 = VOLUME_Z_FAST ? (long)ny * nz : 1;
    long s1 = VOLUME_Z_FAST ? nz : nx, s2 = VOLUME_Z_FAST ? 1 : (long)nx * ny;
    long sa = SEL(a, s0, s1, s2), sb = SEL(b, s0, s1, s2), sc = SEL(cc, s0, s1, s2);
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
    if (a == 0) return plane_sum<Offset, true>(vol, k0, k1, first, last, Sa, Sb, Sc, inv, rb, rc, nb, nc, sa, sb, sc);
    return plane_sum<Offset, false>(vol, k0, k1, first, last, Sa, Sb, Sc, inv, rb, rc, nb, nc, sa, sb, sc);
}

template <typename Offset>
__device__ float register_ray_sum(const float* __restrict__ vol,const float* c,float fu,float fv,
    int nx,int ny,int nz,bool live)
{
    float r0,r1,r2;ray_of(c,fu,fv,r0,r1,r2);
    int a=ray_axis(r0,r1,r2);
    if(!__all_sync(0xffffffffu,!live || a==0))return ray_sum<Offset>(vol,c,fu,fv,nx,ny,nz,live);
    float inv=__frcp_rn(r0),rb=r1,rc=r2;
    float lo=-1e30f,hi=1e30f;
    float slope=rb*inv,base=c[1]-c[0]*slope;
    if(fabsf(slope)<1e-12f){if(base<=-1.f || base>=(float)ny)live=false;}
    else{float k1=(-1.f-base)/slope,k2=((float)ny-base)/slope;
        lo=fmaxf(lo,fminf(k1,k2));hi=fminf(hi,fmaxf(k1,k2));}
    slope=rc*inv;base=c[2]-c[0]*slope;
    if(fabsf(slope)<1e-12f){if(base<=-1.f || base>=(float)nz)live=false;}
    else{float k1=(-1.f-base)/slope,k2=((float)nz-base)/slope;
        lo=fmaxf(lo,fminf(k1,k2));hi=fminf(hi,fmaxf(k1,k2));}
    int k0=max((int)floorf(lo),0),k1=min((int)ceilf(hi),nx-1);
    if(!live){k0=nx;k1=-1;}
    constexpr int GROUP=1;
    int lane=threadIdx.x%GROUP;
    constexpr unsigned mask=0xffffffffu;
    float inverses[GROUP],rbs[GROUP],rcs[GROUP];
    float sas[GROUP],sbs[GROUP],scs[GROUP];
    int lower[GROUP],upper[GROUP];
    #pragma unroll
    for(int p=0;p<GROUP;++p){
        inverses[p]=__shfl_sync(mask,inv,p,GROUP);
        sas[p]=__shfl_sync(mask,c[0],p,GROUP);
        sbs[p]=__shfl_sync(mask,c[1],p,GROUP);
        scs[p]=__shfl_sync(mask,c[2],p,GROUP);
        rbs[p]=__shfl_sync(mask,rb,p,GROUP);
        rcs[p]=__shfl_sync(mask,rc,p,GROUP);
        lower[p]=__shfl_sync(mask,k0,p,GROUP);
        upper[p]=__shfl_sync(mask,k1,p,GROUP);
    }
    int first=warp_min(k0),last=warp_max(k1);
    float sum=0.f;
    int pm=min(8,nx & -nx)-1;
    for(int base=first;base<=last+pm;base+=GROUP){
        float values[GROUP];
        #pragma unroll
        for(int p=0;p<GROUP;++p){
            // p is the destination ray owner, lane is its sample producer.
            int owner=(threadIdx.x/GROUP)*GROUP+p;
            int k=base+lane-(owner & pm);
            float value=0.f;
            if(k>=lower[p] && k<=upper[p]){
                float t=__fmul_rn(__fsub_rn((float)k,sas[p]),inverses[p]);
                float fb=__fmaf_rn(t,rbs[p],sbs[p]),fc=__fmaf_rn(t,rcs[p],scs[p]);
                float fb0=floorf(fb),fc0=floorf(fc);
                int b0=(int)fb0,c0=(int)fc0;
                float wb1=fb-fb0,wc1=fc-fc0,wb0=1.f-wb1,wc0=1.f-wc1;
                bool b0in=(unsigned)b0<(unsigned)ny,b1in=(unsigned)(b0+1)<(unsigned)ny;
                bool c0in=(unsigned)c0<(unsigned)nz,c1in=(unsigned)(c0+1)<(unsigned)nz;
                Offset point=((Offset)c0*(Offset)ny+(Offset)b0)*(Offset)nx+(Offset)k;
                float s0=0.f,s1=0.f;
                if(b0in && c0in)s0=__ldg(vol+point)*wc0;
                if(b0in && c1in)s0+=__ldg(vol+(Offset)(point+(Offset)nx*(Offset)ny))*wc1;
                if(b1in && c0in)s1=__ldg(vol+(Offset)(point+(Offset)nx))*wc0;
                if(b1in && c1in)s1+=__ldg(vol+(Offset)(point+(Offset)nx+(Offset)nx*(Offset)ny))*wc1;
                value=wb0*s0+wb1*s1;
            }
            values[p]=value;
        }
        // Transpose the GROUP x GROUP sample/ray matrix in registers.
        // Each butterfly exchanges one register bit with its lane bit.
        #pragma unroll
        for(int bit=1;bit<GROUP;bit<<=1){
            #pragma unroll
            for(int p=0;p<GROUP;++p)if(!(p&bit)){
                float send=(lane&bit)?values[p]:values[p|bit];
                float receive=__shfl_xor_sync(mask,send,bit,GROUP);
                if(lane&bit)values[p]=receive;else values[p|bit]=receive;
            }
        }
        #pragma unroll
        for(int q=0;q<GROUP;++q)sum=__fadd_rn(sum,values[q]);
    }
    return sum;
}

// Align packet centers only by relabelling which original pixel each lane owns.
// Projection arithmetic in the sampler remains untouched.
__device__ __forceinline__ void projected_center(const float* c, int nx, int ny, int nz,
    float& u, float& v)
{
    float dx=.5f*(nx-1)-c[0],dy=.5f*(ny-1)-c[1],dz=.5f*(nz-1)-c[2];
    float den=c[13]*dx+c[14]*dy+c[15]*dz;
    float lam=c[16]/den;
    u=c[23]+lam*(c[17]*dx+c[18]*dy+c[19]*dz);
    v=c[24]+lam*(c[20]*dx+c[21]*dy+c[22]*dz);
}
__device__ __forceinline__ int pixel_shift(float own,float anchor,int size)
{
    float delta=own-anchor;
    // A center on the source plane gives no useful scheduling proposal.
    // Zero is still an ordinary bijective pixel relabelling.
    return isfinite(delta)?(int)rintf(remainderf(delta,(float)size)):0;
}
__device__ __forceinline__ int wrap_pixel(int pixel,int shift,int size)
{
    int result=pixel+shift;
    if(result<0)result+=size;
    if(result>=size)result-=size;
    return result;
}

// Every subgroup owns matched center-relative pixels across consecutive views.
template <typename Offset>
__device__ __forceinline__ void cone_forward_body(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz, int nviews)
{
    if (!dense_view_packet(coeff, blockIdx.x * 8, nviews, nx, ny, nz)) return;
    constexpr int GROUP=8;
    int firstview=blockIdx.x*GROUP;
    int view=firstview+threadIdx.x%GROUP;
    const float* cfirst=coeff+(long)firstview*NC;
    const float* c=coeff+(long)min(view,nviews-1)*NC;
    // One uniform pixel orientation for every view in this tile; each ray still
    // derives its true major axis and source independently in the helper.
    float cr0,cr1,cr2;
    ray_of(cfirst,(float)(blockIdx.y*GU+GU/2),
        (float)(blockIdx.z*GV+GV/2),cr0,cr1,cr2);
    int a=ray_axis(cr0,cr1,cr2),fast=a==0?1:0;
    float ratio=__fdividef(SEL(fast,cr0,cr1,cr2),SEL(a,cr0,cr1,cr2));
    float du=comp(cfirst+6,fast)-ratio*comp(cfirst+6,a);
    float dv=comp(cfirst+9,fast)-ratio*comp(cfirst+9,a);
    bool ufast=fabsf(du)>fabsf(dv);
    int ray=threadIdx.x/GROUP;
    int u=blockIdx.y*GU+(ufast?ray%GU:ray/GV);
    int v=blockIdx.z*GV+(ufast?ray/GU:ray%GV);
    bool live=view<nviews && u<nu && v<nv;
    float anchoru,anchorv,ownu,ownv;
    projected_center(cfirst,nx,ny,nz,anchoru,anchorv);
    projected_center(c,nx,ny,nz,ownu,ownv);
    u=wrap_pixel(u,pixel_shift(ownu,anchoru,nu),nu);
    v=wrap_pixel(v,pixel_shift(ownv,anchorv,nv),nv);
    float fu=(float)u,fv=(float)v;
    // All lanes, including dead final-view producers, participate in the vote
    // and every collective. The vote and fallback have no CTA barriers.
    float sum=register_ray_sum<Offset>(vol,c,fu,fv,nx,ny,nz,live);
    if(!live)return;
    float r0,r1,r2;ray_of(c,fu,fv,r0,r1,r2);
    float ra=SEL(ray_axis(r0,r1,r2),r0,r1,r2);
    out[((long)view*nv+v)*nu+u]=sum*path_w(r0,r1,r2,ra,sx,sy,sz);
}

extern "C" __global__ void __launch_bounds__(GU * GV * 8) cone_forward_dense(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz, int nviews)
{
    cone_forward_body<unsigned>(coeff,vol,out,nx,ny,nz,nu,nv,sx,sy,sz,nviews);
}

extern "C" __global__ void __launch_bounds__(GU * GV * 8) cone_forward_dense64(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz, int nviews)
{
    cone_forward_body<unsigned long long>(coeff,vol,out,nx,ny,nz,nu,nv,sx,sy,sz,nviews);
}

} // namespace dense_packets


namespace sparse_packets {
constexpr int GU = 16, GV = 16;
// Per-ray Joseph sum for one pixel, any view.
// The smallest and largest of a value over the (whole) warp.
// Planes k0..k1 of a ray, stepped by the warp through first..last (see ray_sum): plane
// k lies sa apart, in-plane axes b and c sb and sc. Axis 0 marches contiguously;
// the other axes have contiguous in-plane x taps.
template <typename Offset, bool AXIS_X>
__device__ __forceinline__ float plane_sum(
    const float* __restrict__ vol, int k0, int k1, int first, int last, float Sa, float Sb,
    float Sc, float inv, float rb, float rc, int nb, int nc, long sa, long sb, long sc = 1)
{
    if constexpr (AXIS_X) sa = 1;
    else sb = 1;
    float sum = 0.f;
    Offset plane = (Offset)first * (Offset)sa;
    for (int k = first; k <= last; ++k, plane += (Offset)sa) {
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
        Offset p = plane + (Offset)b0 * (Offset)sb + (Offset)c0 * (Offset)sc;
        float s0 = 0.f, s1 = 0.f;
        if (b0in && c0in) s0 = __ldg(vol + p) * wc0;
        if (b0in && c1in) s0 += __ldg(vol + (Offset)(p + (Offset)sc)) * wc1;
        if (b1in && c0in) s1 = __ldg(vol + (Offset)(p + (Offset)sb)) * wc0;
        if (b1in && c1in) s1 += __ldg(vol + (Offset)(p + (Offset)sb + (Offset)sc)) * wc1;
        sum += wb0 * s0 + wb1 * s1;
    }
    return sum;
}

// Called by every lane of a warp (``live`` false for lanes without a ray). The lanes
// step through the planes together, each sampling only its own ray's planes: rays
// entering through the volume's top or bottom start at different planes, and lanes
// left on different planes would not share cache lines.
template <typename Offset>
__device__ float ray_sum(const float* __restrict__ vol, const float* c, float fu, float fv,
                         int nx, int ny, int nz, bool live)
{
    float r0, r1, r2; ray_of(c, fu, fv, r0, r1, r2);
    int a = ray_axis(r0, r1, r2), b = a == 0 ? 1 : 0, cc = a == 2 ? 1 : 2;
    int na = SEL(a, nx, ny, nz), nb = SEL(b, nx, ny, nz), nc = SEL(cc, nx, ny, nz);
    long s0 = VOLUME_Z_FAST ? (long)ny * nz : 1;
    long s1 = VOLUME_Z_FAST ? nz : nx, s2 = VOLUME_Z_FAST ? 1 : (long)nx * ny;
    long sa = SEL(a, s0, s1, s2), sb = SEL(b, s0, s1, s2), sc = SEL(cc, s0, s1, s2);
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
    if (a == 0) return plane_sum<Offset, true>(vol, k0, k1, first, last, Sa, Sb, Sc, inv, rb, rc, nb, nc, sa, sb, sc);
    return plane_sum<Offset, false>(vol, k0, k1, first, last, Sa, Sb, Sc, inv, rb, rc, nb, nc, sa, sb, sc);
}

template <typename Offset>
__device__ float register_ray_sum(const float* __restrict__ vol,const float* c,float fu,float fv,
    int nx,int ny,int nz,bool live)
{
    float r0,r1,r2;ray_of(c,fu,fv,r0,r1,r2);
    int a=ray_axis(r0,r1,r2);
    if(!__syncthreads_and(!live || a==0))return ray_sum<Offset>(vol,c,fu,fv,nx,ny,nz,live);
    float inv=__frcp_rn(r0),rb=r1,rc=r2;
    float lo=-1e30f,hi=1e30f;
    float slope=rb*inv,base=c[1]-c[0]*slope;
    if(fabsf(slope)<1e-12f){if(base<=-1.f || base>=(float)ny)live=false;}
    else{float k1=(-1.f-base)/slope,k2=((float)ny-base)/slope;
        lo=fmaxf(lo,fminf(k1,k2));hi=fminf(hi,fmaxf(k1,k2));}
    slope=rc*inv;base=c[2]-c[0]*slope;
    if(fabsf(slope)<1e-12f){if(base<=-1.f || base>=(float)nz)live=false;}
    else{float k1=(-1.f-base)/slope,k2=((float)nz-base)/slope;
        lo=fmaxf(lo,fminf(k1,k2));hi=fminf(hi,fmaxf(k1,k2));}
    int k0=max((int)floorf(lo),0),k1=min((int)ceilf(hi),nx-1);
    if(!live){k0=nx;k1=-1;}
    constexpr int GROUP=8;
    int lane=threadIdx.x%GROUP;
    constexpr unsigned mask=0xffffffffu;
    float inverses[GROUP],rbs[GROUP],rcs[GROUP];
    int lower[GROUP],upper[GROUP];
    #pragma unroll
    for(int p=0;p<GROUP;++p){
        inverses[p]=__shfl_sync(mask,inv,p,GROUP);
        rbs[p]=__shfl_sync(mask,rb,p,GROUP);
        rcs[p]=__shfl_sync(mask,rc,p,GROUP);
        lower[p]=__shfl_sync(mask,k0,p,GROUP);
        upper[p]=__shfl_sync(mask,k1,p,GROUP);
    }
    int first=warp_min(k0),last=warp_max(k1);
    float sum=0.f;
    for(int base=first;base<=last;base+=GROUP){
        int k=base+lane;
        float values[GROUP];
        #pragma unroll
        for(int p=0;p<GROUP;++p){
            float value=0.f;
            if(k>=lower[p] && k<=upper[p]){
                float t=__fmul_rn(__fsub_rn((float)k,c[0]),inverses[p]);
                float fb=__fmaf_rn(t,rbs[p],c[1]),fc=__fmaf_rn(t,rcs[p],c[2]);
                float fb0=floorf(fb),fc0=floorf(fc);
                int b0=(int)fb0,c0=(int)fc0;
                float wb1=fb-fb0,wc1=fc-fc0,wb0=1.f-wb1,wc0=1.f-wc1;
                bool b0in=(unsigned)b0<(unsigned)ny,b1in=(unsigned)(b0+1)<(unsigned)ny;
                bool c0in=(unsigned)c0<(unsigned)nz,c1in=(unsigned)(c0+1)<(unsigned)nz;
                Offset point=((Offset)c0*(Offset)ny+(Offset)b0)*(Offset)nx+(Offset)k;
                float s0=0.f,s1=0.f;
                if(b0in && c0in)s0=__ldg(vol+point)*wc0;
                if(b0in && c1in)s0+=__ldg(vol+(Offset)(point+(Offset)nx*(Offset)ny))*wc1;
                if(b1in && c0in)s1=__ldg(vol+(Offset)(point+(Offset)nx))*wc0;
                if(b1in && c1in)s1+=__ldg(vol+(Offset)(point+(Offset)nx+(Offset)nx*(Offset)ny))*wc1;
                value=wb0*s0+wb1*s1;
            }
            values[p]=value;
        }
        // Transpose the GROUP x GROUP sample/ray matrix in registers.
        // Each butterfly exchanges one register bit with its lane bit.
        #pragma unroll
        for(int bit=1;bit<GROUP;bit<<=1){
            #pragma unroll
            for(int p=0;p<GROUP;++p)if(!(p&bit)){
                float send=(lane&bit)?values[p]:values[p|bit];
                float receive=__shfl_xor_sync(mask,send,bit,GROUP);
                if(lane&bit)values[p]=receive;else values[p|bit]=receive;
            }
        }
        #pragma unroll
        for(int q=0;q<GROUP;++q)sum=__fadd_rn(sum,values[q]);
    }
    return sum;
}

// One ray per thread: grid (views, u tiles, v tiles); out (view, v, u).
template <typename Offset>
__device__ __forceinline__ void cone_forward_body(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz, int nviews)
{
    if (dense_view_packet(coeff, (blockIdx.x / 8) * 8, nviews, nx, ny, nz)) return;
    int view = blockIdx.x;
    const float* c = coeff + (long)view * NC;
    // Coalesce along the fastest available physical in-plane axis. Its
    // detector direction can be u or v for rolled/tilted arrangements.
    float cr0, cr1, cr2;
    ray_of(c, (float)(blockIdx.y * GU + GU / 2),
           (float)(blockIdx.z * GV + GV / 2), cr0, cr1, cr2);
    int a = ray_axis(cr0, cr1, cr2), fast = a == 0 ? 1 : 0;
    float ratio = __fdividef(SEL(fast, cr0, cr1, cr2), SEL(a, cr0, cr1, cr2));
    float du = comp(c + 6, fast) - ratio * comp(c + 6, a);
    float dv = comp(c + 9, fast) - ratio * comp(c + 9, a);
    bool ufast = fabsf(du) > fabsf(dv);
    constexpr int GROUP=8;
    int ray=threadIdx.x;
    if(a==0){
        constexpr int FAST_RAYS=32/GROUP;
        int width=ufast?GU:GV,warp=threadIdx.x/32;
        int fast=(warp%(width/FAST_RAYS))*FAST_RAYS+(threadIdx.x%32)/GROUP;
        int slow=(warp/(width/FAST_RAYS))*GROUP+threadIdx.x%GROUP;
        ray=slow*width+fast;
    }
    int u = blockIdx.y * GU + (ufast ? ray % GU : ray / GV);
    int v = blockIdx.z * GV + (ufast ? ray / GU : ray % GV);
    bool live = u < nu && v < nv;
    float fu = (float)u, fv = (float)v;
    float sum = a==0?register_ray_sum<Offset>(vol,c,fu,fv,nx,ny,nz,live):ray_sum<Offset>(vol,c,fu,fv,nx,ny,nz,live);
    if (!live) return;
    float r0, r1, r2; ray_of(c, fu, fv, r0, r1, r2);
    float ra = SEL(ray_axis(r0, r1, r2), r0, r1, r2);
    // Written (view, v, u), as JAX holds projections: one store per ray, so the
    // stride costs little, and no transposed copy of the projections is needed.
    out[((long)view * nv + v) * nu + u] = sum * path_w(r0, r1, r2, ra, sx, sy, sz);
}

extern "C" __global__ void __launch_bounds__(GU * GV) cone_forward_sparse(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz, int nviews)
{
    cone_forward_body<unsigned>(coeff,vol,out,nx,ny,nz,nu,nv,sx,sy,sz,nviews);
}

extern "C" __global__ void __launch_bounds__(GU * GV) cone_forward_sparse64(
    const float* __restrict__ coeff, const float* __restrict__ vol, float* __restrict__ out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz, int nviews)
{
    cone_forward_body<unsigned long long>(coeff,vol,out,nx,ny,nz,nu,nv,sx,sy,sz,nviews);
}

} // namespace sparse_packets

#endif

// Uniform-volume row normalization: exact boundary taps, counted interior planes.

__device__ __forceinline__ bool box_interior(int k, float Sa, float Sb, float Sc, float inv, float rb, float rc, int nb, int nc) {
    float t=__fmul_rn(__fsub_rn((float)k, Sa), inv);
    float fb=__fmaf_rn(t, rb, Sb), fc=__fmaf_rn(t, rc, Sc);
    return fb>=0.f && fb<=(float)(nb-1) && fc>=0.f && fc<=(float)(nc-1);
}
__device__ __forceinline__ float box_boundary(int k, float Sa, float Sb, float Sc, float inv, float rb, float rc, int nb, int nc) {
    float t=__fmul_rn(__fsub_rn((float)k, Sa), inv);
    float fb=__fmaf_rn(t, rb, Sb), fc=__fmaf_rn(t, rc, Sc);
    float fb0=floorf(fb), fc0=floorf(fc);
    int b0=(int)fb0, c0=(int)fc0;
    float wb1=fb-fb0, wc1=fc-fc0, wb0=1.f-wb1, wc0=1.f-wc1;
    float s0=0.f, s1=0.f;
    if ((unsigned)b0<(unsigned)nb && (unsigned)c0<(unsigned)nc)s0=wc0;
    if ((unsigned)b0<(unsigned)nb && (unsigned)(c0+1)<(unsigned)nc)s0+=wc1;
    if ((unsigned)(b0+1)<(unsigned)nb && (unsigned)c0<(unsigned)nc)s1=wc0;
    if ((unsigned)(b0+1)<(unsigned)nb && (unsigned)(c0+1)<(unsigned)nc)s1+=wc1;
    return wb0*s0+wb1*s1;
}
__device__ float box_sum(const float* c, float fu, float fv, int nx, int ny, int nz) {
    float r0, r1, r2;
    ray_of(c, fu, fv, r0, r1, r2);
    int a=ray_axis(r0, r1, r2), b=a==0?1:0, cc=a==2?1:2;
    int na=SEL(a, nx, ny, nz), nb=SEL(b, nx, ny, nz), nc=SEL(cc, nx, ny, nz);
    float ra=SEL(a, r0, r1, r2), rb=SEL(b, r0, r1, r2), rc=SEL(cc, r0, r1, r2);
    float Sa=c[a], Sb=c[b], Sc=c[cc], inv=__frcp_rn(ra);
    // Retain ordinary FP's outer plane bounds.
    float lo=-1e30f, hi=1e30f;
    float slope=rb*inv, base=Sb-Sa*slope;
    if (fabsf(slope)<1e-12f) {
        if (base<=-1.f||base>=(float)nb)return 0.f;
    } else {
        float k1=(-1.f-base)/slope, k2=((float)nb-base)/slope;
        lo=fmaxf(lo, fminf(k1, k2));
        hi=fminf(hi, fmaxf(k1, k2));
    }
    slope=rc*inv;
    base=Sc-Sa*slope;
    if (fabsf(slope)<1e-12f) {
        if (base<=-1.f||base>=(float)nc)return 0.f;
    } else {
        float k1=(-1.f-base)/slope, k2=((float)nc-base)/slope;
        lo=fmaxf(lo, fminf(k1, k2));
        hi=fminf(hi, fmaxf(k1, k2));
    }
    int k0=max((int)floorf(fminf(fmaxf(lo, -1.f), (float)na)), 0);
    int k1=min((int)ceilf(fminf(fmaxf(hi, -1.f), (float)na)), na-1);
    if (k0>k1)return 0.f;
    // Real-affine bounds only propose an interior interval. Verify both endpoints
    // with the exact ordinary-FP rounding; monotonicity makes every middle point safe.
    double ilo=k0, ihi=k1;
    double ds=(double)rb*(double)inv, db=(double)Sb-(double)Sa*ds;
    if (ds==0.) {
        if (db<0.||db>(double)(nb-1)) {
            ilo=1.;
            ihi=0.;
        }
    } else {
        double p=-db/ds, q=((double)(nb-1)-db)/ds;
        ilo=fmax(ilo, fmin(p, q));
        ihi=fmin(ihi, fmax(p, q));
    }
    ds=(double)rc*(double)inv;
    db=(double)Sc-(double)Sa*ds;
    if (ds==0.) {
        if (db<0.||db>(double)(nc-1)) {
            ilo=1.;
            ihi=0.;
        }
    } else {
        double p=-db/ds, q=((double)(nc-1)-db)/ds;
        ilo=fmax(ilo, fmin(p, q));
        ihi=fmin(ihi, fmax(p, q));
    }
    int first=(int)ceil(fmin(fmax(ilo, (double)k0), (double)k1+1.));
    int last=(int)floor(fmin(fmax(ihi, (double)k0-1.), (double)k1));
    if (ilo>ihi) {
        first=k1+1;
        last=k1;
    }
    while (first<=last&&!box_interior(first, Sa, Sb, Sc, inv, rb, rc, nb, nc))++first;
    while (first<=last&&!box_interior(last, Sa, Sb, Sc, inv, rb, rc, nb, nc))--last;
    float sum=0.f;
    if (first>last) {
        for (int k=k0;k<=k1;++k)sum+=box_boundary(k, Sa, Sb, Sc, inv, rb, rc, nb, nc);
        return sum;
    }
    for (int k=k0;k<first;++k)sum+=box_boundary(k, Sa, Sb, Sc, inv, rb, rc, nb, nc);
    sum+=(float)(last-first+1);
    for (int k=last+1;k<=k1;++k)sum+=box_boundary(k, Sa, Sb, Sc, inv, rb, rc, nb, nc);
    return sum;
}
extern "C" __global__ void cone_box_forward(const float* coeff, float* out, int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz) {
    int j=blockIdx.y*blockDim.x+threadIdx.x;
    if (j>=nu*nv)return;
    int u=j%nu, v=j/nu;
    const float* c=coeff+(long)blockIdx.x*NC;
    float sum=box_sum(c, (float)u, (float)v, nx, ny, nz);
    float r0, r1, r2;
    ray_of(c, (float)u, (float)v, r0, r1, r2);
    float ra=SEL(ray_axis(r0, r1, r2), r0, r1, r2);
    out[(long)blockIdx.x*nu*nv+j]=sum*path_w(r0, r1, r2, ra, sx, sy, sz);
}

// Weight and transpose (view, v, u) images in coalesced 32x32 tiles,
// recording exact nonzero pixel bounds without reading images/rays a second time.
// Views occupy grid x so their count is not limited by CUDA grid z.
template <int RHS>
__device__ __forceinline__ void weight_images_boxes(
    const float* __restrict__ coeff, const float* __restrict__ img, const float* __restrict__ img1, float* __restrict__ out,
    int nviews, int nu, int nv, float sx, float sy, float sz, int* __restrict__ partial)
{
    __shared__ float tile[RHS][32][33];
    __shared__ int warp_boxes[8][12];
    int view = blockIdx.x;
    const float* c = coeff + (long)view * NC;
    int u = blockIdx.y * 32 + threadIdx.x;
    int v0 = blockIdx.z * 32 + threadIdx.y;
    int lo[3][2], hi[3][2];
    #pragma unroll
    for (int a = 0; a < 3; ++a) { lo[a][0] = nu; hi[a][0] = -1; lo[a][1] = nv; hi[a][1] = -1; }
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int v = v0 + j;
        if (u < nu && v < nv) {
            float r0, r1, r2; ray_of(c, (float)u, (float)v, r0, r1, r2);
            int a = ray_axis(r0, r1, r2);
            float ra = SEL(a, r0, r1, r2);
            float weight = path_w(r0, r1, r2, ra, sx, sy, sz);
            long pixel = ((long)view * nv + v) * nu + u;
            float value = img[pixel] * weight;
            float value1 = RHS == 2 ? img1[pixel] * weight : 0.f;
            tile[0][threadIdx.y + j][threadIdx.x] = value;
            if (RHS == 2) tile[1][threadIdx.y + j][threadIdx.x] = value1;
            // Exact-zero contributions need no footprint, including shifted-batch padding.
            if (value != 0.f || value1 != 0.f) {
                #pragma unroll
                for (int q = 0; q < 3; ++q) if (q == a) {
                    lo[q][0] = min(lo[q][0], u); hi[q][0] = max(hi[q][0], u);
                    lo[q][1] = min(lo[q][1], v); hi[q][1] = max(hi[q][1], v);
                }
            }
        }
    }
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        int ulo = warp_min(lo[a][0]), uhi = warp_max(hi[a][0]);
        int vlo = warp_min(lo[a][1]), vhi = warp_max(hi[a][1]);
        if (threadIdx.x == 0) {
            warp_boxes[threadIdx.y][a*4] = ulo; warp_boxes[threadIdx.y][a*4+1] = uhi;
            warp_boxes[threadIdx.y][a*4+2] = vlo; warp_boxes[threadIdx.y][a*4+3] = vhi;
        }
    }
    __syncthreads();
    int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 12) {
        int value = warp_boxes[0][tid];
        bool lower = (tid % 4) == 0 || (tid % 4) == 2;
        #pragma unroll
        for (int w = 1; w < 8; ++w) value = lower ? min(value, warp_boxes[w][tid]) : max(value, warp_boxes[w][tid]);
        long id = ((long)view * gridDim.z + blockIdx.z) * gridDim.y + blockIdx.y;
        partial[id * 12 + tid] = value;
    }
    int vo = blockIdx.z * 32 + threadIdx.x;
    int u0 = blockIdx.y * 32 + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int uo = u0 + j;
        if (uo < nu && vo < nv) {
            long pixel = ((long)view * nu + uo) * nv + vo;
            float value = tile[0][threadIdx.x][threadIdx.y + j];
            if (RHS == 2)
                out[pixel + (long)nviews * nu * nv] = tile[1][threadIdx.x][threadIdx.y + j];
            out[pixel] = value;
        }
    }
}
extern "C" __global__ void weight_images_boxes_tiled(
    const float* __restrict__ coeff, const float* __restrict__ img, float* __restrict__ out,
    int nviews, int nu, int nv, float sx, float sy, float sz, int* __restrict__ partial)
{
    weight_images_boxes<1>(coeff, img, nullptr, out, nviews, nu, nv, sx, sy, sz, partial);
}
extern "C" __global__ void pair_weight_images_boxes_tiled(
    const float* __restrict__ coeff, const float* __restrict__ img, const float* __restrict__ img1, float* __restrict__ out,
    int nviews, int nu, int nv, float sx, float sy, float sz, int* __restrict__ partial)
{
    weight_images_boxes<2>(coeff, img, img1, out, nviews, nu, nv, sx, sy, sz, partial);
}
extern "C" __global__ void reduce_axis_boxes_tiled(
    const int* __restrict__ partial, int* __restrict__ boxes, int ntiles, int nu, int nv)
{
    int q = threadIdx.y;
    bool lower = (q % 4) == 0 || (q % 4) == 2;
    int value = lower ? ((q % 4) == 0 ? nu : nv) : -1;
    for (int t = threadIdx.x; t < ntiles; t += 32) {
        int item = partial[((long)blockIdx.x * ntiles + t) * 12 + q];
        value = lower ? min(value, item) : max(value, item);
    }
    value = lower ? warp_min(value) : warp_max(value);
    if (threadIdx.x == 0) boxes[(long)blockIdx.x * 12 + q] = value;
}

// Separable views, the columns whose plane axis is a: grid (z tiles of ZT planes, row
// tiles, planes). img is path-weighted; axis_box holds each plane axis's nonzero pixel
// bounds. Every voxel's sum runs in the same order whatever ZT is.
template <int ZT>
__device__ __forceinline__ void sep_adjoint_tile(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol,
    int nx, int ny, int nz, int nu, int nv)
{
    __shared__ float R[ULEN][ZT + 1];
    __shared__ float s_fb[ULEN], s_t[ULEN], s_rc0[ULEN], s_f0[ULEN], s_idf[ULEN];
    __shared__ int s_taps[ULEN];
    __shared__ float s_margin[ULEN];
    __shared__ float cf[NC];
    __shared__ int skip;
    int b = 1 - a;
    int nb = a == 0 ? ny : nx;
    int c0 = (VOLUME_Z_FAST ? blockIdx.x : blockIdx.z) * ZT;
    int b0 = (VOLUME_Z_FAST ? blockIdx.y : blockIdx.x) * TB;
    int k = VOLUME_Z_FAST ? blockIdx.z : blockIdx.y;
    int tid = threadIdx.x;
    int my_b = b0 + tid / 4, my_c = (tid % 4) * (ZT / 4);
    float my_bf = (float)my_b;
    if (!any_view(coeff, axis_box, nviews, a, true)) return;
    float acc[ZT / 4];
    #pragma unroll
    for (int j = 0; j < ZT / 4; ++j) acc[j] = 0.f;
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
                float margin = vertical_support_margin(idf, Sc, t, rc0, DVc, nz, nv);
                s_margin[i] = margin;
                // Widen both endpoints, keeping the tap count warp-uniform.
                float taps = fminf(2.f * (fabsf(idf) + margin), (float)nv);
                s_taps[i] = col_axis == a ? min((int)floorf(taps) + 1, nv) : 0;
            }
            __syncthreads();
            for (int i = tid / ZT, j = tid % ZT; i < ulen; i += blockDim.x / ZT) {
                float jcf = (float)(c0 + j);
                float t = s_t[i], rc0 = s_rc0[i];
                float v1 = (jcf - 1.f - s_f0[i]) * s_idf[i];
                float v2 = (jcf + 1.f - s_f0[i]) * s_idf[i];
                float lower = fminf(v1, v2);
                int va = max((int)floorf(pixel_bound(lower - s_margin[i], nv)) + 1, 0);
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
                    for (int j = 0; j < ZT / 4; ++j) acc[j] += wb * R[i][my_c + j];
                }
            }
        }
    }
    // R is no longer read by the accumulation phase. Reuse it to
    // exchange the register-z owners for contiguous-x output lanes.
    if (!VOLUME_Z_FAST && a == 1) {
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < ZT / 4; ++j) {
            int offset = j;
            R[tid / 4][my_c + offset] = acc[j];
        }
        __syncthreads();
        for (int i = tid; i < TB * ZT; i += blockDim.x) {
            int jb = i % TB, jc = i / TB;
            if (b0 + jb < nx && c0 + jc < nz && k < ny)
                vol[((long)(c0 + jc) * ny + k) * nx + b0 + jb] += R[jb][jc];
        }
        return;
    }
    int na = a == 0 ? nx : ny;
    if (my_b < nb && k < na) {
        int ix = a == 0 ? k : my_b, iy = a == 0 ? my_b : k;
        long base = volume_index(ix, iy, c0 + my_c, nx, ny, nz);
        #pragma unroll
        for (int j = 0; j < ZT / 4; ++j) if (c0 + my_c + j < nz) vol[base + j * (VOLUME_Z_FAST ? 1L : (long)nx * ny)] += acc[j];
    }
}

extern "C" __global__ void sep_adjoint(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol,
    int nx, int ny, int nz, int nu, int nv)
{
    sep_adjoint_tile<TC>(coeff, axis_box, nviews, a, img, vol, nx, ny, nz, nu, nv);
}

// At most 32 planes: one 32-plane tile instead of a half-idle 64-plane one.
extern "C" __global__ void sep_adjoint_thin(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol,
    int nx, int ny, int nz, int nu, int nv)
{
    sep_adjoint_tile<32>(coeff, axis_box, nviews, a, img, vol, nx, ny, nz, nu, nv);
}

// Adds x0..x3 to four padded shared cells. Pair the two rows of each column:
// moving one column retains both right-hand cells, so their flush is all zero.
__device__ __forceinline__ void add4(float* p0, float x0, float* p1, float x1, float* p2,
                                     float x2, float* p3, float x3)
{
    if (x0 != 0.f || x2 != 0.f) {
        atomicAdd(p0, x0); atomicAdd(p2, x2);
    }
    if (x1 != 0.f || x3 != 0.f) {
        atomicAdd(p1, x1); atomicAdd(p3, x3);
    }
}

// Non-separable views, rays sampled along axis a: grid (c tiles, b tiles, planes k).
// Each block owns contiguous plane tiles. Each view's footprint on their union is
// found by a thread of its own, and the views' runs of rows of one detector column
// are then walked as one list, without a barrier per view. Each run's crossings of a plane
// step monotonically through the tile, so its thread keeps the 2 x 2 cells around the
// current crossing in registers and adds them to the tile only as it moves on. img is
// path-weighted; nviews <= VMAX.
#define VMAX 32
template <int PC, int RHS = 1, int PLANES = 1>
__device__ __forceinline__ void plane_tile(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol, int nx, int ny, int nz, int nu, int nv,
    float* __restrict__ vol1 = nullptr, const float* __restrict__ img1 = nullptr)
{
    // Two planes share ray arithmetic and pixel loads. Their 15-row tiles
    // retain the single-plane shared-memory footprint and occupancy.
    constexpr int ROWS = PLANES == 1 ? PB : 15;
    constexpr int RUN = PLANES == 1 ? VRUN : 4;
    __shared__ float acc[RHS][PLANES][ROWS + 2][PC + 3];  // a cell of padding round the tile: no bounds tests
    __shared__ float cf[VMAX][12];  // source, detector origin, u and v axes
    __shared__ int foot[VMAX][4];   // first column and row, rows, runs per column
    __shared__ int first[VMAX + 1]; // each view's first run in the block's list
    int b = a == 0 ? 1 : 0, cc = a == 2 ? 1 : 2;
    int na = SEL(a, nx, ny, nz), nb = SEL(b, nx, ny, nz), nc = SEL(cc, nx, ny, nz);
    int c0 = (VOLUME_Z_FAST ? blockIdx.x : blockIdx.z) * PC;
    int b0 = (VOLUME_Z_FAST ? blockIdx.y : (a == 0 ? blockIdx.y : blockIdx.x)) * ROWS;
    int k = (VOLUME_Z_FAST ? blockIdx.z : (a == 0 ? blockIdx.x : blockIdx.y)) * PLANES;
    int nb0 = min(ROWS, nb - b0), nc0 = min(PC, nc - c0);
    if (!any_view(coeff, axis_box, nviews, a, false)) return;
    for (int i = threadIdx.x; i < RHS * PLANES * (ROWS + 2) * (PC + 3); i += blockDim.x) (&acc[0][0][0][0])[i] = 0.f;
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
            for (int q = 0; q < 4 * PLANES; ++q) {
                // Corners of the tile's part inside the volume (see sep_adjoint).
                float pb = q & 1 ? (float)(b0 + nb0) : (float)(b0 - 1);
                float pc = q & 2 ? (float)(c0 + nc0) : (float)(c0 - 1);
                float pk = PLANES == 1 ? (float)k : (float)min(k + q / 4, na - 1);
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
            int ulo = ab[0], uhi = ab[1], vlo = ab[2], vhi = ab[3];
            if (!unbounded) {
                // Rounded forward crossings can lie just inside a voxel while
                // an inverse corner rounds past an integer detector pixel.
                // Round candidate bounds outwards; the forward cell and axis
                // tests below still determine every interpolation weight.
                ulo = max((int)floorf(pixel_bound(umin, nu)), ulo);
                uhi = min((int)ceilf(pixel_bound(umax, nu)), uhi);
                vlo = max((int)floorf(pixel_bound(vmin, nv)), vlo);
                vhi = min((int)ceilf(pixel_bound(vmax, nv)), vhi);
            }
            int wu = uhi - ulo + 1, wv = vhi - vlo + 1;
            if (wu > 0 && wv > 0) {
                int runs = (wv + RUN - 1) / RUN;
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
    float Sa = 0.f, Sb = 0.f, Sc = 0.f, DVa = 0.f, DVb = 0.f, DVc = 0.f;
    for (int p = t; p < total; p += blockDim.x) {
        while (p >= first[w + 1]) ++w;
        const float* c = cf[w];
        if (w != cw) {  // the view's constants, while this thread stays on it
            cw = w;
            ulo = foot[w][0]; vlo = foot[w][1]; wv = foot[w][2]; runs = foot[w][3];
            image = img + (long)w * nu * nv;
            Sa = c[0]; Sb = c[1]; Sc = c[2];
            DVa = c[9]; DVb = c[10]; DVc = c[11];
        }
        int q = p - first[w];
        int u = ulo + q / runs, v0 = vlo + RUN * (q % runs), v1 = min(v0 + RUN, vlo + wv);
        float fu = (float)u;
        // Keep the detector-column position, not position minus source: ray_of
        // adds the v displacement first, then subtracts the source.
        float ra = __fmaf_rn(fu, c[6], c[3]);
        float rb = __fmaf_rn(fu, c[7], c[4]);
        float rc = __fmaf_rn(fu, c[8], c[5]);
        const float* col = image + (long)u * nv;
        const float* col1 = RHS == 2 ? img1 + ((long)w * nu + u) * nv : nullptr;
        // Load the run's pixels together: one wait for memory instead of RUN.
        float values[RHS][RUN];
        #pragma unroll
        for (int i = 0; i < RUN; ++i) {
            values[0][i] = v0 + i < v1 ? __ldg(col + v0 + i) : 0.f;
            if (RHS == 2) values[1][i] = v0 + i < v1 ? __ldg(col1 + v0 + i) : 0.f;
        }
        // The run's crossings of plane k first, independent of each other, so their
        // divisions overlap; then their cells, in order.
        float fbs[PLANES][RUN], fcs[PLANES][RUN];
        #pragma unroll
        for (int i = 0; i < RUN; ++i) {
            float fv = (float)(v0 + i);
            float r_a = __fsub_rn(__fmaf_rn(fv, DVa, ra), Sa);
            float r_b = __fsub_rn(__fmaf_rn(fv, DVb, rb), Sb);
            float r_c = __fsub_rn(__fmaf_rn(fv, DVc, rc), Sc);
            float aa = fabsf(r_a), ab = fabsf(r_b), ac = fabsf(r_c);
            float inverse = __frcp_rn(r_a);
            bool hit = v0 + i < v1
                && (a == 0 ? aa >= ab : aa > ab)
                && (a == 2 ? aa > ac : aa >= ac);
            #pragma unroll
            for (int plane = 0; plane < PLANES; ++plane) {
                // Preserve each plane's original FP32 crossing, not a recurrence.
                float K = __fsub_rn((float)(k + plane), Sa);
                float t_ = __fmul_rn(K, inverse);
                fbs[plane][i] = hit && (PLANES == 1 || k + plane < na) ? __fmaf_rn(t_, r_b, Sb) : -1e9f;
                fcs[plane][i] = __fmaf_rn(t_, r_c, Sc);
            }
        }
        #pragma unroll
        for (int plane = 0; plane < PLANES; ++plane) {
            int cb = -100, ccj = -100;
            float x00[RHS] = {}, x01[RHS] = {}, x10[RHS] = {}, x11[RHS] = {};
            #pragma unroll
            for (int i = 0; i < RUN; ++i) {
                float fb = fbs[plane][i], fc = fcs[plane][i];
                float fb0 = floorf(fb), fc0 = floorf(fc);
                int jb = (int)fb0 - b0, jc = (int)fc0 - c0;
                if (jb < -1 || jb >= ROWS || jc < -1 || jc >= PC) continue;
                float wb1 = fb - fb0, wc1 = fc - fc0, wb0 = 1.f - wb1, wc0 = 1.f - wc1;
                if (jb != cb || jc != ccj) {
                    bool next = jb == cb && jc == ccj + 1;
                    #pragma unroll
                    for (int r = 0; r < RHS; ++r) {
                        float* cell = &acc[r][plane][cb + 1][ccj + 1];
                        add4(cell, x00[r], cell + 1, next ? 0.f : x01[r],
                             cell + (PC + 3), x10[r], cell + (PC + 4), next ? 0.f : x11[r]);
                        x00[r] = next ? x01[r] : 0.f; x10[r] = next ? x11[r] : 0.f;
                        x01[r] = x11[r] = 0.f;
                    }
                    cb = jb; ccj = jc;
                }
                #pragma unroll
                for (int r = 0; r < RHS; ++r) {
                    float value = values[r][i];
                    x00[r] += value * wb0 * wc0; x01[r] += value * wb0 * wc1;
                    x10[r] += value * wb1 * wc0; x11[r] += value * wb1 * wc1;
                }
            }
            if (cb > -100) {
                #pragma unroll
                for (int r = 0; r < RHS; ++r) {
                    float* cell = &acc[r][plane][cb + 1][ccj + 1];
                    add4(cell, x00[r], cell + 1, x01[r], cell + (PC + 3), x10[r], cell + (PC + 4), x11[r]);
                }
            }
        }
    }
    __syncthreads();
    long s0 = VOLUME_Z_FAST ? (long)ny * nz : 1;
    long s1 = VOLUME_Z_FAST ? nz : nx, s2 = VOLUME_Z_FAST ? 1 : (long)nx * ny;
    long sa = SEL(a, s0, s1, s2), sb = SEL(b, s0, s1, s2), sc = SEL(cc, s0, s1, s2);
    if (VOLUME_Z_FAST) {
        for (int i = threadIdx.x; i < ROWS * PC; i += blockDim.x) {
            int jb = i / PC, jc = i % PC;
            if (jb < nb0 && jc < nc0) {
                #pragma unroll
                for (int plane = 0; plane < PLANES; ++plane) if (k + plane < na) {
                    long offset = (long)(k + plane) * sa + (long)(b0 + jb) * sb + (long)(c0 + jc) * sc;
                    vol[offset] += acc[0][plane][jb + 1][jc + 1];
                    if (RHS == 2) vol1[offset] += acc[1][plane][jb + 1][jc + 1];
                }
            }
        }
        return;
    }
    for (int i = threadIdx.x; i < PLANES * ROWS * PC; i += blockDim.x) {
        int plane = a == 0 ? i % PLANES : i / (ROWS * PC);
        int jb = a == 0 ? (i / PLANES) % ROWS : i % ROWS;
        int jc = a == 0 ? i / (PLANES * ROWS) : (i / ROWS) % PC;
        if (jb < nb0 && jc < nc0 && k + plane < na) {
            long offset = (long)(k + plane) * sa + (long)(b0 + jb) * sb + (long)(c0 + jc) * sc;
            vol[offset] += acc[0][plane][jb + 1][jc + 1];
            if (RHS == 2) vol1[offset] += acc[1][plane][jb + 1][jc + 1];
        }
    }

}

extern "C" __global__ void __launch_bounds__(128, 8) plane_adjoint(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol, int nx, int ny, int nz, int nu, int nv)
{
    plane_tile<32, 1, 2>(coeff, axis_box, nviews, a, img, vol, nx, ny, nz, nu, nv);
}

// Wider physical-z tiles amortize detector footprints and tile boundaries;
// z-dominant rays and short z extents retain the narrower shared-memory tile.
extern "C" __global__ void __launch_bounds__(128, 8) plane_adjoint_wide(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol, int nx, int ny, int nz, int nu, int nv)
{
    plane_tile<64, 1, 2>(coeff, axis_box, nviews, a, img, vol, nx, ny, nz, nu, nv);
}

// One traversal accumulates the two spatial gradients and their plane moments.
// Six physical pose directions are then contractions, not six extra ray traversals.
template <bool TRANSPOSE>
__device__ void cone_pose_derivative(
    const float* __restrict__ coeff, const float* __restrict__ tangent,
    const float* __restrict__ vol, float* __restrict__ out, const float* cotangent,
    int nx, int ny, int nz, int nu, int nv, int width, float sx, float sy, float sz)
{
    int view = blockIdx.x;
    const float* c = coeff + (long)view * NC;
    int u = blockIdx.y * 16 + threadIdx.x / 8, v = blockIdx.z * 8 + threadIdx.x % 8;
    bool live = u < nu && v < nv;
    float fu = u, fv = v, r0, r1, r2;
    ray_of(c, fu, fv, r0, r1, r2);
    int a = ray_axis(r0,r1,r2), b = a == 0 ? 1 : 0, cc = a == 2 ? 1 : 2;
    int na = SEL(a,nx,ny,nz), nb = SEL(b,nx,ny,nz), nc = SEL(cc,nx,ny,nz);
    long s0=VOLUME_Z_FAST?(long)ny*nz:1;
    long s1=VOLUME_Z_FAST?nz:nx, s2=VOLUME_Z_FAST?1:(long)nx*ny;
    long sa=SEL(a,s0,s1,s2), sb=SEL(b,s0,s1,s2), sc=SEL(cc,s0,s1,s2);
    float ra=SEL(a,r0,r1,r2), rb=SEL(b,r0,r1,r2), rc=SEL(cc,r0,r1,r2);
    float Sa=c[a], Sb=c[b], Sc=c[cc], inv=__frcp_rn(ra);
    float slopeb=rb*inv, slopec=rc*inv;
    float lo=0.f, hi=(float)(na-1), base=Sb-Sa*slopeb;
    if(fabsf(slopeb)<1e-12f) { if(base < -1.f || base > (float)nb) live=false; }
    else { float k0=(-1.f-base)/slopeb,k1=((float)nb-base)/slopeb;
        lo=fmaxf(lo,fminf(k0,k1)); hi=fminf(hi,fmaxf(k0,k1)); }
    base=Sc-Sa*slopec;
    if(fabsf(slopec)<1e-12f) { if(base < -1.f || base > (float)nc) live=false; }
    else { float k0=(-1.f-base)/slopec,k1=((float)nc-base)/slopec;
        lo=fmaxf(lo,fminf(k0,k1)); hi=fminf(hi,fmaxf(k0,k1)); }
    int k0=max((int)floorf(fminf(lo,(float)na)),0),k1=min((int)ceilf(fmaxf(hi,-1.f)),na-1);
    if(!live) { k0=na;k1=-1; }
    int first=warp_min(k0),last=warp_max(k1);
    // Spatial gradients and their ray-parameter moments suffice for every pose direction.
    float sum=0.f, gb=0.f,gc=0.f,gtb=0.f,gtc=0.f;
    for(int k=first;k<=last;++k) {
        if(k<k0 || k>k1)continue;
        float t=__fmul_rn(__fsub_rn((float)k,Sa),inv);
        float fb=__fmaf_rn(t,rb,Sb),fc=__fmaf_rn(t,rc,Sc);
        float fb0=floorf(fb),fc0=floorf(fc);
        int b0=(int)fb0,c0=(int)fc0;
        float wb1=fb-fb0,wc1=fc-fc0,wb0=1.f-wb1,wc0=1.f-wc1;
        bool b0in=(unsigned)b0<(unsigned)nb,b1in=(unsigned)(b0+1)<(unsigned)nb;
        bool c0in=(unsigned)c0<(unsigned)nc,c1in=(unsigned)(c0+1)<(unsigned)nc;
        const float* p=vol+(long)k*sa+(long)b0*sb+(long)c0*sc;
        float v00=b0in&&c0in?__ldg(p):0.f;
        float v01=b0in&&c1in?__ldg(p+sc):0.f;
        float v10=b1in&&c0in?__ldg(p+sb):0.f;
        float v11=b1in&&c1in?__ldg(p+sb+sc):0.f;
        float s0=v00*wc0+v01*wc1,s1=v10*wc0+v11*wc1;
        sum+=wb0*s0+wb1*s1;
        float db=s1-s0,dc=wb0*(v01-v00)+wb1*(v11-v10);
        gb+=db;gc+=dc;gtb+=t*db;gtc+=t*dc;
    }
    if(!TRANSPOSE && (u>=nu || v>=nv))return;
    long pixel=((long)view*(width+1)*nv+v)*nu+u, stride=(long)nv*nu;
    float w=path_w(r0,r1,r2,ra,sx,sy,sz);
    if(!TRANSPOSE) out[pixel]=sum*w;
    float length2=(r0*sx)*(r0*sx)+(r1*sy)*(r1*sy)+(r2*sz)*(r2*sz);
    // Separate source and ray derivatives, then apply pixel-source geometry.
    // Scalar components avoid dynamically indexed local arrays spilling the
    // six accumulated moments to per-thread device memory.
    float gsa=-slopeb*gb-slopec*gc, gra=-slopeb*gtb-slopec*gtc-sum*inv;
    float gs0=a==0?gsa:gb, gs1=SEL(a,gb,gsa,gc), gs2=a==2?gsa:gc;
    float gr0=sum*r0*sx*sx/length2+(a==0?gra:gtb);
    float gr1=sum*r1*sy*sy/length2+SEL(a,gtb,gra,gtc);
    float gr2=sum*r2*sz*sz/length2+(a==2?gra:gtc);
    if(TRANSPOSE) {
        // Reduce each detector tile before atomically accumulating twelve
        // coefficients. Scratch is constant, independent of the image size.
        __shared__ float partial[12][4];
        float value=(u<nu && v<nv)?cotangent[((long)view*nv+v)*nu+u]*w:0.f;
        int lane=threadIdx.x%32, warp=threadIdx.x/32;
        #pragma unroll
        for(int q=0;q<12;++q) {
            int axis=q%3;
            float gs=SEL(axis,gs0,gs1,gs2), gr=SEL(axis,gr0,gr1,gr2);
            float grad=q<3?gs-gr:gr*(q<6?1.f:(q<9?fu:fv));
            grad*=value;
            for(int offset=16;offset;offset/=2)
                grad+=__shfl_down_sync(0xffffffffu,grad,offset);
            if(lane==0) partial[q][warp]=grad;
        }
        __syncthreads();
        if(threadIdx.x<12) {
            int q=threadIdx.x;
            atomicAdd(out+(long)view*NC+q,
                      (partial[q][0]+partial[q][1])+(partial[q][2]+partial[q][3]));
        }
    } else {
        const float* dc=tangent+(long)view*width*12;
        for(int q=0;q<width;++q) {
            const float* d=dc+q*12;
            float derivative=0.f;
            #pragma unroll
            for(int i=0;i<3;++i) {
                float dr=d[3+i]+fu*d[6+i]+fv*d[9+i]-d[i];
                derivative+=SEL(i,gs0,gs1,gs2)*d[i]+SEL(i,gr0,gr1,gr2)*dr;
            }
            out[pixel+(q+1)*stride]=w*derivative;
        }
    }
}

extern "C" __global__ void cone_pose_columns(
    const float* coeff, const float* tangent, const float* vol, float* out,
    int nx, int ny, int nz, int nu, int nv, int width, float sx, float sy, float sz)
{
    cone_pose_derivative<false>(coeff,tangent,vol,out,nullptr,
                               nx,ny,nz,nu,nv,width,sx,sy,sz);
}

extern "C" __global__ void cone_pose_vjp(
    const float* coeff, const float* vol, const float* cotangent, float* out,
    int nx, int ny, int nz, int nu, int nv, float sx, float sy, float sz)
{
    cone_pose_derivative<true>(coeff,nullptr,vol,out,cotangent,
                              nx,ny,nz,nu,nv,0,sx,sy,sz);
}

// Compact separable tiles cache all view metadata and process two z cells per lane.
#define CACHED_ULEN 78
template <int ZT>
__device__ __forceinline__ void sep_adjoint_cached_tile(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol,
    int nx, int ny, int nz, int nu, int nv)
{
    __shared__ float R[CACHED_ULEN][ZT + 1];
    __shared__ float s_fb[CACHED_ULEN], s_t[CACHED_ULEN], s_rc0[CACHED_ULEN];
    __shared__ float s_f0[CACHED_ULEN], s_idf[CACHED_ULEN];
    __shared__ int s_taps[CACHED_ULEN];
    __shared__ float s_margin[CACHED_ULEN];
    __shared__ float all_cf[VMAX][12];
    __shared__ int all_skip[VMAX];
    __shared__ int2 all_foot[VMAX];
    int b = 1 - a;
    int nb = a == 0 ? ny : nx;
    int c0 = (VOLUME_Z_FAST ? blockIdx.x : blockIdx.z) * ZT;
    int b0 = (VOLUME_Z_FAST ? blockIdx.y : blockIdx.x) * TB;
    int k = VOLUME_Z_FAST ? blockIdx.z : blockIdx.y;
    unsigned tid = threadIdx.x;
    // Stripes of ZT / 8 cells spread the four z lanes across shared-memory banks.
    int my_b = b0 + tid / 4, my_c = (tid % 4) * (ZT / 8);
    float my_bf = (float)my_b;
    if (!any_view(coeff, axis_box, nviews, a, true)) return;
    for (int i = tid; i < nviews * 12; i += blockDim.x)
        (&all_cf[0][0])[i] = coeff[(i / 12) * NC + i % 12];
    // Bounds depend on the lateral tile, not z. Cache them with each view.
    for (int i = tid; i < nviews; i += blockDim.x) {
        const float* c = coeff + i * NC;
        const int* ab = axis_box + (i * 3 + a) * 4;
        all_skip[i] = !separable(c) || ab[1] < ab[0];
        float K = (float)k - c[a], Sb = c[b];
        float ra0 = c[3 + a] - c[a], rb0 = c[3 + b] - c[b];
        float DUa = c[6 + a], DUb = c[6 + b];
        // Clip the far tile edge to the volume before projecting its bounds.
        float g1 = (float)b0 - 1.f - Sb, g2 = (float)min(b0 + TB, nb) - Sb;
        float den1 = g1 * DUa - K * DUb, den2 = g2 * DUa - K * DUb;
        int lo = ab[0], hi = ab[1];
        if (!crosses_zero(den1, den2)) {
            float u1 = (K * rb0 - g1 * ra0) / den1, u2 = (K * rb0 - g2 * ra0) / den2;
            lo = max((int)floorf(pixel_bound(fminf(u1, u2), nu)) - 1, ab[0]);
            hi = min((int)ceilf(pixel_bound(fmaxf(u1, u2), nu)) + 1, ab[1]);
        }
        all_foot[i] = make_int2(lo, hi);
    }

    __syncthreads();
    float acc[ZT / 4];
    #pragma unroll
    for (int j = 0; j < ZT / 4; ++j) acc[j] = 0.f;
    for (int w = 0; w < nviews; ++w) {
        if (all_skip[w]) continue;
        const float* c = all_cf[w];
        float Sa = c[a], Sb = c[b], Sc = c[2], DVc = c[11];
        float K = (float)k - Sa;
        float ra0 = c[3 + a] - c[a], rb0 = c[3 + b] - c[b], DUa = c[6 + a], DUb = c[6 + b];
        int ulo_all = all_foot[w].x, uhi = all_foot[w].y;
        const float* image = img + (long)w * nu * nv;
        for (int ulo = ulo_all; ulo <= uhi; ulo += CACHED_ULEN) {
            int ulen = min(CACHED_ULEN, uhi - ulo + 1);
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
                float margin = vertical_support_margin(idf, Sc, t, rc0, DVc, nz, nv);
                s_margin[i] = margin;
                // Widen both endpoints, keeping the tap count warp-uniform.
                float taps = fminf(2.f * (fabsf(idf) + margin), (float)nv);
                s_taps[i] = col_axis == a ? min((int)floorf(taps) + 1, nv) : 0;
            }
            __syncthreads();
            const float* col = image + (long)(ulo + tid / (ZT / 2)) * nv;
            for (int i = tid / (ZT / 2), j = tid % (ZT / 2); i < ulen;
                 i += blockDim.x / (ZT / 2), col += (blockDim.x / (ZT / 2)) * nv) {
                float t = s_t[i], rc0 = s_rc0[i], f0 = s_f0[i], idf = s_idf[i];
                int taps = s_taps[i];
                #pragma unroll
                for (int run = 0; run < 2; ++run) {
                    int jj = j + run * (ZT / 2);
                    float jcf = (float)(c0 + jj);
                    float v1 = (jcf - 1.f - f0) * idf, v2 = (jcf + 1.f - f0) * idf;
                    int va = max((int)floorf(pixel_bound(fminf(v1, v2) - s_margin[i], nv)) + 1, 0);
                    float s = 0.f;
                    // Common interior footprints avoid repeated bounds checks;
                    // boundary and wider footprints keep every original tap.
                    if (taps == 2 && va + 1 < nv) {
                        s += __ldg(col + va) * trif(Sc + t * ((rc0 + (float)va * DVc) - Sc), jcf);
                        s += __ldg(col + va + 1) * trif(Sc + t * ((rc0 + (float)(va + 1) * DVc) - Sc), jcf);
                    } else if (taps == 3 && va + 2 < nv) {
                        s += __ldg(col + va) * trif(Sc + t * ((rc0 + (float)va * DVc) - Sc), jcf);
                        s += __ldg(col + va + 1) * trif(Sc + t * ((rc0 + (float)(va + 1) * DVc) - Sc), jcf);
                        s += __ldg(col + va + 2) * trif(Sc + t * ((rc0 + (float)(va + 2) * DVc) - Sc), jcf);
                    } else if (taps <= 3) {
                        if (taps > 0 && va < nv)
                            s += __ldg(col + va) * trif(Sc + t * ((rc0 + (float)va * DVc) - Sc), jcf);
                        if (taps > 1 && va + 1 < nv)
                            s += __ldg(col + va + 1) * trif(Sc + t * ((rc0 + (float)(va + 1) * DVc) - Sc), jcf);
                        if (taps > 2 && va + 2 < nv)
                            s += __ldg(col + va + 2) * trif(Sc + t * ((rc0 + (float)(va + 2) * DVc) - Sc), jcf);
                    } else {
                        #pragma unroll 4
                        for (int q = 0; q < taps; ++q) {
                            int vv = va + q;
                            if (vv < 0 || vv >= nv) continue;
                            s += __ldg(col + vv) * trif(Sc + t * ((rc0 + (float)vv * DVc) - Sc), jcf);
                        }
                    }
                    R[i][jj] = s;
                }
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
                    for (int j = 0; j < ZT / 4; ++j) {
                        int offset = j % (ZT / 8) + (j / (ZT / 8)) * (ZT / 2);
                        acc[j] += wb * R[i][my_c + offset];
                    }
                }
            }
        }
    }
    // R is no longer read by the accumulation phase. Reuse it to
    // exchange the register-z owners for contiguous-x output lanes.
    if (!VOLUME_Z_FAST && a == 1) {
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < ZT / 4; ++j) {
            int offset = j % (ZT / 8) + (j / (ZT / 8)) * (ZT / 2);
            R[tid / 4][my_c + offset] = acc[j];
        }
        __syncthreads();
        for (int i = tid; i < TB * ZT; i += blockDim.x) {
            int jb = i % TB, jc = i / TB;
            if (b0 + jb < nx && c0 + jc < nz && k < ny)
                vol[((long)(c0 + jc) * ny + k) * nx + b0 + jb] += R[jb][jc];
        }
        return;
    }
    int na = a == 0 ? nx : ny;
    if (my_b < nb && k < na) {
        int ix = a == 0 ? k : my_b, iy = a == 0 ? my_b : k;
        long base = volume_index(ix, iy, c0 + my_c, nx, ny, nz);
        #pragma unroll
        for (int j = 0; j < ZT / 4; ++j) {
            int offset = j % (ZT / 8) + (j / (ZT / 8)) * (ZT / 2);
            if (c0 + my_c + offset < nz) vol[base + offset * (VOLUME_Z_FAST ? 1L : (long)nx * ny)] += acc[j];
        }
    }
}

extern "C" __global__ void sep_adjoint_cached(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol,
    int nx, int ny, int nz, int nu, int nv)
{
    sep_adjoint_cached_tile<TC>(coeff, axis_box, nviews, a, img, vol, nx, ny, nz, nu, nv);
}

// At most 32 planes, as sep_adjoint_thin.
extern "C" __global__ void sep_adjoint_cached_thin(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, float* __restrict__ vol,
    int nx, int ny, int nz, int nu, int nv)
{
    sep_adjoint_cached_tile<32>(coeff, axis_box, nviews, a, img, vol, nx, ny, nz, nu, nv);
}

// Two independent right-hand sides share ray coordinates; narrower tiles retain occupancy.
extern "C" __global__ void __launch_bounds__(128, 8) pair_plane_adjoint(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, const float* __restrict__ img1,
    float* __restrict__ vol, float* __restrict__ vol1,
    int nx, int ny, int nz, int nu, int nv)
{
    plane_tile<32, 2>(coeff, axis_box, nviews, a, img, vol, nx, ny, nz, nu, nv, vol1, img1);
}

// Two 32-plane row tiles and cached geometry stay within 24 KiB per block.
#define PAIRED_ULEN 77
template <int ZT>
__device__ __forceinline__ void pair_sep_adjoint_cached_tile(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, const float* __restrict__ img1,
    float* __restrict__ vol, float* __restrict__ vol1,
    int nx, int ny, int nz, int nu, int nv)
{
    __shared__ float R[PAIRED_ULEN][ZT + 1], R1[PAIRED_ULEN][ZT + 1];
    __shared__ float s_fb[PAIRED_ULEN], s_t[PAIRED_ULEN], s_rc0[PAIRED_ULEN];
    __shared__ float s_f0[PAIRED_ULEN], s_idf[PAIRED_ULEN];
    __shared__ int s_taps[PAIRED_ULEN];
    __shared__ float s_margin[PAIRED_ULEN];
    __shared__ float all_cf[VMAX][12];
    __shared__ int all_skip[VMAX];
    __shared__ int2 all_foot[VMAX];
    int b = 1 - a;
    int nb = a == 0 ? ny : nx;
    int c0 = (VOLUME_Z_FAST ? blockIdx.x : blockIdx.z) * ZT;
    int b0 = (VOLUME_Z_FAST ? blockIdx.y : blockIdx.x) * TB;
    int k = VOLUME_Z_FAST ? blockIdx.z : blockIdx.y;
    unsigned tid = threadIdx.x;
    // Stripes of ZT / 8 cells spread the four z lanes across shared-memory banks.
    int my_b = b0 + tid / 4, my_c = (tid % 4) * (ZT / 8);
    float my_bf = (float)my_b;
    if (!any_view(coeff, axis_box, nviews, a, true)) return;
    for (int i = tid; i < nviews * 12; i += blockDim.x)
        (&all_cf[0][0])[i] = coeff[(i / 12) * NC + i % 12];
    // Bounds depend on the lateral tile, not z. Cache them with each view.
    for (int i = tid; i < nviews; i += blockDim.x) {
        const float* c = coeff + i * NC;
        const int* ab = axis_box + (i * 3 + a) * 4;
        all_skip[i] = !separable(c) || ab[1] < ab[0];
        float K = (float)k - c[a], Sb = c[b];
        float ra0 = c[3 + a] - c[a], rb0 = c[3 + b] - c[b];
        float DUa = c[6 + a], DUb = c[6 + b];
        // Clip the far tile edge to the volume before projecting its bounds.
        float g1 = (float)b0 - 1.f - Sb, g2 = (float)min(b0 + TB, nb) - Sb;
        float den1 = g1 * DUa - K * DUb, den2 = g2 * DUa - K * DUb;
        int lo = ab[0], hi = ab[1];
        if (!crosses_zero(den1, den2)) {
            float u1 = (K * rb0 - g1 * ra0) / den1, u2 = (K * rb0 - g2 * ra0) / den2;
            lo = max((int)floorf(pixel_bound(fminf(u1, u2), nu)) - 1, ab[0]);
            hi = min((int)ceilf(pixel_bound(fmaxf(u1, u2), nu)) + 1, ab[1]);
        }
        all_foot[i] = make_int2(lo, hi);
    }

    __syncthreads();
    float acc[ZT / 4], acc1[ZT / 4];
    #pragma unroll
    for (int j = 0; j < ZT / 4; ++j) acc[j] = acc1[j] = 0.f;
    for (int w = 0; w < nviews; ++w) {
        if (all_skip[w]) continue;
        const float* c = all_cf[w];
        float Sa = c[a], Sb = c[b], Sc = c[2], DVc = c[11];
        float K = (float)k - Sa;
        float ra0 = c[3 + a] - c[a], rb0 = c[3 + b] - c[b], DUa = c[6 + a], DUb = c[6 + b];
        int ulo_all = all_foot[w].x, uhi = all_foot[w].y;
        const float* image = img + (long)w * nu * nv;
        const float* image1 = img1 + (long)w * nu * nv;
        for (int ulo = ulo_all; ulo <= uhi; ulo += PAIRED_ULEN) {
            int ulen = min(PAIRED_ULEN, uhi - ulo + 1);
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
                float margin = vertical_support_margin(idf, Sc, t, rc0, DVc, nz, nv);
                s_margin[i] = margin;
                // Widen both endpoints, keeping the tap count warp-uniform.
                float taps = fminf(2.f * (fabsf(idf) + margin), (float)nv);
                s_taps[i] = col_axis == a ? min((int)floorf(taps) + 1, nv) : 0;
            }
            __syncthreads();
            const float* col = image + (long)(ulo + tid / ZT) * nv;
            const float* col1 = image1 + (long)(ulo + tid / ZT) * nv;
            for (int i = tid / ZT, j = tid % ZT; i < ulen;
                 i += blockDim.x / ZT, col += (blockDim.x / ZT) * nv,
                 col1 += (blockDim.x / ZT) * nv) {
                float t = s_t[i], rc0 = s_rc0[i], f0 = s_f0[i], idf = s_idf[i];
                int taps = s_taps[i];
                {
                    int jj = j;
                    float jcf = (float)(c0 + jj);
                    float v1 = (jcf - 1.f - f0) * idf, v2 = (jcf + 1.f - f0) * idf;
                    int va = max((int)floorf(pixel_bound(fminf(v1, v2) - s_margin[i], nv)) + 1, 0);
                    float s = 0.f, s1 = 0.f;
                    // Common interior footprints avoid repeated bounds checks;
                    // boundary and wider footprints keep every original tap.
                    if (taps == 2 && va + 1 < nv) {
                        {
                            float weight = trif(Sc + t * ((rc0 + (float)va * DVc) - Sc), jcf);
                            s += __ldg(col + va) * weight; s1 += __ldg(col1 + va) * weight;
                        }
                        {
                            float weight = trif(Sc + t * ((rc0 + (float)(va + 1) * DVc) - Sc), jcf);
                            s += __ldg(col + va + 1) * weight; s1 += __ldg(col1 + va + 1) * weight;
                        }
                    } else if (taps == 3 && va + 2 < nv) {
                        {
                            float weight = trif(Sc + t * ((rc0 + (float)va * DVc) - Sc), jcf);
                            s += __ldg(col + va) * weight; s1 += __ldg(col1 + va) * weight;
                        }
                        {
                            float weight = trif(Sc + t * ((rc0 + (float)(va + 1) * DVc) - Sc), jcf);
                            s += __ldg(col + va + 1) * weight; s1 += __ldg(col1 + va + 1) * weight;
                        }
                        {
                            float weight = trif(Sc + t * ((rc0 + (float)(va + 2) * DVc) - Sc), jcf);
                            s += __ldg(col + va + 2) * weight; s1 += __ldg(col1 + va + 2) * weight;
                        }
                    } else if (taps <= 3) {
                        if (taps > 0 && va < nv)
                            {
                                float weight = trif(Sc + t * ((rc0 + (float)va * DVc) - Sc), jcf);
                                s += __ldg(col + va) * weight; s1 += __ldg(col1 + va) * weight;
                            }
                        if (taps > 1 && va + 1 < nv)
                            {
                                float weight = trif(Sc + t * ((rc0 + (float)(va + 1) * DVc) - Sc), jcf);
                                s += __ldg(col + va + 1) * weight; s1 += __ldg(col1 + va + 1) * weight;
                            }
                        if (taps > 2 && va + 2 < nv)
                            {
                                float weight = trif(Sc + t * ((rc0 + (float)(va + 2) * DVc) - Sc), jcf);
                                s += __ldg(col + va + 2) * weight; s1 += __ldg(col1 + va + 2) * weight;
                            }
                    } else {
                        #pragma unroll 4
                        for (int q = 0; q < taps; ++q) {
                            int vv = va + q;
                            if (vv < 0 || vv >= nv) continue;
                            {
                                float weight = trif(Sc + t * ((rc0 + (float)vv * DVc) - Sc), jcf);
                                s += __ldg(col + vv) * weight; s1 += __ldg(col1 + vv) * weight;
                            }
                        }
                    }
                    R[i][jj] = s; R1[i][jj] = s1;
                }
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
                    for (int j = 0; j < ZT / 4; ++j) {
                        int offset = j % (ZT / 8) + (j / (ZT / 8)) * (ZT / 2);
                        acc[j] += wb * R[i][my_c + offset];
                        acc1[j] += wb * R1[i][my_c + offset];
                    }
                }
            }
        }
    }
    // R is no longer read by the accumulation phase. Reuse it to
    // exchange the register-z owners for contiguous-x output lanes.
    if (!VOLUME_Z_FAST && a == 1) {
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < ZT / 4; ++j) {
            int offset = j % (ZT / 8) + (j / (ZT / 8)) * (ZT / 2);
            R[tid / 4][my_c + offset] = acc[j];
            R1[tid / 4][my_c + offset] = acc1[j];
        }
        __syncthreads();
        for (int i = tid; i < TB * ZT; i += blockDim.x) {
            int jb = i % TB, jc = i / TB;
            if (b0 + jb < nx && c0 + jc < nz && k < ny) {
                long address = ((long)(c0 + jc) * ny + k) * nx + b0 + jb;
                vol[address] += R[jb][jc]; vol1[address] += R1[jb][jc];
            }
        }
        return;
    }
    int na = a == 0 ? nx : ny;
    if (my_b < nb && k < na) {
        int ix = a == 0 ? k : my_b, iy = a == 0 ? my_b : k;
        long base = volume_index(ix, iy, c0 + my_c, nx, ny, nz);
        #pragma unroll
        for (int j = 0; j < ZT / 4; ++j) {
            int offset = j % (ZT / 8) + (j / (ZT / 8)) * (ZT / 2);
            if (c0 + my_c + offset < nz) {
                long address = base + offset * (VOLUME_Z_FAST ? 1L : (long)nx * ny);
                vol[address] += acc[j]; vol1[address] += acc1[j];
            }
        }
    }
}


extern "C" __global__ void __launch_bounds__(256) pair_sep_adjoint_cached(
    const float* __restrict__ coeff, const int* __restrict__ axis_box, int nviews, int a,
    const float* __restrict__ img, const float* __restrict__ img1,
    float* __restrict__ vol, float* __restrict__ vol1,
    int nx, int ny, int nz, int nu, int nv)
{
    pair_sep_adjoint_cached_tile<32>(coeff, axis_box, nviews, a, img, img1, vol, vol1,
                                     nx, ny, nz, nu, nv);
}
// Exact dense axis-0/axis-2 permutation, tiled for coalesced reads and writes.
extern "C" __global__ void transpose_xz(const float* src,float* dst,int nx,int ny,int nz) {
    __shared__ float tile[32][33];
    int x=blockIdx.x*32+threadIdx.x, z=blockIdx.y*32+threadIdx.y, y=blockIdx.z;
    #pragma unroll
    for(int j=0;j<32;j+=8) if(x<nx && z+j<nz)
        tile[threadIdx.y+j][threadIdx.x]=src[((long)(z+j)*ny+y)*nx+x];
    __syncthreads();
    x=blockIdx.x*32+threadIdx.y; z=blockIdx.y*32+threadIdx.x;
    #pragma unroll
    for(int j=0;j<32;j+=8) if(x+j<nx && z<nz)
        dst[((long)(x+j)*ny+y)*nz+z]=tile[threadIdx.x][threadIdx.y+j];
}
