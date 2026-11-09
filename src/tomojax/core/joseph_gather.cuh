// Shared matched Joseph gathers, using global voxel and detector coordinates.
__device__ __forceinline__ float joseph_pixel(const float* image,int nu,int u,int v) {
    return image[(long)v*nu+u];
}
// Adjacent gather threads follow z, hence neighbouring detector rows. Keep
// v contiguous inside these kernels; public and region sinograms remain VU.
struct DetectorColumns { const float* data; int nv; };
__device__ __forceinline__ float joseph_pixel(DetectorColumns image,int,int u,int v) {
    return image.data[(long)u*image.nv+v];
}
#define RADIUS (CUBIC ? 2.f : 1.f)
__device__ __forceinline__ float plane_weight(float q) {
#if CUBIC
    float w=joseph_cubic_weight(q);
    return ABSOLUTE ? fabsf(w) : w;
#else
    return fmaxf(1.f-fabsf(q),0.f);
#endif
}

// The interval estimate only seeds a search. Correct it with the very same
// rounded affine coordinate as the forward sampler, so a near-edge ray is
// never removed by inverse-coordinate roundoff or an approximate reciprocal.
template<bool Inclusive>
__device__ __forceinline__ int coordinate_crossing(
    float base,float slope,float inverse,float offset,float target,int first,int stop) {
    float estimate=((target-offset)-base)*inverse;
    estimate=Inclusive?ceilf(estimate):floorf(estimate)+1.f;
    int at=(int)fminf(fmaxf(estimate,(float)first),(float)stop);
    while(at>first) {
        float q=__fadd_rn(__fadd_rn(base,__fmul_rn(slope,(float)(at-1))),offset);
        bool crossed=slope>0.f ? (Inclusive?q>=target:q>target)
                              : (Inclusive?q<=target:q<target);
        if(!crossed) break;
        --at;
    }
    while(at<stop) {
        float q=__fadd_rn(__fadd_rn(base,__fmul_rn(slope,(float)at)),offset);
        bool crossed=slope>0.f ? (Inclusive?q>=target:q>target)
                              : (Inclusive?q<=target:q<target);
        if(crossed) break;
        ++at;
    }
    return at;
}

__device__ __forceinline__ void clip_coordinate(
    float base,float slope,float inverse,float offset_lo,float offset_hi,
    float lower,float upper,int& first,int& stop) {
    if(first>=stop) return;
    if(slope==0.f) {
        if(__fadd_rn(base,offset_hi)<=lower || __fadd_rn(base,offset_lo)>=upper) stop=first;
        return;
    }
    float start_offset=slope>0.f?offset_hi:offset_lo;
    float end_offset=slope>0.f?offset_lo:offset_hi;
    float start_target=slope>0.f?lower:upper, end_target=slope>0.f?upper:lower;
    float q=__fadd_rn(__fadd_rn(base,__fmul_rn(slope,(float)first)),start_offset);
    if(slope>0.f?q<=start_target:q>=start_target)
        first=coordinate_crossing<false>(base,slope,inverse,start_offset,start_target,first,stop);
    if(first>=stop) return;
    q=__fadd_rn(__fadd_rn(base,__fmul_rn(slope,(float)(stop-1))),end_offset);
    if(slope>0.f?q>=end_target:q<=end_target)
        stop=coordinate_crossing<true>(base,slope,inverse,end_offset,end_target,first,stop);
}

template<int R, typename Image>
__device__ __forceinline__ void gather_unit_rows(
 const float* c,Image image,int ix,int iy,int iz0,int nu,int nv,float (&total)[R]) {
 const int a=(int)c[0], radius=CUBIC?2:1, taps=2*radius;
 float ub=c[1],vb=c[2],kb=c[3],cb=c[4],uc=c[5],vc=c[6],kc=c[7],cc=c[8],weight=c[9];
 float iub=c[10],iuc=c[11];
 bool diagonal=vb==0.f && uc==0.f && fabsf(vc)==1.f;
 #pragma unroll
 for(int j=0;j<R;++j) {
  int iz=iz0+j;
  float k=(float)(a==0?ix:(a==1?iy:iz));
  float b=(float)(a==0?iy:(a==1?iz:ix)),t=(float)(a==0?iz:(a==1?ix:iy));
  float hb=kb*k+cb,hc=kc*k+cc;
  float us=diagonal?ub:uc,uh=diagonal?hb:hc,ut=diagonal?b:t;
  float vs=diagonal?vc:vb,vh=diagonal?hc:hb,vt=diagonal?t:b;
  int first_v=(int)floorf((vt-vh)*vs)-radius+1;
  float rows[taps];
  #pragma unroll
  for(int q=0;q<taps;q++) rows[q]=plane_weight((vs*(first_v+q)+vh)-vt);
  float cu=iub*(b-hb)+iuc*(t-hc),ru=radius*(fabsf(iub)+fabsf(iuc));
  float eb=fabsf(b)+fabsf(hb),ec=fabsf(t)+fabsf(hc);
  float mu=8*1.1920928955078125e-7f*(fabsf(iub)*eb+fabsf(iuc)*ec+ru+1.f);
  int first=max((int)floorf(cu-ru-mu)+1,0),stop=min((int)ceilf(cu+ru+mu),nu);
  for(int u=first;u<stop;++u) {
   float wu=plane_weight((us*u+uh)-ut);
   #pragma unroll
   for(int q=0;q<taps;q++) {
    int v=first_v+q;
    float w=(diagonal?wu*rows[q]:rows[q]*wu)*weight;
    if(v>=0 && v<nv) total[j]+=joseph_pixel(image,nu,u,v)*w;
   }
  }
 }
}

template<int R, typename Image>
__device__ __forceinline__ void gather_view(
    const float* c, Image image, int ix, int iy, int iz0,
    int nu, int nv, float (&total)[R]) {
    const int a=(int)c[0];
    float ub = c[1], vb = c[2], kb = c[3], cb = c[4];
    float uc = c[5], vc = c[6], kc = c[7], cc = c[8], weight = c[9];
    float iub = c[10], iuc = c[11], ivb = c[12], ivc = c[13];
    int k0 = a == 0 ? ix : (a == 1 ? iy : iz0);
    int b0 = a == 0 ? iy : (a == 1 ? iz0 : ix);
    int c0 = a == 0 ? iz0 : (a == 1 ? ix : iy);
    // Step of the run's voxel index along the plane coordinates.
    int dk = a == 2, db = a == 1, dc = a == 0;
    float hb[R], hc[R], tb[R], tc[R];
    float lo_u = 3.4e38f, hi_u = -3.4e38f, lo_v = 3.4e38f, hi_v = -3.4e38f;
    #pragma unroll
    for (int j = 0; j < R; ++j) {
        float k = (float)(k0 + j * dk);
        hb[j] = __fadd_rn(__fmul_rn(kb, k), cb);
        hc[j] = __fadd_rn(__fmul_rn(kc, k), cc);
        tb[j] = (float)(b0 + j * db);
        tc[j] = (float)(c0 + j * dc);
        float cu = iub * (tb[j] - hb[j]) + iuc * (tc[j] - hc[j]);
        float cv = ivb * (tb[j] - hb[j]) + ivc * (tc[j] - hc[j]);
        lo_u = fminf(lo_u, cu); hi_u = fmaxf(hi_u, cu);
        lo_v = fminf(lo_v, cv); hi_v = fmaxf(hi_v, cv);
    }
    float ru = RADIUS*(fabsf(iub) + fabsf(iuc)), rv = RADIUS*(fabsf(ivb) + fabsf(ivc));
    // Bound inverse-coordinate roundoff without expanding the physical weights.
    float eb = fmaxf(fabsf(tb[0]), fabsf(tb[R-1])) + fmaxf(fabsf(hb[0]), fabsf(hb[R-1]));
    float ec = fmaxf(fabsf(tc[0]), fabsf(tc[R-1])) + fmaxf(fabsf(hc[0]), fabsf(hc[R-1]));
    float mu = 8*1.1920928955078125e-7f*(fabsf(iub)*eb + fabsf(iuc)*ec + ru + 1);
    float mv = 8*1.1920928955078125e-7f*(fabsf(ivb)*eb + fabsf(ivc)*ec + rv + 1);
    int u_start = max((int)floorf(lo_u - ru - mu) + 1, 0);
    int u_stop = min((int)ceilf(hi_u + ru + mu), nu);
    int v_start = max((int)floorf(lo_v - rv - mv) + 1, 0);
    int v_stop = min((int)ceilf(hi_v + rv + mv), nv);
    float inv_vb = vb == 0.f ? 0.f : 1.f / vb, inv_vc = vc == 0.f ? 0.f : 1.f / vc;
    for (int u = u_start; u < u_stop; ++u) {
        float bu = __fmul_rn(ub, (float)u), cu = __fmul_rn(uc, (float)u);
        int first = v_start, stop = v_stop;
        clip_coordinate(bu,vb,inv_vb,fminf(hb[0],hb[R-1]),fmaxf(hb[0],hb[R-1]),
                        tb[0]-RADIUS,tb[R-1]+RADIUS,first,stop);
        clip_coordinate(cu,vc,inv_vc,fminf(hc[0],hc[R-1]),fmaxf(hc[0],hc[R-1]),
                        tc[0]-RADIUS,tc[R-1]+RADIUS,first,stop);
        for (int v = first; v < stop; ++v) {
            // Round each product as the forward projection does; weights
            // must match it exactly to keep the transpose matched.
            float pb = __fadd_rn(bu, __fmul_rn(vb, (float)v));
            float pc = __fadd_rn(cu, __fmul_rn(vc, (float)v));
            float weights[R];
            bool active = false;
            if (a != 2) {
                // k and one transverse coordinate are constant along a z run.
                // Reuse that cubic weight, without changing forward rounding.
                float qb = __fadd_rn(pb, hb[0]), qc = __fadd_rn(pc, hc[0]);
                float fixed = plane_weight(a == 0 ? qb - tb[0] : qc - tc[0]);
                if (fixed == 0.f) continue;
                #pragma unroll
                for (int j = 0; j < R; ++j) {
                    float moving = plane_weight(a == 0 ? qc - tc[j] : qb - tb[j]);
                    weights[j] = fixed * moving * weight;
                    active |= weights[j] != 0.f;
                }
            } else {
                #pragma unroll
                for (int j = 0; j < R; ++j) {
                    float qb = __fadd_rn(pb, hb[j]), qc = __fadd_rn(pc, hc[j]);
                    weights[j] = plane_weight(qb - tb[j]) * plane_weight(qc - tc[j]) * weight;
                    active |= weights[j] != 0.f;
                }
            }
            // The union bounding box includes rays outside every voxel support.
            // Do not fetch those pixels, or fetch a shared pixel once per voxel.
            if (!active) continue;
            float value = joseph_pixel(image,nu,u,v);
            #pragma unroll
            for (int j = 0; j < R; ++j) {
                total[j] += value * weights[j];
            }
        }
    }
}

