// Shared matched Joseph gathers, using global voxel and detector coordinates.
__device__ __forceinline__ float joseph_pixel(const float* image,int nu,int u,int v) {
    return image[(long)v*nu+u];
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
    for (int u = u_start; u < u_stop; ++u) {
        for (int v = v_start; v < v_stop; ++v) {
            float value = joseph_pixel(image,nu,u,v);
            // Round each product as the forward projection does; weights
            // must match it exactly to keep the transpose matched.
            float pb = __fadd_rn(__fmul_rn(ub, (float)u), __fmul_rn(vb, (float)v));
            float pc = __fadd_rn(__fmul_rn(uc, (float)u), __fmul_rn(vc, (float)v));
            #pragma unroll
            for (int j = 0; j < R; ++j) {
                float qb = __fadd_rn(pb, hb[j]), qc = __fadd_rn(pc, hc[j]);
                total[j] += value * (plane_weight(qb - tb[j])
                                     * plane_weight(qc - tc[j]) * weight);
            }
        }
    }
}

