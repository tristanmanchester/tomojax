// A thread owns a fixed-major-plane B by C voxel tile. Each candidate ray
// needs B+C cubic weights and one detector read, shared across all B*C voxels.
template<int B,int C,typename Image>
__device__ __forceinline__ void gather_tile(
    const float* c,Image image,int k,int b0,int c0,int nu,int nv,float (&total)[B*C]) {
    float hb=__fadd_rn(__fmul_rn(c[3],(float)k),c[4]);
    float hc=__fadd_rn(__fmul_rn(c[7],(float)k),c[8]);
    float iub=c[10],iuc=c[11],ivb=c[12],ivc=c[13];
    float lo_u=3.4e38f,hi_u=-3.4e38f,lo_v=3.4e38f,hi_v=-3.4e38f;
    #pragma unroll
    for(int i=0;i<2;++i) {
        #pragma unroll
        for(int j=0;j<2;++j) {
            float b=(float)(b0+i*(B-1)),t=(float)(c0+j*(C-1));
            float u=iub*(b-hb)+iuc*(t-hc),v=ivb*(b-hb)+ivc*(t-hc);
            lo_u=fminf(lo_u,u);hi_u=fmaxf(hi_u,u);lo_v=fminf(lo_v,v);hi_v=fmaxf(hi_v,v);
        }
    }
    float ru=RADIUS*(fabsf(iub)+fabsf(iuc)),rv=RADIUS*(fabsf(ivb)+fabsf(ivc));
    float eb=fmaxf(fabsf((float)b0),fabsf((float)(b0+B-1)))+fabsf(hb);
    float ec=fmaxf(fabsf((float)c0),fabsf((float)(c0+C-1)))+fabsf(hc);
    float mu=8*1.1920928955078125e-7f*(fabsf(iub)*eb+fabsf(iuc)*ec+ru+1.f);
    float mv=8*1.1920928955078125e-7f*(fabsf(ivb)*eb+fabsf(ivc)*ec+rv+1.f);
    int u_start=max((int)floorf(lo_u-ru-mu)+1,0),u_stop=min((int)ceilf(hi_u+ru+mu),nu);
    int v_start=max((int)floorf(lo_v-rv-mv)+1,0),v_stop=min((int)ceilf(hi_v+rv+mv),nv);
    float inv_vb=c[2]==0.f?0.f:1.f/c[2],inv_vc=c[6]==0.f?0.f:1.f/c[6];
    for(int u=u_start;u<u_stop;++u) {
        float bu=__fmul_rn(c[1],(float)u),cu=__fmul_rn(c[5],(float)u);
        int first=v_start,stop=v_stop;
        clip_coordinate(bu,c[2],inv_vb,hb,hb,b0-RADIUS,b0+B-1+RADIUS,first,stop);
        clip_coordinate(cu,c[6],inv_vc,hc,hc,c0-RADIUS,c0+C-1+RADIUS,first,stop);
        for(int v=first;v<stop;++v) {
            float qb=__fadd_rn(__fadd_rn(bu,__fmul_rn(c[2],(float)v)),hb);
            float qc=__fadd_rn(__fadd_rn(cu,__fmul_rn(c[6],(float)v)),hc);
            float wb[B],wc[C]; bool active_b=false,active_c=false;
            #pragma unroll
            for(int i=0;i<B;++i) {wb[i]=plane_weight(qb-(float)(b0+i));active_b|=wb[i]!=0.f;}
            if(!active_b) continue;
            #pragma unroll
            for(int j=0;j<C;++j) {wc[j]=plane_weight(qc-(float)(c0+j));active_c|=wc[j]!=0.f;}
            if(!active_c) continue;
            float value=joseph_pixel(image,nu,u,v);
            #pragma unroll
            for(int i=0;i<B;++i) {
                #pragma unroll
                for(int j=0;j<C;++j) total[i*C+j]+=value*(wb[i]*wc[j]*c[9]);
            }
        }
    }
}
