#include "kvl/ops.h"
#include "kvl/v3_fixture.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_span(FILE *f, uint64_t off, size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) return NULL;
    if (fseek(f, (long)off, SEEK_SET) != 0 || fread(p, 1, n, f) != n) {
        free(p); return NULL;
    }
    return p;
}

static double max_abs(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s experts.bin experts.idx fixture.bin cache_bytes\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[3], "rb");
    if (!f) { perror("fixture"); return 2; }
    KvlV3FixtureHeader h;
    if (fread(&h, 1, sizeof h, f) != sizeof h ||
        memcmp(h.magic, KVL_V3_FIXTURE_MAGIC, 8) || h.version != 1) {
        fprintf(stderr, "bad V3 fixture\n"); fclose(f); return 2;
    }

    const int S=(int)h.seq_len,H=(int)h.hidden,N=(int)h.num_heads;
    const int DN=(int)h.qk_nope_dim,DR=(int)h.qk_rope_dim,DV=(int)h.v_head_dim;
    const int R=(int)h.kv_lora_rank,I=(int)h.expert_intermediate,SI=(int)h.shared_intermediate;
    const int QO=N*(DN+DR), KVA=R+DR, KVB=N*(DN+DV);
    const size_t SH=(size_t)S*H;

    float *x=read_span(f,h.off_x,SH*4);
    uint16_t *inorm=read_span(f,h.off_input_norm,(size_t)H*2);
    uint16_t *qproj=read_span(f,h.off_q_proj,(size_t)QO*H*2);
    uint16_t *kva=read_span(f,h.off_kv_a,(size_t)KVA*H*2);
    uint16_t *kvan=read_span(f,h.off_kv_a_norm,(size_t)R*2);
    uint16_t *kvb=read_span(f,h.off_kv_b,(size_t)KVB*R*2);
    uint16_t *oproj=read_span(f,h.off_o_proj,(size_t)H*N*DV*2);
    uint16_t *pnorm=read_span(f,h.off_post_norm,(size_t)H*2);
    float *rw=read_span(f,h.off_router,(size_t)h.n_experts*H*4);
    float *bias=read_span(f,h.off_bias,(size_t)h.n_experts*4);
    uint16_t *sg=read_span(f,h.off_shared_gate,(size_t)SI*H*2);
    uint16_t *su=read_span(f,h.off_shared_up,(size_t)SI*H*2);
    uint16_t *sd=read_span(f,h.off_shared_down,(size_t)H*SI*2);
    float *e_norm1=read_span(f,h.off_expected_norm1,SH*4);
    float *e_attn=read_span(f,h.off_expected_attn,SH*4);
    float *e_resid1=read_span(f,h.off_expected_resid1,SH*4);
    float *e_norm2=read_span(f,h.off_expected_norm2,SH*4);
    int32_t *e_ids=read_span(f,h.off_expected_ids,(size_t)S*h.top_k*4);
    float *e_weights=read_span(f,h.off_expected_weights,(size_t)S*h.top_k*4);
    float *e_out=read_span(f,h.off_expected_out,SH*4);
    fclose(f);
    if(!x||!inorm||!qproj||!kva||!kvan||!kvb||!oproj||!pnorm||!rw||!bias||
       !sg||!su||!sd||!e_norm1||!e_attn||!e_resid1||!e_norm2||!e_ids||!e_weights||!e_out) {
        fprintf(stderr,"fixture OOM/read\n"); return 2;
    }

    float *norm1=malloc(SH*4), *attn=malloc(SH*4), *resid1=malloc(SH*4);
    float *norm2=malloc(SH*4), *out=malloc(SH*4), *moe=malloc((size_t)H*4);
    int *ids=malloc((size_t)h.top_k*sizeof(int));
    float *weights=malloc((size_t)h.top_k*4);
    const int maxI=I>SI?I:SI;
    float *scratch=malloc((size_t)(3*maxI+H)*4);
    if(!norm1||!attn||!resid1||!norm2||!out||!moe||!ids||!weights||!scratch){
        fprintf(stderr,"runtime OOM\n"); return 2;
    }

    for(int t=0;t<S;++t)
        kvl_rmsnorm_bf16(norm1+(size_t)t*H,x+(size_t)t*H,inorm,H,h.rms_eps);
    KvlMlaConfig ac={H,N,DN,DR,DV,R,h.rms_eps,h.rope_theta};
    KvlMlaBF16 aw={qproj,kva,kvan,kvb,oproj};
    if(kvl_mla_prefill_bf16(attn,norm1,S,&ac,&aw)!=0){fprintf(stderr,"MLA failed\n");return 1;}
    for(size_t i=0;i<SH;++i) resid1[i]=x[i]+attn[i];
    for(int t=0;t<S;++t)
        kvl_rmsnorm_bf16(norm2+(size_t)t*H,resid1+(size_t)t*H,pnorm,H,h.rms_eps);

    KvlExpertStore st;
    if(kvl_expert_store_open(&st,argv[1],argv[2],1)!=0){fprintf(stderr,"store open failed\n");return 2;}
    KvlExpertCache cache;
    size_t budget=(size_t)strtoull(argv[4],NULL,10);
    if(kvl_expert_cache_init(&cache,&st,budget)!=0){fprintf(stderr,"cache init failed\n");return 2;}
    KvlRouterConfig rc={H,(int)h.n_experts,(int)h.top_k,(int)h.n_group,(int)h.topk_group,
                        (int)h.norm_topk_prob,h.routed_scaling_factor};
    KvlMlpBF16 shared={sg,su,sd,SI};

    int id_ok=1; double max_w=0.0;
    for(int t=0;t<S;++t){
        if(kvl_moe_token_bf16(&cache,(int)h.layer,&rc,norm2+(size_t)t*H,rw,bias,I,
                              &shared,moe,ids,weights,scratch)!=0){
            fprintf(stderr,"MoE failed at token %d\n",t);return 1;
        }
        for(int i=0;i<H;++i) out[(size_t)t*H+i]=resid1[(size_t)t*H+i]+moe[i];
        const int32_t *ei=e_ids+(size_t)t*h.top_k;
        const float *ew=e_weights+(size_t)t*h.top_k;
        for(uint32_t i=0;i<h.top_k;++i){
            int found=-1;
            for(uint32_t j=0;j<h.top_k;++j) if(ids[j]==ei[i]){found=(int)j;break;}
            if(found<0){id_ok=0;continue;}
            double d=fabs((double)weights[found]-ew[i]); if(d>max_w)max_w=d;
        }
    }

    const double d_n1=max_abs(norm1,e_norm1,SH);
    const double d_attn=max_abs(attn,e_attn,SH);
    const double d_r1=max_abs(resid1,e_resid1,SH);
    const double d_n2=max_abs(norm2,e_norm2,SH);
    const double d_out=max_abs(out,e_out,SH);
    double rms=0.0; for(size_t i=0;i<SH;++i){double d=(double)out[i]-e_out[i];rms+=d*d;} rms=sqrt(rms/SH);
    printf("norm1_max=%.8g attn_max=%.8g resid1_max=%.8g norm2_max=%.8g router_ids=%s max_weight_abs=%.8g final_max=%.8g final_rms=%.8g direct_io=%s\n",
           d_n1,d_attn,d_r1,d_n2,id_ok?"OK":"BAD",max_w,d_out,rms,st.direct_io?"yes":"no");
    kvl_expert_cache_report(&cache);

    const int ok=id_ok && max_w<3e-5 && d_n1<5e-5 && d_attn<8e-4 && d_r1<8e-4 &&
                 d_n2<8e-4 && d_out<2e-3;
    kvl_expert_cache_close(&cache);kvl_expert_store_close(&st);
    free(x);free(inorm);free(qproj);free(kva);free(kvan);free(kvb);free(oproj);free(pnorm);
    free(rw);free(bias);free(sg);free(su);free(sd);free(e_norm1);free(e_attn);free(e_resid1);
    free(e_norm2);free(e_ids);free(e_weights);free(e_out);free(norm1);free(attn);free(resid1);
    free(norm2);free(out);free(moe);free(ids);free(weights);free(scratch);
    return ok?0:1;
}
