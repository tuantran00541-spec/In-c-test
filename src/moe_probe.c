#include "kvl/ops.h"
#include "kvl/v2_fixture.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_span(FILE *f, uint64_t off, size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) return NULL;
    if (fseek(f, (long)off, SEEK_SET) != 0 || fread(p, 1, n, f) != n) { free(p); return NULL; }
    return p;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s experts.bin experts.idx fixture.bin cache_bytes\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[3], "rb");
    if (!f) { perror("fixture"); return 2; }
    KvlV2FixtureHeader h;
    if (fread(&h, 1, sizeof h, f) != sizeof h || memcmp(h.magic, KVL_V2_FIXTURE_MAGIC, 8) || h.version != 1) {
        fprintf(stderr, "bad V2 fixture\n"); fclose(f); return 2;
    }
    const int H=(int)h.hidden, I=(int)h.expert_intermediate, SI=(int)h.shared_intermediate;
    float *x=read_span(f,h.off_x,(size_t)H*4);
    float *rw=read_span(f,h.off_router,(size_t)h.n_experts*H*4);
    float *bias=read_span(f,h.off_bias,(size_t)h.n_experts*4);
    uint16_t *sg=read_span(f,h.off_shared_gate,(size_t)SI*H*2);
    uint16_t *su=read_span(f,h.off_shared_up,(size_t)SI*H*2);
    uint16_t *sd=read_span(f,h.off_shared_down,(size_t)H*SI*2);
    int32_t *exp_ids=read_span(f,h.off_expected_ids,(size_t)h.top_k*4);
    float *exp_w=read_span(f,h.off_expected_weights,(size_t)h.top_k*4);
    float *exp_out=read_span(f,h.off_expected_out,(size_t)H*4);
    fclose(f);
    if(!x||!rw||!bias||!sg||!su||!sd||!exp_ids||!exp_w||!exp_out){fprintf(stderr,"fixture OOM/read\n");return 2;}

    KvlExpertStore st;
    if (kvl_expert_store_open(&st, argv[1], argv[2], 1) != 0) { fprintf(stderr,"store open failed\n"); return 2; }
    size_t budget=(size_t)strtoull(argv[4],NULL,10);
    KvlExpertCache cache;
    if(kvl_expert_cache_init(&cache,&st,budget)!=0){fprintf(stderr,"cache init failed\n");return 2;}

    KvlRouterConfig rc={H,(int)h.n_experts,(int)h.top_k,(int)h.n_group,(int)h.topk_group,
                        (int)h.norm_topk_prob,h.routed_scaling_factor};
    KvlMlpBF16 shared={sg,su,sd,SI};
    int *ids=(int*)malloc((size_t)h.top_k*sizeof(int));
    float *weights=(float*)malloc((size_t)h.top_k*sizeof(float));
    float *out=(float*)malloc((size_t)H*sizeof(float));
    const int maxI=I>SI?I:SI;
    float *scratch=(float*)malloc((size_t)(3*maxI+H)*sizeof(float));
    if(!ids||!weights||!out||!scratch){fprintf(stderr,"OOM\n");return 2;}

    int rc_run=kvl_moe_token_bf16(&cache,(int)h.layer,&rc,x,rw,bias,I,&shared,out,ids,weights,scratch);
    if(rc_run!=0){fprintf(stderr,"moe forward failed\n");return 1;}

    /* Official torch.topk(sorted=False) does not promise ordering. Compare id->weight maps. */
    double max_w=0.0;
    int id_ok=1;
    for(uint32_t i=0;i<h.top_k;i++){
        int found=-1;
        for(uint32_t j=0;j<h.top_k;j++) if(ids[j]==exp_ids[i]){found=(int)j;break;}
        if(found<0){id_ok=0;continue;}
        double d=fabs((double)weights[found]-exp_w[i]); if(d>max_w)max_w=d;
    }
    double max_out=0.0, rms=0.0, signal=0.0;
    for(int i=0;i<H;i++){
        double d=(double)out[i]-exp_out[i]; double a=fabs(d);
        if(a>max_out)max_out=a; rms+=d*d; signal+=(double)exp_out[i]*exp_out[i];
    }
    rms=sqrt(rms/H); signal=sqrt(signal/H);
    const double rel_rms=rms/(signal+1e-30);
    printf("router_ids=%s max_weight_abs=%.8g moe_max_abs=%.8g moe_rms=%.8g moe_rel_rms=%.8g dtype=%u direct_io=%s\n",
           id_ok?"OK":"BAD",max_w,max_out,rms,rel_rms,st.hdr.dtype,st.direct_io?"yes":"no");
    kvl_expert_cache_report(&cache);
    const int ok = st.hdr.dtype == KVL_DTYPE_Q8_ROW
        ? (id_ok && max_w<2e-5 && rel_rms<0.05)
        : (id_ok && max_w<2e-5 && max_out<3e-4);

    free(x);free(rw);free(bias);free(sg);free(su);free(sd);free(exp_ids);free(exp_w);free(exp_out);
    free(ids);free(weights);free(out);free(scratch);
    kvl_expert_cache_close(&cache);kvl_expert_store_close(&st);
    return ok?0:1;
}
