#include "kvl/ops.h"
#include "kvl/trunk_store.h"
#include "kvl/v4_fixture.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_span(FILE *f, uint64_t off, size_t n) {
    void *p=malloc(n?n:1); if(!p) return NULL;
    if(fseek(f,(long)off,SEEK_SET)!=0 || fread(p,1,n,f)!=n){free(p);return NULL;}
    return p;
}
static double max_abs(const float *a,const float *b,size_t n){double m=0;for(size_t i=0;i<n;++i){double d=fabs((double)a[i]-b[i]);if(d>m)m=d;}return m;}

static int load_kind(KvlTrunkStore *ts,uint32_t layer,uint32_t kind,KvlTrunkTensor *t){
    if(kvl_trunk_load(ts,layer,kind,t)!=0){fprintf(stderr,"trunk missing/load failed layer=%u kind=%u\n",layer,kind);return -1;}return 0;
}
static void expand_bf16(float *dst,const uint16_t *src,size_t n){for(size_t i=0;i<n;++i)dst[i]=kvl_bf16_to_f32(src[i]);}

static int run_common_attention(KvlTrunkStore *ts,uint32_t layer,const KvlV4FixtureHeader *h,
                                const float *x,float *norm1,float *attn,float *resid1,float *norm2){
    const int S=(int)h->seq_len,H=(int)h->hidden,N=(int)h->num_heads;
    const int DN=(int)h->qk_nope_dim,DR=(int)h->qk_rope_dim,DV=(int)h->v_head_dim,R=(int)h->kv_lora_rank;
    KvlTrunkTensor in={0},q={0},kva={0},kvan={0},kvb={0},o={0},pn={0};
    int ok=0;
    if(load_kind(ts,layer,KVL_TENSOR_INPUT_NORM,&in)||load_kind(ts,layer,KVL_TENSOR_Q_PROJ,&q)||
       load_kind(ts,layer,KVL_TENSOR_KV_A_PROJ,&kva)||load_kind(ts,layer,KVL_TENSOR_KV_A_NORM,&kvan)||
       load_kind(ts,layer,KVL_TENSOR_KV_B_PROJ,&kvb)||load_kind(ts,layer,KVL_TENSOR_O_PROJ,&o)||
       load_kind(ts,layer,KVL_TENSOR_POST_ATTN_NORM,&pn)) goto done;
    for(int t=0;t<S;++t) kvl_rmsnorm_bf16(norm1+(size_t)t*H,x+(size_t)t*H,(uint16_t*)in.data,H,h->rms_eps);
    KvlMlaConfig ac={H,N,DN,DR,DV,R,h->rms_eps,h->rope_theta};
    KvlMlaBF16 aw={(uint16_t*)q.data,(uint16_t*)kva.data,(uint16_t*)kvan.data,(uint16_t*)kvb.data,(uint16_t*)o.data};
    if(kvl_mla_prefill_bf16(attn,norm1,S,&ac,&aw)!=0) goto done;
    for(size_t i=0;i<(size_t)S*H;++i) resid1[i]=x[i]+attn[i];
    for(int t=0;t<S;++t) kvl_rmsnorm_bf16(norm2+(size_t)t*H,resid1+(size_t)t*H,(uint16_t*)pn.data,H,h->rms_eps);
    ok=1;
done:
    kvl_trunk_tensor_free(&in);kvl_trunk_tensor_free(&q);kvl_trunk_tensor_free(&kva);kvl_trunk_tensor_free(&kvan);
    kvl_trunk_tensor_free(&kvb);kvl_trunk_tensor_free(&o);kvl_trunk_tensor_free(&pn);
    return ok?0:-1;
}

int main(int argc,char **argv){
    if(argc!=7){fprintf(stderr,"usage: %s trunk.bin trunk.idx experts.bin experts.idx fixture.bin cache_bytes\n",argv[0]);return 2;}
    FILE *f=fopen(argv[5],"rb");if(!f){perror("fixture");return 2;}
    KvlV4FixtureHeader h;
    if(fread(&h,1,sizeof h,f)!=sizeof h||memcmp(h.magic,KVL_V4_FIXTURE_MAGIC,8)||h.version!=1||h.n_layers!=2){fprintf(stderr,"bad V4 fixture\n");fclose(f);return 2;}
    const int S=(int)h.seq_len,H=(int)h.hidden;const size_t SH=(size_t)S*H;
    float *x=read_span(f,h.off_x,SH*4),*e0=read_span(f,h.off_expected_after_dense,SH*4),*ef=read_span(f,h.off_expected_final,SH*4);
    int32_t *eids=read_span(f,h.off_expected_ids,(size_t)S*h.top_k*4);float *ew=read_span(f,h.off_expected_weights,(size_t)S*h.top_k*4);fclose(f);
    if(!x||!e0||!ef||!eids||!ew){fprintf(stderr,"fixture read failed\n");return 2;}
    float *cur=malloc(SH*4),*n1=malloc(SH*4),*attn=malloc(SH*4),*r1=malloc(SH*4),*n2=malloc(SH*4),*next=malloc(SH*4),*mlp=malloc((size_t)H*4);
    const int maxI=h.dense_intermediate>h.shared_intermediate?(int)h.dense_intermediate:(int)h.shared_intermediate;
    float *scratch=malloc((size_t)(3*maxI+H)*4);int *ids=malloc((size_t)h.top_k*sizeof(int));float *weights=malloc((size_t)h.top_k*4);
    if(!cur||!n1||!attn||!r1||!n2||!next||!mlp||!scratch||!ids||!weights){fprintf(stderr,"OOM\n");return 2;} memcpy(cur,x,SH*4);

    KvlTrunkStore ts;if(kvl_trunk_store_open(&ts,argv[1],argv[2],1)!=0){fprintf(stderr,"trunk open failed\n");return 2;}
    KvlExpertStore es;if(kvl_expert_store_open(&es,argv[3],argv[4],1)!=0){fprintf(stderr,"expert store open failed\n");return 2;}
    KvlExpertCache cache;size_t budget=(size_t)strtoull(argv[6],NULL,10);if(kvl_expert_cache_init(&cache,&es,budget)!=0){fprintf(stderr,"cache init failed\n");return 2;}

    const uint32_t L0=h.first_layer,L1=h.first_layer+1;
    if(run_common_attention(&ts,L0,&h,cur,n1,attn,r1,n2)!=0){fprintf(stderr,"layer0 attention failed\n");return 1;}
    KvlTrunkTensor dg={0},du={0},dd={0};
    if(load_kind(&ts,L0,KVL_TENSOR_DENSE_GATE,&dg)||load_kind(&ts,L0,KVL_TENSOR_DENSE_UP,&du)||load_kind(&ts,L0,KVL_TENSOR_DENSE_DOWN,&dd)) return 1;
    KvlMlpBF16 dense={(uint16_t*)dg.data,(uint16_t*)du.data,(uint16_t*)dd.data,(int)h.dense_intermediate};
    for(int t=0;t<S;++t){if(kvl_mlp_bf16(mlp,n2+(size_t)t*H,&dense,H,scratch)!=0)return 1;for(int i=0;i<H;++i)next[(size_t)t*H+i]=r1[(size_t)t*H+i]+mlp[i];}
    const double d0=max_abs(next,e0,SH);memcpy(cur,next,SH*4);
    kvl_trunk_tensor_free(&dg);kvl_trunk_tensor_free(&du);kvl_trunk_tensor_free(&dd);

    if(run_common_attention(&ts,L1,&h,cur,n1,attn,r1,n2)!=0){fprintf(stderr,"layer1 attention failed\n");return 1;}
    KvlTrunkTensor rt={0},rb={0},sg={0},su={0},sd={0};
    if(load_kind(&ts,L1,KVL_TENSOR_ROUTER_WEIGHT,&rt)||load_kind(&ts,L1,KVL_TENSOR_ROUTER_BIAS,&rb)||
       load_kind(&ts,L1,KVL_TENSOR_SHARED_GATE,&sg)||load_kind(&ts,L1,KVL_TENSOR_SHARED_UP,&su)||load_kind(&ts,L1,KVL_TENSOR_SHARED_DOWN,&sd)) return 1;
    float *router=malloc((size_t)h.n_experts*H*4),*bias=malloc((size_t)h.n_experts*4);if(!router||!bias)return 2;
    expand_bf16(router,(uint16_t*)rt.data,(size_t)h.n_experts*H);expand_bf16(bias,(uint16_t*)rb.data,h.n_experts);
    KvlRouterConfig rc={H,(int)h.n_experts,(int)h.top_k,(int)h.n_group,(int)h.topk_group,(int)h.norm_topk_prob,h.routed_scaling_factor};
    KvlMlpBF16 shared={(uint16_t*)sg.data,(uint16_t*)su.data,(uint16_t*)sd.data,(int)h.shared_intermediate};
    int id_ok=1;double max_w=0;
    for(int t=0;t<S;++t){
        if(kvl_moe_token_bf16(&cache,(int)L1,&rc,n2+(size_t)t*H,router,bias,(int)h.expert_intermediate,&shared,mlp,ids,weights,scratch)!=0)return 1;
        for(int i=0;i<H;++i)next[(size_t)t*H+i]=r1[(size_t)t*H+i]+mlp[i];
        const int32_t *ei=eids+(size_t)t*h.top_k;const float *we=ew+(size_t)t*h.top_k;
        for(uint32_t i=0;i<h.top_k;++i){int found=-1;for(uint32_t j=0;j<h.top_k;++j)if(ids[j]==ei[i]){found=(int)j;break;}if(found<0){id_ok=0;continue;}double d=fabs((double)weights[found]-we[i]);if(d>max_w)max_w=d;}
    }
    const double df=max_abs(next,ef,SH);double rms=0;for(size_t i=0;i<SH;++i){double d=(double)next[i]-ef[i];rms+=d*d;}rms=sqrt(rms/SH);
    printf("dense_layer_max=%.8g router_ids=%s max_weight_abs=%.8g final_max=%.8g final_rms=%.8g trunk_direct_io=%s expert_direct_io=%s\n",
           d0,id_ok?"OK":"BAD",max_w,df,rms,ts.direct_io?"yes":"no",es.direct_io?"yes":"no");
    kvl_expert_cache_report(&cache);
    const int ok=d0<2e-3&&id_ok&&max_w<3e-5&&df<3e-3;
    free(router);free(bias);kvl_trunk_tensor_free(&rt);kvl_trunk_tensor_free(&rb);kvl_trunk_tensor_free(&sg);kvl_trunk_tensor_free(&su);kvl_trunk_tensor_free(&sd);
    kvl_expert_cache_close(&cache);kvl_expert_store_close(&es);kvl_trunk_store_close(&ts);
    free(x);free(e0);free(ef);free(eids);free(ew);free(cur);free(n1);free(attn);free(r1);free(n2);free(next);free(mlp);free(scratch);free(ids);free(weights);
    return ok?0:1;
}
