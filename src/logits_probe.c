#include "kvl/ops.h"
#include "kvl/trunk_store.h"
#include "kvl/v5_fixture.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_span(FILE *f, uint64_t off, size_t n) {
    void *p=malloc(n?n:1); if(!p) return NULL;
    if(fseek(f,(long)off,SEEK_SET)!=0 || fread(p,1,n,f)!=n){free(p);return NULL;} return p;
}
static double max_abs(const float *a,const float *b,size_t n){double m=0;for(size_t i=0;i<n;++i){double d=fabs((double)a[i]-b[i]);if(d>m)m=d;}return m;}
static void expand_bf16(float *dst,const uint16_t *src,size_t n){for(size_t i=0;i<n;++i)dst[i]=kvl_bf16_to_f32(src[i]);}
static int load_kind(KvlTrunkStore *ts,uint32_t layer,uint32_t kind,KvlTrunkTensor *t){
    if(kvl_trunk_load(ts,layer,kind,t)!=0){fprintf(stderr,"trunk load failed layer=%u kind=%u\n",layer,kind);return -1;}return 0;
}

static int attention_one(KvlTrunkStore *ts,uint32_t layer,const KvlV5FixtureHeader *h,
                         const float *x,float *norm1,float *attn,float *resid1,float *norm2){
    const int H=(int)h->hidden,N=(int)h->num_heads,DN=(int)h->qk_nope_dim,DR=(int)h->qk_rope_dim,DV=(int)h->v_head_dim,R=(int)h->kv_lora_rank;
    KvlTrunkTensor in={0},q={0},kva={0},kvan={0},kvb={0},o={0},pn={0}; int ok=0;
    if(load_kind(ts,layer,KVL_TENSOR_INPUT_NORM,&in)||load_kind(ts,layer,KVL_TENSOR_Q_PROJ,&q)||
       load_kind(ts,layer,KVL_TENSOR_KV_A_PROJ,&kva)||load_kind(ts,layer,KVL_TENSOR_KV_A_NORM,&kvan)||
       load_kind(ts,layer,KVL_TENSOR_KV_B_PROJ,&kvb)||load_kind(ts,layer,KVL_TENSOR_O_PROJ,&o)||
       load_kind(ts,layer,KVL_TENSOR_POST_ATTN_NORM,&pn)) goto done;
    kvl_rmsnorm_bf16(norm1,x,(const uint16_t*)in.data,H,h->rms_eps);
    KvlMlaConfig ac={H,N,DN,DR,DV,R,h->rms_eps,h->rope_theta};
    KvlMlaBF16 aw={(const uint16_t*)q.data,(const uint16_t*)kva.data,(const uint16_t*)kvan.data,(const uint16_t*)kvb.data,(const uint16_t*)o.data};
    if(kvl_mla_prefill_bf16(attn,norm1,1,&ac,&aw)!=0) goto done;
    for(int i=0;i<H;++i)resid1[i]=x[i]+attn[i];
    kvl_rmsnorm_bf16(norm2,resid1,(const uint16_t*)pn.data,H,h->rms_eps); ok=1;
done:
    kvl_trunk_tensor_free(&in);kvl_trunk_tensor_free(&q);kvl_trunk_tensor_free(&kva);kvl_trunk_tensor_free(&kvan);kvl_trunk_tensor_free(&kvb);kvl_trunk_tensor_free(&o);kvl_trunk_tensor_free(&pn);
    return ok?0:-1;
}

int main(int argc,char **argv){
    if(argc!=7){fprintf(stderr,"usage: %s trunk.bin trunk.idx experts.bin experts.idx fixture.bin cache_bytes\n",argv[0]);return 2;}
    FILE *ff=fopen(argv[5],"rb"); if(!ff){perror("fixture");return 2;} KvlV5FixtureHeader h;
    if(fread(&h,1,sizeof h,ff)!=sizeof h||memcmp(h.magic,KVL_V5_FIXTURE_MAGIC,8)||h.version!=1){fprintf(stderr,"bad V5 fixture\n");return 2;}
    const int H=(int)h.hidden,V=(int)h.vocab_size,LN=(int)h.n_layers;
    float *expected_layers=read_span(ff,h.off_expected_layers,(size_t)LN*H*sizeof(float));
    float *expected_logits=read_span(ff,h.off_expected_logits,(size_t)V*sizeof(float)); fclose(ff);
    if(!expected_layers||!expected_logits)return 2;
    float *x=calloc((size_t)H,sizeof(float)),*norm1=malloc((size_t)H*4),*attn=malloc((size_t)H*4),*r1=malloc((size_t)H*4),*norm2=malloc((size_t)H*4),*mlp=malloc((size_t)H*4),*next=malloc((size_t)H*4);
    const int maxI=h.dense_intermediate>h.shared_intermediate?(int)h.dense_intermediate:(int)h.shared_intermediate;
    float *scratch=malloc((size_t)(3*maxI+H)*sizeof(float)),*router=malloc((size_t)h.n_experts*H*sizeof(float)),*bias=malloc((size_t)h.n_experts*sizeof(float));
    int *ids=malloc((size_t)h.top_k*sizeof(int));float *weights=malloc((size_t)h.top_k*sizeof(float));
    if(!x||!norm1||!attn||!r1||!norm2||!mlp||!next||!scratch||!router||!bias||!ids||!weights)return 2;
    KvlTrunkStore ts; if(kvl_trunk_store_open(&ts,argv[1],argv[2],1)!=0){fprintf(stderr,"trunk open failed\n");return 2;}
    KvlExpertStore es; if(kvl_expert_store_open(&es,argv[3],argv[4],1)!=0){fprintf(stderr,"expert store open failed\n");return 2;}
    KvlExpertCache cache; size_t budget=(size_t)strtoull(argv[6],NULL,10); if(kvl_expert_cache_init(&cache,&es,budget)!=0)return 2;

    KvlTrunkTensor emb={0}; if(load_kind(&ts,KVL_TRUNK_GLOBAL_LAYER,KVL_TENSOR_EMBED_TOKENS,&emb))return 1;
    if(emb.record->ndim!=2||emb.record->dims[0]!=(uint32_t)V||emb.record->dims[1]!=(uint32_t)H||h.token_id>=(uint32_t)V){fprintf(stderr,"bad embedding shape/token\n");return 1;}
    const uint16_t *erow=(const uint16_t*)emb.data+(size_t)h.token_id*H; expand_bf16(x,erow,H); kvl_trunk_tensor_free(&emb);

    double worst=0.0;int worst_layer=-1;
    for(int L=0;L<LN;++L){
        if(attention_one(&ts,(uint32_t)L,&h,x,norm1,attn,r1,norm2)!=0)return 1;
        const int is_moe=(L>=(int)h.first_k_dense_replace)&&((L%(int)h.moe_layer_freq)==0);
        if(!is_moe){
            KvlTrunkTensor g={0},u={0},d={0}; if(load_kind(&ts,L,KVL_TENSOR_DENSE_GATE,&g)||load_kind(&ts,L,KVL_TENSOR_DENSE_UP,&u)||load_kind(&ts,L,KVL_TENSOR_DENSE_DOWN,&d))return 1;
            KvlMlpBF16 z={(const uint16_t*)g.data,(const uint16_t*)u.data,(const uint16_t*)d.data,(int)h.dense_intermediate};
            if(kvl_mlp_bf16(mlp,norm2,&z,H,scratch)!=0)return 1;
            kvl_trunk_tensor_free(&g);kvl_trunk_tensor_free(&u);kvl_trunk_tensor_free(&d);
        }else{
            KvlTrunkTensor rt={0},rb={0},sg={0},su={0},sd={0};
            if(load_kind(&ts,L,KVL_TENSOR_ROUTER_WEIGHT,&rt)||load_kind(&ts,L,KVL_TENSOR_ROUTER_BIAS,&rb)||load_kind(&ts,L,KVL_TENSOR_SHARED_GATE,&sg)||load_kind(&ts,L,KVL_TENSOR_SHARED_UP,&su)||load_kind(&ts,L,KVL_TENSOR_SHARED_DOWN,&sd))return 1;
            expand_bf16(router,(const uint16_t*)rt.data,(size_t)h.n_experts*H);expand_bf16(bias,(const uint16_t*)rb.data,h.n_experts);
            KvlRouterConfig rc={H,(int)h.n_experts,(int)h.top_k,(int)h.n_group,(int)h.topk_group,(int)h.norm_topk_prob,h.routed_scaling_factor};
            KvlMlpBF16 shared={(const uint16_t*)sg.data,(const uint16_t*)su.data,(const uint16_t*)sd.data,(int)h.shared_intermediate};
            if(kvl_moe_token_bf16(&cache,L,&rc,norm2,router,bias,(int)h.expert_intermediate,&shared,mlp,ids,weights,scratch)!=0)return 1;
            kvl_trunk_tensor_free(&rt);kvl_trunk_tensor_free(&rb);kvl_trunk_tensor_free(&sg);kvl_trunk_tensor_free(&su);kvl_trunk_tensor_free(&sd);
        }
        for(int i=0;i<H;++i)next[i]=r1[i]+mlp[i];
        const double d=max_abs(next,expected_layers+(size_t)L*H,H); if(d>worst){worst=d;worst_layer=L;}
        if(d>3e-3){fprintf(stderr,"layer %d mismatch max=%.8g\n",L,d);return 1;}
        memcpy(x,next,(size_t)H*sizeof(float));
    }

    KvlTrunkTensor fn={0};if(load_kind(&ts,KVL_TRUNK_GLOBAL_LAYER,KVL_TENSOR_FINAL_NORM,&fn))return 1;
    kvl_rmsnorm_bf16(norm1,x,(const uint16_t*)fn.data,H,h.rms_eps);kvl_trunk_tensor_free(&fn);
    KvlTrunkTensor lm={0};if(load_kind(&ts,KVL_TRUNK_GLOBAL_LAYER,KVL_TENSOR_LM_HEAD,&lm))return 1;
    if(lm.record->ndim!=2||lm.record->dims[0]!=(uint32_t)V||lm.record->dims[1]!=(uint32_t)H){fprintf(stderr,"bad lm head shape\n");return 1;}
    float *logits=malloc((size_t)V*sizeof(float));if(!logits)return 2; kvl_matvec_bf16(logits,norm1,(const uint16_t*)lm.data,H,V);kvl_trunk_tensor_free(&lm);
    const double lmax=max_abs(logits,expected_logits,V);double lrms=0;int ai=0,bi=0;for(int i=0;i<V;++i){double d=(double)logits[i]-expected_logits[i];lrms+=d*d;if(logits[i]>logits[ai])ai=i;if(expected_logits[i]>expected_logits[bi])bi=i;}lrms=sqrt(lrms/V);
    printf("layers=%d worst_layer=%d layer_max=%.8g logits_max=%.8g logits_rms=%.8g argmax=%d expected_argmax=%d trunk_direct_io=%s expert_direct_io=%s\n",LN,worst_layer,worst,lmax,lrms,ai,bi,ts.direct_io?"yes":"no",es.direct_io?"yes":"no");
    kvl_expert_cache_report(&cache);
    const int ok=worst<3e-3&&lmax<5e-3&&ai==bi;
    free(logits);free(expected_layers);free(expected_logits);free(x);free(norm1);free(attn);free(r1);free(norm2);free(mlp);free(next);free(scratch);free(router);free(bias);free(ids);free(weights);
    kvl_expert_cache_close(&cache);kvl_expert_store_close(&es);kvl_trunk_store_close(&ts);return ok?0:1;
}
