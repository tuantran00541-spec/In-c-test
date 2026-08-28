#include "kvl/mla_compressed_state.h"
#include "kvl/ops.h"
#include "kvl/trunk_store.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* V6 end-to-end correctness probe for the released Kimi-VL-A3B text architecture.
 * It compares a two-token full causal prefill with token-by-token decoding using one
 * persistent compressed MLA state per decoder layer. This is a probe, not model metadata. */
enum { S=2, H=2048, NH=16, DN=128, DR=64, DV=128, R=512,
       LN=27, V=163840, E=64, TOPK=6, DENSE_I=11264, EXP_I=1408, SHARED_I=2816 };

static int load_kind(KvlTrunkStore *ts,uint32_t L,uint32_t kind,KvlTrunkTensor *t){
    if(kvl_trunk_load(ts,L,kind,t)!=0){fprintf(stderr,"trunk load failed L=%u kind=%u\n",L,kind);return -1;}return 0;
}
static void expand(float *dst,const uint16_t *src,size_t n){for(size_t i=0;i<n;++i)dst[i]=kvl_bf16_to_f32(src[i]);}
static double maxabs(const float*a,const float*b,size_t n){double m=0;for(size_t i=0;i<n;++i){double d=fabs((double)a[i]-b[i]);if(d>m)m=d;}return m;}

static int layer_attention_prefill(KvlTrunkStore *ts,int L,const float *x,float *r1,float *n2){
    KvlTrunkTensor in={0},q={0},kva={0},kvan={0},kvb={0},o={0},pn={0};int rc=-1;
    if(load_kind(ts,L,KVL_TENSOR_INPUT_NORM,&in)||load_kind(ts,L,KVL_TENSOR_Q_PROJ,&q)||
       load_kind(ts,L,KVL_TENSOR_KV_A_PROJ,&kva)||load_kind(ts,L,KVL_TENSOR_KV_A_NORM,&kvan)||
       load_kind(ts,L,KVL_TENSOR_KV_B_PROJ,&kvb)||load_kind(ts,L,KVL_TENSOR_O_PROJ,&o)||
       load_kind(ts,L,KVL_TENSOR_POST_ATTN_NORM,&pn))goto done;
    float *n1=malloc((size_t)S*H*sizeof(float)),*a=malloc((size_t)S*H*sizeof(float));if(!n1||!a)goto done2;
    for(int t=0;t<S;++t)kvl_rmsnorm_bf16(n1+(size_t)t*H,x+(size_t)t*H,(const uint16_t*)in.data,H,1e-6f);
    KvlMlaConfig c={H,NH,DN,DR,DV,R,1e-6f,800000.0f};
    KvlMlaBF16 w={(const uint16_t*)q.data,(const uint16_t*)kva.data,(const uint16_t*)kvan.data,(const uint16_t*)kvb.data,(const uint16_t*)o.data};
    if(kvl_mla_prefill_bf16(a,n1,S,&c,&w)!=0)goto done2;
    for(int t=0;t<S;++t){for(int i=0;i<H;++i)r1[(size_t)t*H+i]=x[(size_t)t*H+i]+a[(size_t)t*H+i];kvl_rmsnorm_bf16(n2+(size_t)t*H,r1+(size_t)t*H,(const uint16_t*)pn.data,H,1e-6f);}
    rc=0;
done2:free(n1);free(a);
done:kvl_trunk_tensor_free(&in);kvl_trunk_tensor_free(&q);kvl_trunk_tensor_free(&kva);kvl_trunk_tensor_free(&kvan);kvl_trunk_tensor_free(&kvb);kvl_trunk_tensor_free(&o);kvl_trunk_tensor_free(&pn);return rc;
}

static int layer_attention_decode(KvlTrunkStore *ts,int L,const float *x,int pos,KvlMlaCompressedState *state,float *r1,float *n2){
    KvlTrunkTensor in={0},q={0},kva={0},kvan={0},kvb={0},o={0},pn={0};int rc=-1;float *n1=malloc((size_t)H*sizeof(float)),*a=malloc((size_t)H*sizeof(float));if(!n1||!a)goto done;
    if(load_kind(ts,L,KVL_TENSOR_INPUT_NORM,&in)||load_kind(ts,L,KVL_TENSOR_Q_PROJ,&q)||load_kind(ts,L,KVL_TENSOR_KV_A_PROJ,&kva)||load_kind(ts,L,KVL_TENSOR_KV_A_NORM,&kvan)||load_kind(ts,L,KVL_TENSOR_KV_B_PROJ,&kvb)||load_kind(ts,L,KVL_TENSOR_O_PROJ,&o)||load_kind(ts,L,KVL_TENSOR_POST_ATTN_NORM,&pn))goto done;
    kvl_rmsnorm_bf16(n1,x,(const uint16_t*)in.data,H,1e-6f);KvlMlaConfig c={H,NH,DN,DR,DV,R,1e-6f,800000.0f};KvlMlaBF16 w={(const uint16_t*)q.data,(const uint16_t*)kva.data,(const uint16_t*)kvan.data,(const uint16_t*)kvb.data,(const uint16_t*)o.data};
    if(kvl_mla_decode_compressed_bf16(a,n1,pos,&c,&w,state)!=0)goto done;
    for(int i=0;i<H;++i)r1[i]=x[i]+a[i];kvl_rmsnorm_bf16(n2,r1,(const uint16_t*)pn.data,H,1e-6f);rc=0;
done:free(n1);free(a);kvl_trunk_tensor_free(&in);kvl_trunk_tensor_free(&q);kvl_trunk_tensor_free(&kva);kvl_trunk_tensor_free(&kvan);kvl_trunk_tensor_free(&kvb);kvl_trunk_tensor_free(&o);kvl_trunk_tensor_free(&pn);return rc;
}

static int mlp_one(KvlTrunkStore *ts,KvlExpertCache *cache,int L,const float *n,float *y,float *router,float *bias,int *ids,float *ww,float *scratch){
    if(L==0){KvlTrunkTensor g={0},u={0},d={0};if(load_kind(ts,L,KVL_TENSOR_DENSE_GATE,&g)||load_kind(ts,L,KVL_TENSOR_DENSE_UP,&u)||load_kind(ts,L,KVL_TENSOR_DENSE_DOWN,&d))return -1;KvlMlpBF16 m={(const uint16_t*)g.data,(const uint16_t*)u.data,(const uint16_t*)d.data,DENSE_I};int rc=kvl_mlp_bf16(y,n,&m,H,scratch);kvl_trunk_tensor_free(&g);kvl_trunk_tensor_free(&u);kvl_trunk_tensor_free(&d);return rc;}
    KvlTrunkTensor rt={0},rb={0},sg={0},su={0},sd={0};if(load_kind(ts,L,KVL_TENSOR_ROUTER_WEIGHT,&rt)||load_kind(ts,L,KVL_TENSOR_ROUTER_BIAS,&rb)||load_kind(ts,L,KVL_TENSOR_SHARED_GATE,&sg)||load_kind(ts,L,KVL_TENSOR_SHARED_UP,&su)||load_kind(ts,L,KVL_TENSOR_SHARED_DOWN,&sd))return -1;
    expand(router,(const uint16_t*)rt.data,(size_t)E*H);expand(bias,(const uint16_t*)rb.data,E);KvlRouterConfig r={H,E,TOPK,1,1,1,2.446f};KvlMlpBF16 sh={(const uint16_t*)sg.data,(const uint16_t*)su.data,(const uint16_t*)sd.data,SHARED_I};int rc=kvl_moe_token_bf16(cache,L,&r,n,router,bias,EXP_I,&sh,y,ids,ww,scratch);kvl_trunk_tensor_free(&rt);kvl_trunk_tensor_free(&rb);kvl_trunk_tensor_free(&sg);kvl_trunk_tensor_free(&su);kvl_trunk_tensor_free(&sd);return rc;
}

int main(int argc,char **argv){
    if(argc!=6){fprintf(stderr,"usage: %s trunk.bin trunk.idx experts.bin experts.idx cache_bytes\n",argv[0]);return 2;}
    KvlTrunkStore ts;if(kvl_trunk_store_open(&ts,argv[1],argv[2],1)!=0)return 2;KvlExpertStore es;if(kvl_expert_store_open(&es,argv[3],argv[4],1)!=0)return 2;KvlExpertCache cache;if(kvl_expert_cache_init(&cache,&es,(size_t)strtoull(argv[5],NULL,10))!=0)return 2;
    const int token_ids[S]={1,1008};KvlTrunkTensor emb={0};if(load_kind(&ts,KVL_TRUNK_GLOBAL_LAYER,KVL_TENSOR_EMBED_TOKENS,&emb))return 2;float *initial=malloc((size_t)S*H*sizeof(float));for(int t=0;t<S;++t)expand(initial+(size_t)t*H,(const uint16_t*)emb.data+(size_t)token_ids[t]*H,H);kvl_trunk_tensor_free(&emb);
    float *pf=malloc((size_t)S*H*sizeof(float)),*next=malloc((size_t)S*H*sizeof(float)),*r1=malloc((size_t)S*H*sizeof(float)),*n2=malloc((size_t)S*H*sizeof(float)),*y=malloc((size_t)H*sizeof(float));memcpy(pf,initial,(size_t)S*H*sizeof(float));
    float *expected=malloc((size_t)LN*S*H*sizeof(float));float *router=malloc((size_t)E*H*sizeof(float)),*bias=malloc((size_t)E*sizeof(float));int *ids=malloc((size_t)TOPK*sizeof(int));float *ww=malloc((size_t)TOPK*sizeof(float));float *scratch=malloc((size_t)(3*DENSE_I+H)*sizeof(float));if(!initial||!pf||!next||!r1||!n2||!y||!expected||!router||!bias||!ids||!ww||!scratch)return 2;
    for(int L=0;L<LN;++L){if(layer_attention_prefill(&ts,L,pf,r1,n2)!=0)return 1;for(int t=0;t<S;++t){if(mlp_one(&ts,&cache,L,n2+(size_t)t*H,y,router,bias,ids,ww,scratch)!=0)return 1;for(int i=0;i<H;++i)next[(size_t)t*H+i]=r1[(size_t)t*H+i]+y[i];}memcpy(expected+(size_t)L*S*H,next,(size_t)S*H*sizeof(float));memcpy(pf,next,(size_t)S*H*sizeof(float));}

    KvlMlaCompressedState states[LN];KvlMlaConfig mc={H,NH,DN,DR,DV,R,1e-6f,800000.0f};size_t state_bytes=0;for(int L=0;L<LN;++L){if(kvl_mla_compressed_state_init(&states[L],&mc,S)!=0)return 1;state_bytes+=kvl_mla_compressed_state_bytes(&states[L]);}
    double worst=0;int worstL=-1,worstT=-1;float *x=malloc((size_t)H*sizeof(float)),*rr=malloc((size_t)H*sizeof(float)),*nn=malloc((size_t)H*sizeof(float));
    for(int t=0;t<S;++t){memcpy(x,initial+(size_t)t*H,(size_t)H*sizeof(float));for(int L=0;L<LN;++L){if(layer_attention_decode(&ts,L,x,t,&states[L],rr,nn)!=0)return 1;if(mlp_one(&ts,&cache,L,nn,y,router,bias,ids,ww,scratch)!=0)return 1;for(int i=0;i<H;++i)next[i]=rr[i]+y[i];double d=maxabs(next,expected+((size_t)L*S+t)*H,H);if(d>worst){worst=d;worstL=L;worstT=t;}memcpy(x,next,(size_t)H*sizeof(float));}}

    KvlTrunkTensor fn={0},lm={0};if(load_kind(&ts,KVL_TRUNK_GLOBAL_LAYER,KVL_TENSOR_FINAL_NORM,&fn)||load_kind(&ts,KVL_TRUNK_GLOBAL_LAYER,KVL_TENSOR_LM_HEAD,&lm))return 1;float *zpf=malloc((size_t)H*sizeof(float)),*zinc=malloc((size_t)H*sizeof(float));kvl_rmsnorm_bf16(zpf,pf+(size_t)(S-1)*H,(const uint16_t*)fn.data,H,1e-6f);kvl_rmsnorm_bf16(zinc,x,(const uint16_t*)fn.data,H,1e-6f);float *lpf=malloc((size_t)V*sizeof(float)),*linc=malloc((size_t)V*sizeof(float));kvl_matvec_bf16(lpf,zpf,(const uint16_t*)lm.data,H,V);kvl_matvec_bf16(linc,zinc,(const uint16_t*)lm.data,H,V);double ld=maxabs(lpf,linc,V),lr=0;int ap=0,ai=0;for(int i=0;i<V;++i){double d=(double)lpf[i]-linc[i];lr+=d*d;if(lpf[i]>lpf[ap])ap=i;if(linc[i]>linc[ai])ai=i;}lr=sqrt(lr/V);
    printf("tokens=%d compressed_state_bytes=%zu worst_layer=%d worst_token=%d hidden_max=%.9g logits_max=%.9g logits_rms=%.9g prefill_argmax=%d incremental_argmax=%d trunk_direct_io=%s expert_direct_io=%s\n",S,state_bytes,worstL,worstT,worst,ld,lr,ap,ai,ts.direct_io?"yes":"no",es.direct_io?"yes":"no");kvl_expert_cache_report(&cache);
    for(int L=0;L<LN;++L)kvl_mla_compressed_state_free(&states[L]);kvl_trunk_tensor_free(&fn);kvl_trunk_tensor_free(&lm);free(initial);free(pf);free(next);free(r1);free(n2);free(y);free(expected);free(router);free(bias);free(ids);free(ww);free(scratch);free(x);free(rr);free(nn);free(zpf);free(zinc);free(lpf);free(linc);kvl_expert_cache_close(&cache);kvl_expert_store_close(&es);kvl_trunk_store_close(&ts);
    return (worst<5e-3&&ld<1e-2&&ap==ai)?0:1;
}
