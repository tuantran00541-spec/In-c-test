#include "kvl/mla_state.h"
#include "kvl/mla_compressed_state.h"
#include "kvl/trunk_store.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int load(KvlTrunkStore *s,unsigned L,unsigned kind,KvlTrunkTensor *t){
    if(kvl_trunk_load(s,L,kind,t)!=0){fprintf(stderr,"load failed L=%u kind=%u\n",L,kind);return -1;}return 0;
}
static double maxabs(const float*a,const float*b,size_t n){double m=0;for(size_t i=0;i<n;++i){double d=fabs((double)a[i]-b[i]);if(d>m)m=d;}return m;}

int main(int argc,char **argv){
    if(argc!=4){fprintf(stderr,"usage: %s trunk.bin trunk.idx layer\n",argv[0]);return 2;}
    const unsigned L=(unsigned)strtoul(argv[3],NULL,10);
    enum{S=4,H=2048,NH=16,DN=128,DR=64,DV=128,R=512};
    KvlMlaConfig cfg={H,NH,DN,DR,DV,R,1e-6f,800000.0f};
    KvlTrunkStore ts;if(kvl_trunk_store_open(&ts,argv[1],argv[2],1)!=0)return 2;
    KvlTrunkTensor q={0},kva={0},kvan={0},kvb={0},o={0};
    if(load(&ts,L,KVL_TENSOR_Q_PROJ,&q)||load(&ts,L,KVL_TENSOR_KV_A_PROJ,&kva)||
       load(&ts,L,KVL_TENSOR_KV_A_NORM,&kvan)||load(&ts,L,KVL_TENSOR_KV_B_PROJ,&kvb)||
       load(&ts,L,KVL_TENSOR_O_PROJ,&o))return 2;
    KvlMlaBF16 w={(const uint16_t*)q.data,(const uint16_t*)kva.data,(const uint16_t*)kvan.data,(const uint16_t*)kvb.data,(const uint16_t*)o.data};
    float *x=malloc((size_t)S*H*sizeof(float)),*pf=calloc((size_t)S*H,sizeof(float));
    float *ex=calloc((size_t)S*H,sizeof(float)),*co=calloc((size_t)S*H,sizeof(float));
    if(!x||!pf||!ex||!co)return 2;
    for(int t=0;t<S;++t)for(int i=0;i<H;++i)x[(size_t)t*H+i]=0.08f*(float)sin((double)(t+1)*(i+3)*0.0017)+0.02f*(float)cos((double)(i+11)*0.0031);
    if(kvl_mla_prefill_bf16(pf,x,S,&cfg,&w)!=0)return 1;
    KvlMlaState es;if(kvl_mla_state_init(&es,&cfg,S)!=0)return 1;
    KvlMlaCompressedState cs;if(kvl_mla_compressed_state_init(&cs,&cfg,S)!=0)return 1;
    for(int t=0;t<S;++t){
        if(kvl_mla_decode_bf16(ex+(size_t)t*H,x+(size_t)t*H,t,&cfg,&w,&es)!=0)return 1;
        if(kvl_mla_decode_compressed_bf16(co+(size_t)t*H,x+(size_t)t*H,t,&cfg,&w,&cs)!=0)return 1;
    }
    const double de=maxabs(pf,ex,(size_t)S*H),dc=maxabs(pf,co,(size_t)S*H);
    const size_t eb=kvl_mla_state_bytes(&es),cb=kvl_mla_compressed_state_bytes(&cs);
    printf("layer=%u seq=%d expanded_bytes=%zu compressed_bytes=%zu ratio=%.3fx expanded_max=%.9g compressed_max=%.9g direct_io=%s\n",
           L,S,eb,cb,(double)eb/(double)cb,de,dc,ts.direct_io?"yes":"no");
    kvl_mla_state_free(&es);kvl_mla_compressed_state_free(&cs);
    kvl_trunk_tensor_free(&q);kvl_trunk_tensor_free(&kva);kvl_trunk_tensor_free(&kvan);kvl_trunk_tensor_free(&kvb);kvl_trunk_tensor_free(&o);kvl_trunk_store_close(&ts);
    free(x);free(pf);free(ex);free(co);
    if(de>2e-5||dc>2e-4)return 1;
    puts("PASS: official Kimi-VL MLA weights match prefill with compressed incremental state");return 0;
}
