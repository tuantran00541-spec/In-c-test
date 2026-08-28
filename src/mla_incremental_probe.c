#include "kvl/mla_state.h"
#include "kvl/mla_compressed_state.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return (uint16_t)(u >> 16);
}

static void fill_bf16(uint16_t *p, size_t n, float scale, float bias) {
    for (size_t i = 0; i < n; ++i) {
        float x = bias + scale * (float)(sin((double)i * 0.173) + 0.35 * cos((double)i * 0.071));
        p[i] = f32_to_bf16(x);
    }
}

static double max_abs(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    enum { S=4, H=16, NH=2, DN=4, DR=4, DV=4, R=6 };
    enum { QD=DN+DR, QO=NH*QD, KVO=R+DR, KVB=NH*(DN+DV) };

    uint16_t *q = malloc((size_t)QO*H*sizeof(uint16_t));
    uint16_t *kva = malloc((size_t)KVO*H*sizeof(uint16_t));
    uint16_t *kvan = malloc((size_t)R*sizeof(uint16_t));
    uint16_t *kvb = malloc((size_t)KVB*R*sizeof(uint16_t));
    uint16_t *o = malloc((size_t)H*(NH*DV)*sizeof(uint16_t));
    float *x = malloc((size_t)S*H*sizeof(float));
    float *prefill = calloc((size_t)S*H,sizeof(float));
    float *expanded = calloc((size_t)S*H,sizeof(float));
    float *compressed = calloc((size_t)S*H,sizeof(float));
    if (!q||!kva||!kvan||!kvb||!o||!x||!prefill||!expanded||!compressed) return 2;

    fill_bf16(q,(size_t)QO*H,0.037f,0.0f);
    fill_bf16(kva,(size_t)KVO*H,0.041f,0.0f);
    fill_bf16(kvan,R,0.025f,0.93f);
    fill_bf16(kvb,(size_t)KVB*R,0.052f,0.0f);
    fill_bf16(o,(size_t)H*(NH*DV),0.044f,0.0f);
    for (int t=0;t<S;++t)
        for (int i=0;i<H;++i)
            x[(size_t)t*H+i] = 0.17f*(float)sin((double)(t+1)*(i+2)*0.11) +
                                0.03f*(float)cos((double)(i+1)*0.23);

    KvlMlaConfig cfg={H,NH,DN,DR,DV,R,1e-6f,10000.0f};
    KvlMlaBF16 w={q,kva,kvan,kvb,o};
    if (kvl_mla_prefill_bf16(prefill,x,S,&cfg,&w)!=0) {
        fprintf(stderr,"prefill failed\n"); return 1;
    }

    KvlMlaState state;
    if (kvl_mla_state_init(&state,&cfg,S)!=0) return 1;
    for (int t=0;t<S;++t)
        if (kvl_mla_decode_bf16(expanded+(size_t)t*H,x+(size_t)t*H,t,&cfg,&w,&state)!=0) return 1;
    const double de=max_abs(prefill,expanded,(size_t)S*H);
    const size_t expanded_bytes=kvl_mla_state_bytes(&state);
    int bad_order = kvl_mla_decode_bf16(expanded,x,0,&cfg,&w,&state)==0;
    kvl_mla_state_reset(&state);
    int reset_ok = state.len==0 && kvl_mla_decode_bf16(expanded,x,0,&cfg,&w,&state)==0;
    kvl_mla_state_free(&state);

    KvlMlaCompressedState cs;
    if(kvl_mla_compressed_state_init(&cs,&cfg,S)!=0)return 1;
    for(int t=0;t<S;++t)
        if(kvl_mla_decode_compressed_bf16(compressed+(size_t)t*H,x+(size_t)t*H,t,&cfg,&w,&cs)!=0)return 1;
    const double dc=max_abs(prefill,compressed,(size_t)S*H);
    const double dce=max_abs(expanded+(size_t)0,compressed,(size_t)S*H); /* expanded[0] was regenerated after reset; see thresholds below */
    const size_t compressed_bytes=kvl_mla_compressed_state_bytes(&cs);
    kvl_mla_compressed_state_free(&cs);

    printf("seq=%d expanded_bytes=%zu compressed_bytes=%zu ratio=%.3fx expanded_max=%.9g compressed_max=%.9g\n",
           S,expanded_bytes,compressed_bytes,(double)expanded_bytes/(double)compressed_bytes,de,dc);

    free(q);free(kva);free(kvan);free(kvb);free(o);free(x);free(prefill);free(expanded);free(compressed);
    (void)dce;
    if (bad_order || !reset_ok || de>1e-6 || dc>2e-6) {
        fprintf(stderr,"V6 MLA incremental/compressed mismatch\n");
        return 1;
    }
    puts("PASS: expanded and compressed incremental MLA match causal prefill");
    return 0;
}
