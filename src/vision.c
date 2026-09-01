#include "kvl/vision.h"
#include "kvl/ops.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef KVL_USE_AVX2
#include <immintrin.h>
#endif

enum {
    VD = 1152,
    VHEADS = 16,
    VHD = 72,
    VI = 4304,
    PATCH = 14,
    PATCH_IN = 3 * PATCH * PATCH,
    PROJECT = 4 * VD,
    TEXT_D = 2048,
    VLAYERS = 27
};

static const float LN_EPS = 1.0e-5f;
static const float ROPE_THETA = 10000.0f;

static int loadv(KvlTrunkStore *s, uint32_t layer, uint32_t kind, KvlTrunkTensor *t) {
    return kvl_trunk_load(s, layer, kind, t);
}

static void layernorm_bf16(float *y, const float *x,
                           const uint16_t *weight, const uint16_t *bias,
                           int n, float eps) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += x[i];
    const double mean = sum / (double)n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = (double)x[i] - mean;
        var += d * d;
    }
    const float inv = 1.0f / sqrtf((float)(var / (double)n) + eps);
    for (int i = 0; i < n; ++i) {
        const float w = kvl_bf16_to_f32(weight[i]);
        const float b = bias ? kvl_bf16_to_f32(bias[i]) : 0.0f;
        y[i] = ((x[i] - (float)mean) * inv) * w + b;
    }
}

/* Vision-only batched BF16 projection prototype. Each individual dot product is
 * deliberately identical to kvl_matvec_bf16; only loop ordering changes so a
 * weight row stays hot while it is applied to every patch/token in the batch. */
static float vision_dot_bf16_ref(const uint16_t *row, const float *x, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i)
        acc += (double)kvl_bf16_to_f32(row[i]) * (double)x[i];
    return (float)acc;
}

#ifdef KVL_USE_AVX2
static float vision_dot_bf16_avx2(const uint16_t *row, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i w16 = _mm_loadu_si128((const __m128i *)(row + i));
        __m256i w32 = _mm256_cvtepu16_epi32(w16);
        w32 = _mm256_slli_epi32(w32, 16);
        const __m256 wf = _mm256_castsi256_ps(w32);
        const __m256 xv = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(wf, xv));
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float out = _mm_cvtss_f32(sum);
    for (; i < n; ++i) out += kvl_bf16_to_f32(row[i]) * x[i];
    return out;
}
#endif

static void vision_matmul_bf16(float *y, const float *x, const uint16_t *w,
                               int rows, int in, int out) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(out > 64 && rows > 1)
#endif
    for (int o = 0; o < out; ++o) {
        const uint16_t *wr = w + (size_t)o * (size_t)in;
        for (int r = 0; r < rows; ++r) {
            const float *xr = x + (size_t)r * (size_t)in;
#ifdef KVL_USE_AVX2
            y[(size_t)r * (size_t)out + o] = vision_dot_bf16_avx2(wr, xr, in);
#else
            y[(size_t)r * (size_t)out + o] = vision_dot_bf16_ref(wr, xr, in);
#endif
        }
    }
}

static void linear_bias_batch(float *y, const float *x,
                              const uint16_t *weight, const uint16_t *bias,
                              int rows, int in, int out) {
    vision_matmul_bf16(y, x, weight, rows, in, out);
    if (!bias) return;
    for (int r = 0; r < rows; ++r) {
        float *yr = y + (size_t)r * (size_t)out;
        for (int o = 0; o < out; ++o) yr[o] += kvl_bf16_to_f32(bias[o]);
    }
}

static float gelu_tanh(float x) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
}

static float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

/* PyTorch bicubic interpolation with align_corners=False uses the half-pixel coordinate
 * transform and cubic coefficient -0.75. Border samples are clamped. */
static float cubic_weight(float x) {
    const float a = -0.75f;
    x = fabsf(x);
    if (x <= 1.0f)
        return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f)
        return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
    return 0.0f;
}

static int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float interp_pos(const uint16_t *src, int in_h, int in_w,
                        int out_h, int out_w, int oh, int ow, int d) {
    const float sy = ((float)oh + 0.5f) * (float)in_h / (float)out_h - 0.5f;
    const float sx = ((float)ow + 0.5f) * (float)in_w / (float)out_w - 0.5f;
    const int y0 = (int)floorf(sy);
    const int x0 = (int)floorf(sx);
    double acc = 0.0;
    for (int ky = -1; ky <= 2; ++ky) {
        const int iy_raw = y0 + ky;
        const int iy = clampi(iy_raw, 0, in_h - 1);
        const float wy = cubic_weight(sy - (float)iy_raw);
        for (int kx = -1; kx <= 2; ++kx) {
            const int ix_raw = x0 + kx;
            const int ix = clampi(ix_raw, 0, in_w - 1);
            const float wx = cubic_weight(sx - (float)ix_raw);
            const float v = kvl_bf16_to_f32(src[((size_t)iy * in_w + ix) * VD + d]);
            acc += (double)wy * (double)wx * (double)v;
        }
    }
    return (float)acc;
}

static void rope2d_pair(float *q, float *k, int row, int col) {
    /* Source layout has 36 complex pairs. Pair 2*i uses x/column and pair 2*i+1
     * uses y/row with exponent (4*i)/72. */
    for (int p = 0; p < VHD / 2; ++p) {
        const int i = p / 2;
        const float pos = (p & 1) ? (float)row : (float)col;
        const double inv = pow((double)ROPE_THETA, -(double)(4 * i) / (double)VHD);
        const float angle = pos * (float)inv;
        const float c = cosf(angle), s = sinf(angle);
        const int a = 2 * p, b = a + 1;
        const float qa = q[a], qb = q[b], ka = k[a], kb = k[b];
        q[a] = qa * c - qb * s; q[b] = qb * c + qa * s;
        k[a] = ka * c - kb * s; k[b] = kb * c + ka * s;
    }
}

static int run_block(KvlTrunkStore *vs, int layer, float *x, int seq, int gh, int gw,
                     float *norm, float *qkv, float *attn, float *scores,
                     float *inter, float *tmpd) {
    KvlTrunkTensor n0w={0},n0b={0},qw={0},qb={0},ow={0},ob={0};
    KvlTrunkTensor n1w={0},n1b={0},f0w={0},f0b={0},f1w={0},f1b={0};
    int rc = -1;
    if (loadv(vs,layer,KVL_VISION_NORM0_WEIGHT,&n0w) ||
        loadv(vs,layer,KVL_VISION_NORM0_BIAS,&n0b) ||
        loadv(vs,layer,KVL_VISION_WQKV_WEIGHT,&qw) ||
        loadv(vs,layer,KVL_VISION_WQKV_BIAS,&qb) ||
        loadv(vs,layer,KVL_VISION_WO_WEIGHT,&ow) ||
        loadv(vs,layer,KVL_VISION_WO_BIAS,&ob) ||
        loadv(vs,layer,KVL_VISION_NORM1_WEIGHT,&n1w) ||
        loadv(vs,layer,KVL_VISION_NORM1_BIAS,&n1b) ||
        loadv(vs,layer,KVL_VISION_MLP0_WEIGHT,&f0w) ||
        loadv(vs,layer,KVL_VISION_MLP0_BIAS,&f0b) ||
        loadv(vs,layer,KVL_VISION_MLP1_WEIGHT,&f1w) ||
        loadv(vs,layer,KVL_VISION_MLP1_BIAS,&f1b)) goto done;

    for (int t=0;t<seq;++t)
        layernorm_bf16(norm+(size_t)t*VD,x+(size_t)t*VD,
                       (const uint16_t*)n0w.data,(const uint16_t*)n0b.data,VD,LN_EPS);
    linear_bias_batch(qkv,norm,(const uint16_t*)qw.data,(const uint16_t*)qb.data,
                      seq,VD,3*VD);

    for (int t=0;t<seq;++t) {
        const int row=t/gw, col=t%gw;
        float *base=qkv+(size_t)t*3*VD;
        for(int h=0;h<VHEADS;++h)
            rope2d_pair(base+(size_t)h*VHD,
                        base+VD+(size_t)h*VHD,row,col);
    }

    const float scale=1.0f/sqrtf((float)VHD);
    for(int t=0;t<seq;++t) {
        const float *tb=qkv+(size_t)t*3*VD;
        for(int h=0;h<VHEADS;++h) {
            const float *q=tb+(size_t)h*VHD;
            float mx=-INFINITY;
            for(int j=0;j<seq;++j) {
                const float *k=qkv+(size_t)j*3*VD+VD+(size_t)h*VHD;
                double dot=0.0;
                for(int d=0;d<VHD;++d) dot+=(double)q[d]*(double)k[d];
                scores[j]=(float)dot*scale;
                if(scores[j]>mx)mx=scores[j];
            }
            double den=0.0;
            for(int j=0;j<seq;++j){scores[j]=expf(scores[j]-mx);den+=scores[j];}
            float *ho=attn+(size_t)t*VD+(size_t)h*VHD;
            for(int d=0;d<VHD;++d){
                double a=0.0;
                for(int j=0;j<seq;++j){
                    const float *v=qkv+(size_t)j*3*VD+2*VD+(size_t)h*VHD;
                    a+=((double)scores[j]/den)*(double)v[d];
                }
                ho[d]=(float)a;
            }
        }
    }
    linear_bias_batch(tmpd,attn,(const uint16_t*)ow.data,(const uint16_t*)ob.data,
                      seq,VD,VD);
    for (int t=0;t<seq;++t)
        for(int d=0;d<VD;++d)x[(size_t)t*VD+d]+=tmpd[(size_t)t*VD+d];

    for(int t=0;t<seq;++t)
        layernorm_bf16(norm+(size_t)t*VD,x+(size_t)t*VD,
                       (const uint16_t*)n1w.data,(const uint16_t*)n1b.data,VD,LN_EPS);
    linear_bias_batch(inter,norm,(const uint16_t*)f0w.data,(const uint16_t*)f0b.data,
                      seq,VD,VI);
    for(size_t i=0;i<(size_t)seq*VI;++i)inter[i]=gelu_tanh(inter[i]);
    linear_bias_batch(tmpd,inter,(const uint16_t*)f1w.data,(const uint16_t*)f1b.data,
                      seq,VI,VD);
    for(int t=0;t<seq;++t)
        for(int d=0;d<VD;++d)x[(size_t)t*VD+d]+=tmpd[(size_t)t*VD+d];
    rc=0;

done:
    kvl_trunk_tensor_free(&n0w);kvl_trunk_tensor_free(&n0b);
    kvl_trunk_tensor_free(&qw);kvl_trunk_tensor_free(&qb);
    kvl_trunk_tensor_free(&ow);kvl_trunk_tensor_free(&ob);
    kvl_trunk_tensor_free(&n1w);kvl_trunk_tensor_free(&n1b);
    kvl_trunk_tensor_free(&f0w);kvl_trunk_tensor_free(&f0b);
    kvl_trunk_tensor_free(&f1w);kvl_trunk_tensor_free(&f1b);
    return rc;
}

int kvl_vision_forward(KvlTrunkStore *vs, const float *patches,
                       int gh, int gw, float *out, int *out_tokens) {
    if(!vs||!patches||!out||!out_tokens||gh<=0||gw<=0||gh>512||gw>512||
       (gh&1)||(gw&1)) return -1;
    const int seq=gh*gw;
    const int merged=(gh/2)*(gw/2);

    KvlTrunkTensor pw={0},pb={0},pos={0};
    KvlTrunkTensor fnw={0},fnb={0};
    KvlTrunkTensor pnw={0},pnb={0},l1w={0},l1b={0},l2w={0},l2b={0};
    float *x=NULL,*norm=NULL,*qkv=NULL,*attn=NULL,*scores=NULL,*inter=NULL,*tmpd=NULL;
    float *concat=NULL,*projtmp=NULL;
    int rc=-1;

    if(loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_PATCH_WEIGHT,&pw)||
       loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_PATCH_BIAS,&pb)||
       loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_POS_EMB,&pos)) goto done;
    if(pos.record->ndim!=3||pos.record->dims[2]!=VD)goto done;
    const int ph=(int)pos.record->dims[0], pwid=(int)pos.record->dims[1];

    x=(float*)malloc((size_t)seq*VD*sizeof(float));
    norm=(float*)malloc((size_t)seq*VD*sizeof(float));
    qkv=(float*)malloc((size_t)seq*3*VD*sizeof(float));
    attn=(float*)malloc((size_t)seq*VD*sizeof(float));
    scores=(float*)malloc((size_t)seq*sizeof(float));
    inter=(float*)malloc((size_t)seq*VI*sizeof(float));
    tmpd=(float*)malloc((size_t)seq*VD*sizeof(float));
    concat=(float*)malloc((size_t)merged*PROJECT*sizeof(float));
    projtmp=(float*)malloc((size_t)merged*PROJECT*sizeof(float));
    if(!x||!norm||!qkv||!attn||!scores||!inter||!tmpd||!concat||!projtmp)goto done;

    linear_bias_batch(x,patches,(const uint16_t*)pw.data,(const uint16_t*)pb.data,
                      seq,PATCH_IN,VD);
    for(int t=0;t<seq;++t){
        const int oh=t/gw,ow=t%gw;
        if(gh==ph&&gw==pwid){
            const uint16_t *pp=(const uint16_t*)pos.data+(size_t)t*VD;
            for(int d=0;d<VD;++d)x[(size_t)t*VD+d]+=kvl_bf16_to_f32(pp[d]);
        }else{
            for(int d=0;d<VD;++d)x[(size_t)t*VD+d]+=
                interp_pos((const uint16_t*)pos.data,ph,pwid,gh,gw,oh,ow,d);
        }
    }
    kvl_trunk_tensor_free(&pw);kvl_trunk_tensor_free(&pb);kvl_trunk_tensor_free(&pos);

    for(int layer=0;layer<VLAYERS;++layer)
        if(run_block(vs,layer,x,seq,gh,gw,norm,qkv,attn,scores,inter,tmpd)!=0)goto done;

    if(loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_FINAL_NORM_WEIGHT,&fnw)||
       loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_FINAL_NORM_BIAS,&fnb))goto done;
    for(int t=0;t<seq;++t)
        layernorm_bf16(norm+(size_t)t*VD,x+(size_t)t*VD,
                       (const uint16_t*)fnw.data,(const uint16_t*)fnb.data,VD,LN_EPS);
    kvl_trunk_tensor_free(&fnw);kvl_trunk_tensor_free(&fnb);

    if(loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_PROJECTOR_NORM_WEIGHT,&pnw)||
       loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_PROJECTOR_NORM_BIAS,&pnb)||
       loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_PROJECTOR_L1_WEIGHT,&l1w)||
       loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_PROJECTOR_L1_BIAS,&l1b)||
       loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_PROJECTOR_L2_WEIGHT,&l2w)||
       loadv(vs,KVL_TRUNK_GLOBAL_LAYER,KVL_VISION_PROJECTOR_L2_BIAS,&l2b))goto done;

    for(int mh=0;mh<gh/2;++mh)for(int mw=0;mw<gw/2;++mw){
        const int m=mh*(gw/2)+mw;
        const int ids[4]={
            (2*mh)*gw+2*mw,
            (2*mh)*gw+2*mw+1,
            (2*mh+1)*gw+2*mw,
            (2*mh+1)*gw+2*mw+1
        };
        float *row=concat+(size_t)m*PROJECT;
        for(int k=0;k<4;++k)
            layernorm_bf16(row+(size_t)k*VD,norm+(size_t)ids[k]*VD,
                           (const uint16_t*)pnw.data,(const uint16_t*)pnb.data,VD,LN_EPS);
    }
    linear_bias_batch(projtmp,concat,(const uint16_t*)l1w.data,(const uint16_t*)l1b.data,
                      merged,PROJECT,PROJECT);
    for(size_t i=0;i<(size_t)merged*PROJECT;++i)projtmp[i]=gelu_exact(projtmp[i]);
    linear_bias_batch(out,projtmp,(const uint16_t*)l2w.data,(const uint16_t*)l2b.data,
                      merged,PROJECT,TEXT_D);
    *out_tokens=merged;
    rc=0;

done:
    kvl_trunk_tensor_free(&pw);kvl_trunk_tensor_free(&pb);kvl_trunk_tensor_free(&pos);
    kvl_trunk_tensor_free(&fnw);kvl_trunk_tensor_free(&fnb);
    kvl_trunk_tensor_free(&pnw);kvl_trunk_tensor_free(&pnb);
    kvl_trunk_tensor_free(&l1w);kvl_trunk_tensor_free(&l1b);
    kvl_trunk_tensor_free(&l2w);kvl_trunk_tensor_free(&l2b);
    free(x);free(norm);free(qkv);free(attn);free(scores);free(inter);free(tmpd);
    free(concat);free(projtmp);
    return rc;
}
