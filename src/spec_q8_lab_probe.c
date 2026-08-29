#include "kvl/mla_compressed_state.h"
#include "kvl/mla_compressed_q8_state.h"

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
        const float x = bias + scale *
            (float)(sin((double)i * 0.137) + 0.41 * cos((double)i * 0.053));
        p[i] = f32_to_bf16(x);
    }
}

static double max_abs(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

static double rel_rms(const float *ref, const float *got, size_t n) {
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)got[i] - (double)ref[i];
        se += d * d;
        sr += (double)ref[i] * (double)ref[i];
    }
    return sqrt(se / (sr + 1.0e-30));
}

int main(void) {
    enum {
        S = 13, PREFIX = 5, BLOCK = 4, ACCEPT = 2,
        H = 64, NH = 4, DN = 8, DR = 8, DV = 8, R = 32
    };
    enum { QD = DN + DR, QO = NH * QD, KVO = R + DR, KVB = NH * (DN + DV) };

    uint16_t *q = (uint16_t *)malloc((size_t)QO * H * sizeof(uint16_t));
    uint16_t *kva = (uint16_t *)malloc((size_t)KVO * H * sizeof(uint16_t));
    uint16_t *kvan = (uint16_t *)malloc((size_t)R * sizeof(uint16_t));
    uint16_t *kvb = (uint16_t *)malloc((size_t)KVB * R * sizeof(uint16_t));
    uint16_t *o = (uint16_t *)malloc((size_t)H * (NH * DV) * sizeof(uint16_t));
    float *x = (float *)malloc((size_t)S * H * sizeof(float));
    float *seq = (float *)calloc((size_t)S * H, sizeof(float));
    float *block = (float *)calloc((size_t)BLOCK * H, sizeof(float));
    float *q8 = (float *)calloc((size_t)(S - PREFIX) * H, sizeof(float));
    float *replacement = (float *)malloc((size_t)H * sizeof(float));
    float *rollback = (float *)calloc((size_t)H, sizeof(float));
    float *rollback_ref = (float *)calloc((size_t)H, sizeof(float));
    float *q8_rollback = (float *)calloc((size_t)H, sizeof(float));
    float *q8_rollback_ref = (float *)calloc((size_t)H, sizeof(float));
    if (!q || !kva || !kvan || !kvb || !o || !x || !seq || !block || !q8 ||
        !replacement || !rollback || !rollback_ref || !q8_rollback || !q8_rollback_ref)
        return 2;

    fill_bf16(q, (size_t)QO * H, 0.031f, 0.0f);
    fill_bf16(kva, (size_t)KVO * H, 0.029f, 0.0f);
    fill_bf16(kvan, R, 0.019f, 0.97f);
    fill_bf16(kvb, (size_t)KVB * R, 0.036f, 0.0f);
    fill_bf16(o, (size_t)H * (NH * DV), 0.034f, 0.0f);
    for (int t = 0; t < S; ++t)
        for (int i = 0; i < H; ++i)
            x[(size_t)t * H + i] =
                0.21f * (float)sin((double)(t + 1) * (i + 3) * 0.071) +
                0.05f * (float)cos((double)(i + 1) * 0.113) +
                0.003f * (float)t;
    for (int i = 0; i < H; ++i)
        replacement[i] = 0.18f * (float)cos((double)(i + 5) * 0.097) -
                         0.07f * (float)sin((double)(i + 2) * 0.151);

    KvlMlaConfig cfg = {H, NH, DN, DR, DV, R, 1.0e-6f, 10000.0f};
    KvlMlaBF16 w = {q, kva, kvan, kvb, o};

    KvlMlaCompressedState seq_state;
    if (kvl_mla_compressed_state_init(&seq_state, &cfg, S + 1) != 0) return 1;
    for (int t = 0; t < S; ++t)
        if (kvl_mla_decode_compressed_bf16(seq + (size_t)t * H,
                                           x + (size_t)t * H, t,
                                           &cfg, &w, &seq_state) != 0)
            return 1;

    KvlMlaCompressedState block_state;
    if (kvl_mla_compressed_state_init(&block_state, &cfg, S + 1) != 0) return 1;
    if (kvl_mla_compressed_state_prefill_bf16(x, PREFIX, &cfg, &w, &block_state) != 0)
        return 1;
    if (kvl_mla_decode_compressed_block_bf16(block,
                                             x + (size_t)PREFIX * H,
                                             BLOCK, PREFIX, &cfg, &w,
                                             &block_state) != 0)
        return 1;
    const double block_max = max_abs(seq + (size_t)PREFIX * H,
                                     block, (size_t)BLOCK * H);

    if (kvl_mla_compressed_state_truncate(&block_state, PREFIX + ACCEPT) != 0)
        return 1;
    if (kvl_mla_decode_compressed_bf16(rollback, replacement, PREFIX + ACCEPT,
                                       &cfg, &w, &block_state) != 0)
        return 1;

    KvlMlaCompressedState rollback_state;
    if (kvl_mla_compressed_state_init(&rollback_state, &cfg, S + 1) != 0) return 1;
    if (kvl_mla_compressed_state_prefill_bf16(x, PREFIX + ACCEPT,
                                               &cfg, &w, &rollback_state) != 0 ||
        kvl_mla_decode_compressed_bf16(rollback_ref, replacement, PREFIX + ACCEPT,
                                       &cfg, &w, &rollback_state) != 0)
        return 1;
    const double rollback_max = max_abs(rollback_ref, rollback, H);

    KvlMlaCompressedQ8State q8_state;
    if (kvl_mla_compressed_q8_state_init(&q8_state, &cfg, S + 1) != 0) return 1;
    if (kvl_mla_compressed_q8_state_prefill_bf16(x, PREFIX, &cfg, &w, &q8_state) != 0)
        return 1;
    for (int t = PREFIX; t < S; ++t)
        if (kvl_mla_decode_compressed_q8_bf16(q8 + (size_t)(t - PREFIX) * H,
                                              x + (size_t)t * H, t,
                                              &cfg, &w, &q8_state) != 0)
            return 1;
    const size_t q8_n = (size_t)(S - PREFIX) * H;
    const double q8_max = max_abs(seq + (size_t)PREFIX * H, q8, q8_n);
    const double q8_rel = rel_rms(seq + (size_t)PREFIX * H, q8, q8_n);

    if (kvl_mla_compressed_q8_state_truncate(&q8_state, PREFIX + ACCEPT) != 0 ||
        kvl_mla_decode_compressed_q8_bf16(q8_rollback, replacement, PREFIX + ACCEPT,
                                          &cfg, &w, &q8_state) != 0)
        return 1;
    KvlMlaCompressedQ8State q8_fresh;
    if (kvl_mla_compressed_q8_state_init(&q8_fresh, &cfg, S + 1) != 0) return 1;
    if (kvl_mla_compressed_q8_state_prefill_bf16(x, PREFIX + ACCEPT,
                                                  &cfg, &w, &q8_fresh) != 0 ||
        kvl_mla_decode_compressed_q8_bf16(q8_rollback_ref, replacement,
                                          PREFIX + ACCEPT, &cfg, &w, &q8_fresh) != 0)
        return 1;
    const double q8_rollback_max = max_abs(q8_rollback_ref, q8_rollback, H);

    const size_t fp32_bytes = kvl_mla_compressed_state_bytes(&seq_state);
    const size_t q8_bytes = kvl_mla_compressed_q8_state_bytes(&q8_state);
    const double kimi_payload_fp32 = 512.0 * 4.0 + 64.0 * 4.0;
    const double kimi_payload_q8 = 512.0 + 4.0 + 64.0 * 2.0;

    printf("EXACT_BLOCK prefix=%d block=%d accepted=%d block_max=%.9g rollback_max=%.9g\n",
           PREFIX, BLOCK, ACCEPT, block_max, rollback_max);
    printf("Q8_MLA synthetic_max=%.9g synthetic_rel_rms=%.9g fp32_state_bytes=%zu "
           "q8_state_bytes=%zu synthetic_ratio=%.3fx\n",
           q8_max, q8_rel, fp32_bytes, q8_bytes,
           (double)fp32_bytes / (double)q8_bytes);
    printf("KIMI_Q8_LAYOUT latent=int8_per_token rope=bf16 scale=f32 payload=%.0f->%.0f "
           "raw_ratio=%.3fx\n",
           kimi_payload_fp32, kimi_payload_q8,
           kimi_payload_fp32 / kimi_payload_q8);
    printf("Q8_ROLLBACK max=%.9g\n", q8_rollback_max);

    kvl_mla_compressed_state_free(&seq_state);
    kvl_mla_compressed_state_free(&block_state);
    kvl_mla_compressed_state_free(&rollback_state);
    kvl_mla_compressed_q8_state_free(&q8_state);
    kvl_mla_compressed_q8_state_free(&q8_fresh);
    free(q); free(kva); free(kvan); free(kvb); free(o); free(x); free(seq); free(block);
    free(q8); free(replacement); free(rollback); free(rollback_ref);
    free(q8_rollback); free(q8_rollback_ref);

    if (block_max != 0.0 || rollback_max != 0.0 || q8_rollback_max != 0.0) {
        fprintf(stderr, "exact block/truncate semantics mismatch\n");
        return 1;
    }
    if (!isfinite(q8_max) || !isfinite(q8_rel) || q8_max > 0.02 || q8_rel > 0.02) {
        fprintf(stderr, "hybrid INT8 MLA synthetic error above conservative lab gate\n");
        return 1;
    }
    puts("PASS: exact block rollback and hybrid INT8 MLA synthetic gate");
    return 0;
}
