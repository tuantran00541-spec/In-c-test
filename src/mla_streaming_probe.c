#include "kvl/ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* src/ops.c is source-renamed on the long-context lab branch so the old path
 * remains linked as a numerical oracle. */
int kvl_mla_prefill_materialized_bf16(float *out, const float *x, int seq_len,
                                      const KvlMlaConfig *cfg,
                                      const KvlMlaBF16 *w);

static uint32_t rng_state = UINT32_C(0x12345678);

static uint32_t rng_u32(void) {
    rng_state = rng_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return rng_state;
}

static float rng_f32(void) {
    const int32_t v = (int32_t)(rng_u32() >> 8) - INT32_C(0x00800000);
    return (float)v / 8388608.0f;
}

static uint16_t f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    const uint32_t exp = u & UINT32_C(0x7f800000);
    if (exp != UINT32_C(0x7f800000))
        u += UINT32_C(0x00007fff) + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static void fill_bf16(uint16_t *p, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i) p[i] = f32_to_bf16(rng_f32() * scale);
}

static int run_case(int seq_len, int dv) {
    enum { H = 16, NH = 2, DN = 4, DR = 4, R = 6 };
    const int QD = DN + DR;
    const int QO = NH * QD;
    const int KVO = R + DR;
    const int KVB = NH * (DN + dv);
    const int HO = NH * dv;

    KvlMlaConfig cfg = {H, NH, DN, DR, dv, R, 1.0e-5f, 800000.0f};
    float *x = (float *)malloc((size_t)seq_len * H * sizeof(float));
    float *ref = (float *)malloc((size_t)seq_len * H * sizeof(float));
    float *got = (float *)malloc((size_t)seq_len * H * sizeof(float));
    uint16_t *q = (uint16_t *)malloc((size_t)QO * H * sizeof(uint16_t));
    uint16_t *kva = (uint16_t *)malloc((size_t)KVO * H * sizeof(uint16_t));
    uint16_t *kvan = (uint16_t *)malloc((size_t)R * sizeof(uint16_t));
    uint16_t *kvb = (uint16_t *)malloc((size_t)KVB * R * sizeof(uint16_t));
    uint16_t *o = (uint16_t *)malloc((size_t)H * HO * sizeof(uint16_t));
    if (!x || !ref || !got || !q || !kva || !kvan || !kvb || !o) {
        free(x); free(ref); free(got); free(q); free(kva); free(kvan); free(kvb); free(o);
        return -1;
    }

    for (int i = 0; i < seq_len * H; ++i) x[i] = rng_f32() * 0.25f;
    fill_bf16(q, (size_t)QO * H, 0.20f);
    fill_bf16(kva, (size_t)KVO * H, 0.20f);
    for (int i = 0; i < R; ++i) kvan[i] = f32_to_bf16(0.75f + 0.25f * rng_f32());
    fill_bf16(kvb, (size_t)KVB * R, 0.20f);
    fill_bf16(o, (size_t)H * HO, 0.20f);

    KvlMlaBF16 w = {q, kva, kvan, kvb, o};
    if (kvl_mla_prefill_materialized_bf16(ref, x, seq_len, &cfg, &w) != 0 ||
        kvl_mla_prefill_bf16(got, x, seq_len, &cfg, &w) != 0) {
        free(x); free(ref); free(got); free(q); free(kva); free(kvan); free(kvb); free(o);
        return -1;
    }

    double max_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    for (int i = 0; i < seq_len * H; ++i) {
        const double d = (double)got[i] - (double)ref[i];
        const double a = fabs(d);
        if (a > max_abs) max_abs = a;
        sum_sq += d * d;
        ref_sq += (double)ref[i] * (double)ref[i];
    }
    const double rel_rms = sqrt(sum_sq / (ref_sq + 1.0e-30));
    printf("mla_streaming_probe seq=%d dv=%d ho=%d max_abs=%.9g rel_rms=%.9g\n",
           seq_len, dv, HO, max_abs, rel_rms);

    free(x); free(ref); free(got); free(q); free(kva); free(kvan); free(kvb); free(o);
    return (max_abs <= 2.0e-5 && rel_rms <= 2.0e-5) ? 0 : 1;
}

int main(void) {
    const int cases[] = {1, 7, 64};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const int rc = run_case(cases[i], 8); /* HO == H: official-style fast path. */
        if (rc != 0) {
            fprintf(stderr, "mla_streaming_probe FAILED seq=%d dv=8 rc=%d\n", cases[i], rc);
            return 1;
        }
    }

    /* Generic shape used by stack oracles: HO=NH*DV=8 while H=16. The old
     * materialized kernel supports this because o_proj maps HO -> H. */
    const int generic_rc = run_case(7, 4);
    if (generic_rc != 0) {
        fprintf(stderr, "mla_streaming_probe FAILED seq=7 dv=4 rc=%d\n", generic_rc);
        return 1;
    }

    puts("mla_streaming_probe PASS");
    return 0;
}
