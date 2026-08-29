#include "kvl/ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Synthetic latency probe for the exact long-context MLA prefill core.
 *
 * Keep official per-head dimensions and MLA latent rank, but use two heads and
 * H=256 so CI can sweep into the quadratic regime without downloading weights.
 * These timings are only for before/after kernel comparisons; they are not an
 * estimate of full 27-layer Kimi prompt latency. */
enum { H = 256, NH = 2, DN = 128, DR = 64, DV = 128, R = 512 };
enum { QD = DN + DR, QO = NH * QD, KVO = R + DR, KVB = NH * (DN + DV), HO = NH * DV };

static uint32_t rng_state = UINT32_C(0x4d595df4);

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

static double wall_seconds(void) {
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int run_case(int seq_len, int report) {
    KvlMlaConfig cfg = {H, NH, DN, DR, DV, R, 1.0e-5f, 800000.0f};
    float *x = (float *)malloc((size_t)seq_len * H * sizeof(float));
    float *out = (float *)malloc((size_t)seq_len * H * sizeof(float));
    uint16_t *q = (uint16_t *)malloc((size_t)QO * H * sizeof(uint16_t));
    uint16_t *kva = (uint16_t *)malloc((size_t)KVO * H * sizeof(uint16_t));
    uint16_t *kvan = (uint16_t *)malloc((size_t)R * sizeof(uint16_t));
    uint16_t *kvb = (uint16_t *)malloc((size_t)KVB * R * sizeof(uint16_t));
    uint16_t *o = (uint16_t *)malloc((size_t)H * HO * sizeof(uint16_t));
    if (!x || !out || !q || !kva || !kvan || !kvb || !o) {
        free(x); free(out); free(q); free(kva); free(kvan); free(kvb); free(o);
        return -1;
    }

    for (int i = 0; i < seq_len * H; ++i) x[i] = rng_f32() * 0.125f;
    fill_bf16(q, (size_t)QO * H, 0.05f);
    fill_bf16(kva, (size_t)KVO * H, 0.05f);
    for (int i = 0; i < R; ++i) kvan[i] = f32_to_bf16(0.875f + 0.125f * rng_f32());
    fill_bf16(kvb, (size_t)KVB * R, 0.05f);
    fill_bf16(o, (size_t)H * HO, 0.05f);

    KvlMlaBF16 w = {q, kva, kvan, kvb, o};
    const double begin = wall_seconds();
    const int rc = kvl_mla_prefill_bf16(out, x, seq_len, &cfg, &w);
    const double end = wall_seconds();
    if (rc != 0) {
        free(x); free(out); free(q); free(kva); free(kvan); free(kvb); free(o);
        return rc;
    }

    double checksum = 0.0;
    for (int t = 0; t < seq_len; ++t)
        checksum += (double)out[(size_t)t * H + (t % H)];
    if (report) {
        const double sec = end - begin;
        printf("mla_prefill_bench seq=%d seconds=%.6f ms_per_token=%.6f checksum=%.9g\n",
               seq_len, sec, 1000.0 * sec / (double)seq_len, checksum);
        fflush(stdout);
    }

    free(x); free(out); free(q); free(kva); free(kvan); free(kvb); free(o);
    return 0;
}

int main(int argc, char **argv) {
    printf("mla_prefill_bench dims H=%d NH=%d DN=%d DR=%d DV=%d R=%d\n",
           H, NH, DN, DR, DV, R);

    if (run_case(8, 0) != 0) return 1; /* initialize runtime/thread machinery. */

    if (argc == 1) {
        const int cases[] = {256, 512, 1024, 2048};
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
            if (run_case(cases[i], 1) != 0) return 1;
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        const int seq = atoi(argv[i]);
        if (seq <= 0 || run_case(seq, 1) != 0) return 1;
    }
    return 0;
}
