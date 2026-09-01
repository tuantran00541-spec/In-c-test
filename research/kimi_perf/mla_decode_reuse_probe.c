#include "mla_decode_reuse.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    S = 12, N = 8, H = 32, NH = 2, DN = 8, DR = 8, DV = 16, R = 12,
    QD = DN + DR, QO = NH * QD, KVO = R + DR, KHV = DN + DV, HO = NH * DV
};

static uint16_t f32_to_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    return (uint16_t)(bits >> 16);
}

static void fill_bf16(uint16_t *dst, size_t n, int seed) {
    for (size_t i = 0; i < n; ++i) {
        const int v = (int)((i * 43u + (unsigned)seed * 23u) % 263u) - 131;
        dst[i] = f32_to_bf16((float)v / 1024.0f);
    }
}

static void fill_x(float *x, int count, int seed) {
    for (int t = 0; t < count; ++t)
        for (int i = 0; i < H; ++i)
            x[(size_t)t * H + i] =
                (float)(((t + seed) * 31 + i * 13) % 211 - 105) / 512.0f;
}

int main(void) {
    uint16_t *q = (uint16_t *)malloc((size_t)QO * H * sizeof(uint16_t));
    uint16_t *kva = (uint16_t *)malloc((size_t)KVO * H * sizeof(uint16_t));
    uint16_t *kvan = (uint16_t *)malloc((size_t)R * sizeof(uint16_t));
    uint16_t *kvb = (uint16_t *)malloc((size_t)NH * KHV * R * sizeof(uint16_t));
    uint16_t *o = (uint16_t *)malloc((size_t)H * HO * sizeof(uint16_t));
    float *prefill_x = (float *)malloc((size_t)S * H * sizeof(float));
    if (!q || !kva || !kvan || !kvb || !o || !prefill_x) return 2;

    fill_bf16(q, (size_t)QO * H, 1);
    fill_bf16(kva, (size_t)KVO * H, 2);
    fill_bf16(kvan, R, 3);
    fill_bf16(kvb, (size_t)NH * KHV * R, 4);
    fill_bf16(o, (size_t)H * HO, 5);
    fill_x(prefill_x, S, 7);

    KvlMlaConfig cfg = {H, NH, DN, DR, DV, R, 1.0e-5f, 800000.0f};
    KvlMlaBF16 w = {q, kva, kvan, kvb, o};
    KvlMlaCompressedState base_state, reuse_state;
    if (kvl_mla_compressed_state_init(&base_state, &cfg, S + N) != 0 ||
        kvl_mla_compressed_state_init(&reuse_state, &cfg, S + N) != 0)
        return 2;
    if (kvl_mla_compressed_state_prefill_bf16(prefill_x, S, &cfg, &w,
                                               &base_state) != 0 ||
        kvl_mla_compressed_state_prefill_bf16(prefill_x, S, &cfg, &w,
                                               &reuse_state) != 0)
        return 2;

    int exact = 1;
    for (int step = 0; step < N; ++step) {
        float xt[H], base_out[H], reuse_out[H];
        fill_x(xt, 1, 100 + step);
        const int position = S + step;
        if (kvl_mla_decode_compressed_bf16(base_out, xt, position, &cfg, &w,
                                           &base_state) != 0 ||
            kvl_mla_decode_compressed_reuse_bf16(reuse_out, xt, position, &cfg, &w,
                                                  &reuse_state) != 0) {
            fprintf(stderr, "decode failed at step=%d\n", step);
            return 1;
        }

        const size_t latent_bytes = (size_t)(position + 1) * R * sizeof(float);
        const size_t rope_bytes = (size_t)(position + 1) * DR * sizeof(float);
        const int out_exact = memcmp(base_out, reuse_out, sizeof base_out) == 0;
        const int latent_exact = memcmp(base_state.latent, reuse_state.latent,
                                        latent_bytes) == 0;
        const int rope_exact = memcmp(base_state.rope, reuse_state.rope,
                                      rope_bytes) == 0;
        const int len_exact = base_state.len == reuse_state.len;
        printf("step=%d out=%s latent=%s rope=%s len=%s\n", step,
               out_exact ? "exact" : "DIFF",
               latent_exact ? "exact" : "DIFF",
               rope_exact ? "exact" : "DIFF",
               len_exact ? "exact" : "DIFF");
        if (!out_exact || !latent_exact || !rope_exact || !len_exact) {
            exact = 0;
            break;
        }
    }

    kvl_mla_decode_reuse_release();
    kvl_mla_compressed_state_free(&base_state);
    kvl_mla_compressed_state_free(&reuse_state);
    free(q); free(kva); free(kvan); free(kvb); free(o); free(prefill_x);

    if (!exact) return 1;
    puts("MLA_DECODE_REUSE_BIT_EXACT_PASS");
    return 0;
}
