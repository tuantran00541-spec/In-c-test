#include "mla_prefill_token_parallel.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    S = 24, H = 32, NH = 2, DN = 8, DR = 8, DV = 16, R = 12,
    QD = DN + DR, QO = NH * QD, KVO = R + DR, KHV = DN + DV, HO = NH * DV
};

static uint16_t f32_to_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    return (uint16_t)(bits >> 16);
}

static void fill_bf16(uint16_t *dst, size_t n, int seed) {
    for (size_t i = 0; i < n; ++i) {
        const int v = (int)((i * 37u + (unsigned)seed * 17u) % 257u) - 128;
        dst[i] = f32_to_bf16((float)v / 1024.0f);
    }
}

int main(void) {
    float *x = (float *)malloc((size_t)S * H * sizeof(float));
    uint16_t *q = (uint16_t *)malloc((size_t)QO * H * sizeof(uint16_t));
    uint16_t *kva = (uint16_t *)malloc((size_t)KVO * H * sizeof(uint16_t));
    uint16_t *kvan = (uint16_t *)malloc((size_t)R * sizeof(uint16_t));
    uint16_t *kvb = (uint16_t *)malloc((size_t)NH * KHV * R * sizeof(uint16_t));
    uint16_t *o = (uint16_t *)malloc((size_t)H * HO * sizeof(uint16_t));
    float *base_out = (float *)malloc((size_t)S * H * sizeof(float));
    float *candidate_out = (float *)malloc((size_t)S * H * sizeof(float));
    if (!x || !q || !kva || !kvan || !kvb || !o || !base_out || !candidate_out)
        return 2;

    for (int t = 0; t < S; ++t)
        for (int i = 0; i < H; ++i)
            x[(size_t)t * H + i] = (float)(((t + 3) * 29 + i * 11) % 193 - 96) / 512.0f;
    fill_bf16(q, (size_t)QO * H, 1);
    fill_bf16(kva, (size_t)KVO * H, 2);
    fill_bf16(kvan, R, 3);
    fill_bf16(kvb, (size_t)NH * KHV * R, 4);
    fill_bf16(o, (size_t)H * HO, 5);

    KvlMlaConfig cfg = {H, NH, DN, DR, DV, R, 1.0e-5f, 800000.0f};
    KvlMlaBF16 w = {q, kva, kvan, kvb, o};
    KvlMlaCompressedState base_state, candidate_state;
    if (kvl_mla_compressed_state_init(&base_state, &cfg, S + 4) != 0 ||
        kvl_mla_compressed_state_init(&candidate_state, &cfg, S + 4) != 0)
        return 2;

    if (kvl_mla_prefill_bf16(base_out, x, S, &cfg, &w) != 0 ||
        kvl_mla_compressed_state_prefill_bf16(x, S, &cfg, &w, &base_state) != 0 ||
        kvl_mla_prefill_compressed_token_parallel_bf16(candidate_out, x, S, &cfg, &w,
                                                        &candidate_state) != 0) {
        fprintf(stderr, "token-parallel prefill probe failed to execute\n");
        return 1;
    }

    const int out_exact = memcmp(base_out, candidate_out,
                                 (size_t)S * H * sizeof(float)) == 0;
    const int latent_exact = memcmp(base_state.latent, candidate_state.latent,
                                    (size_t)S * R * sizeof(float)) == 0;
    const int rope_exact = memcmp(base_state.rope, candidate_state.rope,
                                  (size_t)S * DR * sizeof(float)) == 0;
    const int len_exact = base_state.len == candidate_state.len && base_state.len == S;
    printf("out_exact=%s latent_exact=%s rope_exact=%s len_exact=%s\n",
           out_exact ? "yes" : "no",
           latent_exact ? "yes" : "no",
           rope_exact ? "yes" : "no",
           len_exact ? "yes" : "no");

    kvl_mla_compressed_state_free(&base_state);
    kvl_mla_compressed_state_free(&candidate_state);
    free(x); free(q); free(kva); free(kvan); free(kvb); free(o);
    free(base_out); free(candidate_out);

    if (!out_exact || !latent_exact || !rope_exact || !len_exact) return 1;
    puts("MLA_TOKEN_PARALLEL_PREFILL_BIT_EXACT_PASS");
    return 0;
}
