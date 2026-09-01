#include "kvl/ops.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int kvl_router_noaux_tc_stack(const KvlRouterConfig *cfg,
                              const float *x,
                              const float *router_weight,
                              const float *correction_bias,
                              int *top_ids, float *top_weights);

static uint32_t next_u32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static float signed_unit(uint32_t *s) {
    return ((float)(next_u32(s) & 0xffffu) / 32767.5f) - 1.0f;
}

static int run_case(KvlRouterConfig cfg, int rounds, uint32_t seed) {
    const int E = cfg.n_experts, H = cfg.hidden_size, K = cfg.top_k;
    float *x = (float *)malloc((size_t)H * sizeof(float));
    float *rw = (float *)malloc((size_t)E * H * sizeof(float));
    float *bias = (float *)malloc((size_t)E * sizeof(float));
    int *base_ids = (int *)malloc((size_t)K * sizeof(int));
    int *stack_ids = (int *)malloc((size_t)K * sizeof(int));
    float *base_w = (float *)malloc((size_t)K * sizeof(float));
    float *stack_w = (float *)malloc((size_t)K * sizeof(float));
    if (!x || !rw || !bias || !base_ids || !stack_ids || !base_w || !stack_w)
        return -1;

    uint32_t rng = seed ? seed : 1u;
    for (int r = 0; r < rounds; ++r) {
        for (int i = 0; i < H; ++i)
            x[i] = signed_unit(&rng) * 0.35f;
        for (int e = 0; e < E; ++e) {
            bias[e] = signed_unit(&rng) * 0.22f + (float)e * 1.0e-5f;
            for (int i = 0; i < H; ++i)
                rw[(size_t)e * H + i] = signed_unit(&rng) * 0.18f;
        }

        if (kvl_router_noaux_tc(&cfg, x, rw, bias, base_ids, base_w) != 0 ||
            kvl_router_noaux_tc_stack(&cfg, x, rw, bias, stack_ids, stack_w) != 0) {
            fprintf(stderr, "router execution failed round=%d\n", r);
            return -1;
        }
        if (memcmp(base_ids, stack_ids, (size_t)K * sizeof(int)) != 0 ||
            memcmp(base_w, stack_w, (size_t)K * sizeof(float)) != 0) {
            fprintf(stderr, "router mismatch round=%d groups=%d topk_group=%d\n",
                    r, cfg.n_group, cfg.topk_group);
            for (int i = 0; i < K; ++i)
                fprintf(stderr, " rank=%d base=(%d,%.9g) stack=(%d,%.9g)\n",
                        i, base_ids[i], base_w[i], stack_ids[i], stack_w[i]);
            return 1;
        }
    }

    free(x); free(rw); free(bias);
    free(base_ids); free(stack_ids); free(base_w); free(stack_w);
    return 0;
}

int main(void) {
    KvlRouterConfig kimi = {2048, 64, 6, 1, 1, 1, 2.446f};
    KvlRouterConfig grouped = {128, 64, 6, 8, 4, 1, 2.446f};
    if (run_case(kimi, 48, 0x51a7c3u) != 0) return 1;
    if (run_case(grouped, 96, 0x8f39b1u) != 0) return 1;
    puts("ROUTER_STACK_BIT_EXACT_PASS");
    return 0;
}
