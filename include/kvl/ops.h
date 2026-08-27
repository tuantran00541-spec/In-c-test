#ifndef KVL_OPS_H
#define KVL_OPS_H

#include <stdint.h>
#include "kvl/expert_cache.h"

typedef struct {
    int hidden_size;
    int n_experts;
    int top_k;
    int n_group;
    int topk_group;
    int norm_topk_prob;
    float routed_scaling_factor;
} KvlRouterConfig;

typedef struct {
    const uint16_t *gate;
    const uint16_t *up;
    const uint16_t *down;
    int intermediate_size;
} KvlMlpBF16;

float kvl_bf16_to_f32(uint16_t x);
void kvl_matvec_bf16(float *y, const float *x, const uint16_t *w,
                     int in, int out);
void kvl_silu_mul(float *y, const float *gate, const float *up, int n);
int kvl_mlp_bf16(float *y, const float *x, const KvlMlpBF16 *mlp,
                 int hidden_size, float *scratch);

/* Kimi/DeepSeek noaux_tc router. Router weights are row-major [n_experts, hidden].
 * Selection uses sigmoid(logit) + correction_bias. Mixing weights use the unbiased
 * sigmoid scores, optionally normalized, then multiplied by routed_scaling_factor. */
int kvl_router_noaux_tc(const KvlRouterConfig *cfg, const float *x,
                        const float *router_weight, const float *correction_bias,
                        int *top_ids, float *top_weights);

/* One-token routed+shared MoE forward. Routed expert bytes come from the V1 cache.
 * Shared expert weights stay resident and may be NULL. `scratch` must hold at least
 * 3*max(expert_intermediate_size, shared_intermediate_size) + hidden_size floats. */
int kvl_moe_token_bf16(KvlExpertCache *cache, int layer,
                       const KvlRouterConfig *router_cfg,
                       const float *x,
                       const float *router_weight,
                       const float *correction_bias,
                       int expert_intermediate_size,
                       const KvlMlpBF16 *shared,
                       float *out,
                       int *top_ids,
                       float *top_weights,
                       float *scratch);

#endif
