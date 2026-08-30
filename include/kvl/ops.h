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
    int hidden_size;
    int num_heads;
    int qk_nope_dim;
    int qk_rope_dim;
    int v_head_dim;
    int kv_lora_rank;
    float rms_eps;
    float rope_theta;
} KvlMlaConfig;

typedef struct {
    const uint16_t *q_proj;
    const uint16_t *kv_a_proj;
    const uint16_t *kv_a_norm;
    const uint16_t *kv_b_proj;
    const uint16_t *o_proj;
} KvlMlaBF16;

typedef struct {
    const uint16_t *gate;
    const uint16_t *up;
    const uint16_t *down;
    int intermediate_size;
} KvlMlpBF16;

float kvl_bf16_to_f32(uint16_t x);
void kvl_matvec_bf16(float *y, const float *x, const uint16_t *w,
                     int in, int out);
/* Experimental expert-only format: each matrix blob is [out FP32 row scales]
 * followed by row-major signed int8 weights [out,in]. */
void kvl_matvec_q8_rowwise(float *y, const float *x, const void *blob,
                           int in, int out);
/* Experimental Q5 expert-only format: each matrix blob is FP32 scales for
 * contiguous input groups of 128, then a row-major packed 5-bit signed stream.
 * Values use symmetric RTN range [-15,15]. */
void kvl_matvec_q5_g128(float *y, const float *x, const void *blob,
                        int in, int out);
void kvl_silu_mul(float *y, const float *gate, const float *up, int n);
void kvl_rmsnorm_bf16(float *y, const float *x, const uint16_t *weight,
                      int n, float eps);
int kvl_mla_prefill_bf16(float *out, const float *x, int seq_len,
                         const KvlMlaConfig *cfg, const KvlMlaBF16 *w);

int kvl_mlp_bf16(float *y, const float *x, const KvlMlpBF16 *mlp,
                 int hidden_size, float *scratch);

int kvl_router_noaux_tc(const KvlRouterConfig *cfg, const float *x,
                        const float *router_weight, const float *correction_bias,
                        int *top_ids, float *top_weights);

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

/* Lab dispatch: BF16 stores delegate to the production function above byte-for-byte;
 * Q8/Q5 stores alter only routed expert gate/up/down matrices. */
int kvl_moe_token_auto(KvlExpertCache *cache, int layer,
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
