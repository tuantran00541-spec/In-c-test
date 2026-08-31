#include "kvl/ops.h"

#include <stddef.h>

#define Q8_0_BLOCK 32u
#define Q8_0_BYTES 34u

static size_t q8_0_matrix_bytes(int in, int out) {
    if (in <= 0 || out <= 0 || (in % (int)Q8_0_BLOCK) != 0) return 0;
    return (size_t)out * ((size_t)in / Q8_0_BLOCK) * Q8_0_BYTES;
}

int kvl_moe_token_gguf_q8_auto(KvlExpertCache *cache, int layer,
                               const KvlRouterConfig *router_cfg,
                               const float *x,
                               const float *router_weight,
                               const float *correction_bias,
                               int expert_intermediate_size,
                               const KvlMlpBF16 *shared,
                               float *out,
                               int *top_ids,
                               float *top_weights,
                               float *scratch) {
    if (!cache || !cache->store || cache->store->hdr.dtype != KVL_DTYPE_GGUF_Q8_0 ||
        !router_cfg || !x || !router_weight || !correction_bias || !out ||
        !top_ids || !top_weights || !scratch || expert_intermediate_size <= 0)
        return -1;

    const int H = router_cfg->hidden_size;
    const int I = expert_intermediate_size;
    const int maxI = (shared && shared->intermediate_size > I) ? shared->intermediate_size : I;
    if (kvl_router_noaux_tc(router_cfg, x, router_weight, correction_bias,
                            top_ids, top_weights) != 0)
        return -1;

    (void)kvl_expert_cache_getmany(cache, layer, top_ids, router_cfg->top_k);
    for (int i = 0; i < H; ++i) out[i] = 0.0f;

    float *gate = scratch;
    float *up = gate + maxI;
    float *act = up + maxI;
    float *tmp = act + maxI;

    const size_t need_gu = q8_0_matrix_bytes(H, I);
    const size_t need_dn = q8_0_matrix_bytes(I, H);
    if (!need_gu || !need_dn) return -1;

    for (int j = 0; j < router_cfg->top_k; ++j) {
        KvlCachedExpert q;
        if (kvl_expert_cache_get(cache, layer, top_ids[j], &q) != 0) return -1;
        if (q.record->gate_bytes != need_gu || q.record->up_bytes != need_gu ||
            q.record->down_bytes != need_dn)
            return -1;
        kvl_matvec_ggml_q8_0(gate, x, q.gate, H, I);
        kvl_matvec_ggml_q8_0(up, x, q.up, H, I);
        kvl_silu_mul(act, gate, up, I);
        kvl_matvec_ggml_q8_0(tmp, act, q.down, I, H);
        const float w = top_weights[j];
        for (int i = 0; i < H; ++i) out[i] += w * tmp[i];
    }

    if (shared) {
        if (kvl_mlp_bf16(tmp, x, shared, H, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) out[i] += tmp[i];
    }
    return 0;
}
