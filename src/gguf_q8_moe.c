#include "kvl/ops.h"

#include <stddef.h>
#include <stdio.h>

#define Q8_0_BLOCK 32u
#define Q8_0_BYTES 34u
#define LAYER_PIN_COUNT 2

/* Research-only phase detector for the separate layer-pin generator. The current
 * text prefill is layer-major: L1 is called for every prompt token, then L2, ...,
 * through L26. The first autoregressive forward therefore produces the unique
 * transition L26 -> L1. Pins stay completely disabled until that transition, so
 * the established prefill cache behavior is unchanged. One generation process
 * owns one cache; if that assumption changes, this pilot must be redesigned. */
static KvlExpertCache *g_layerpin_cache = NULL;
static int g_layerpin_last_layer = -1;

static int layerpin_phase_update(KvlExpertCache *cache, int layer) {
    if (cache != g_layerpin_cache) {
        g_layerpin_cache = cache;
        g_layerpin_last_layer = -1;
    }
    if (!cache->pinned_of && g_layerpin_last_layer == 26 && layer == 1) {
        /* n=0 on dense L0 is only an activation sentinel: it lazily allocates
         * pinned_of without protecting any record. The routed layers populate
         * their two pins as the first decode forward walks L1..L26. */
        if (kvl_expert_cache_pin_layer(cache, 0, NULL, 0) != 0) return -1;
        fprintf(stderr, "kvl_layerpin: activated_after_prefill=yes slots=%d\n", cache->n_slots);
    }
    g_layerpin_last_layer = layer;
    return 0;
}

static size_t q8_0_matrix_bytes(int in, int out) {
    if (in <= 0 || out <= 0 || (in % (int)Q8_0_BLOCK) != 0) return 0;
    return (size_t)out * ((size_t)in / Q8_0_BLOCK) * Q8_0_BYTES;
}

static int pin_heaviest_routes(KvlExpertCache *cache, int layer,
                               const KvlRouterConfig *router_cfg,
                               const int *top_ids, const float *top_weights) {
    if (!cache || !cache->store || !router_cfg || !top_ids || !top_weights)
        return -1;

    const int n_pin = router_cfg->top_k < LAYER_PIN_COUNT ? router_cfg->top_k : LAYER_PIN_COUNT;
    if (n_pin <= 0) return kvl_expert_cache_pin_layer(cache, layer, NULL, 0);

    /* This pilot intentionally reserves two experts for every routed layer and
     * leaves one full top-k batch transient. Kimi has one dense layer (L0), so
     * 26*2 + 6 = 58 slots, exactly the 512 MiB direct-GGUF cache. Refuse smaller
     * caches instead of silently degrading into a partially pinned policy. */
    const int routed_layers = (int)cache->store->hdr.n_layers - 1;
    const int needed_slots = routed_layers * LAYER_PIN_COUNT + router_cfg->top_k;
    if (routed_layers <= 0 || cache->n_slots < needed_slots) return -1;

    int best[LAYER_PIN_COUNT] = {-1, -1};
    for (int j = 0; j < router_cfg->top_k; ++j) {
        if (best[0] < 0 || top_weights[j] > top_weights[best[0]]) {
            best[1] = best[0];
            best[0] = j;
        } else if (best[1] < 0 || top_weights[j] > top_weights[best[1]]) {
            best[1] = j;
        }
    }

    int pins[LAYER_PIN_COUNT];
    for (int i = 0; i < n_pin; ++i) {
        if (best[i] < 0) return -1;
        pins[i] = top_ids[best[i]];
    }
    return kvl_expert_cache_pin_layer(cache, layer, pins, n_pin);
}

static int moe_token_gguf_q8_impl(KvlExpertCache *cache, int layer,
                                  const KvlRouterConfig *router_cfg,
                                  const float *x,
                                  const float *router_weight,
                                  const float *correction_bias,
                                  int expert_intermediate_size,
                                  const KvlMlpBF16 *shared,
                                  float *out,
                                  int *top_ids,
                                  float *top_weights,
                                  float *scratch,
                                  int use_layer_pins) {
    if (!cache || !cache->store || cache->store->hdr.dtype != KVL_DTYPE_GGUF_Q8_0 ||
        !router_cfg || !x || !router_weight || !correction_bias || !out ||
        !top_ids || !top_weights || !scratch || expert_intermediate_size <= 0)
        return -1;

    if (use_layer_pins && layerpin_phase_update(cache, layer) != 0) return -1;

    const int H = router_cfg->hidden_size;
    const int I = expert_intermediate_size;
    const int maxI = (shared && shared->intermediate_size > I) ? shared->intermediate_size : I;
    if (kvl_router_noaux_tc(router_cfg, x, router_weight, correction_bias,
                            top_ids, top_weights) != 0)
        return -1;

    /* pinned_of is NULL for all prefill calls. It becomes non-NULL only at the
     * L26->L1 transition above, so prefill remains the exact baseline cache path. */
    if (use_layer_pins && cache->pinned_of &&
        pin_heaviest_routes(cache, layer, router_cfg, top_ids, top_weights) != 0)
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
    return moe_token_gguf_q8_impl(cache, layer, router_cfg, x, router_weight,
                                  correction_bias, expert_intermediate_size,
                                  shared, out, top_ids, top_weights, scratch, 0);
}

int kvl_moe_token_gguf_q8_layerpin_auto(KvlExpertCache *cache, int layer,
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
    return moe_token_gguf_q8_impl(cache, layer, router_cfg, x, router_weight,
                                  correction_bias, expert_intermediate_size,
                                  shared, out, top_ids, top_weights, scratch, 1);
}
