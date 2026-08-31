#include "kvl/ops.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

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
static uint64_t g_layerpin_hyst_decode_passes = 0;
static uint64_t g_layerpin_hyst_retained = 0;

static int layerpin_hysteresis_enabled(void) {
    const char *v = getenv("KVL_LAYERPIN_HYSTERESIS");
    return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

static int layerpin_phase_update(KvlExpertCache *cache, int layer,
                                 int use_hysteresis) {
    if (cache != g_layerpin_cache) {
        g_layerpin_cache = cache;
        g_layerpin_last_layer = -1;
        g_layerpin_hyst_decode_passes = 0;
        g_layerpin_hyst_retained = 0;
    }
    if (!cache->pinned_of && g_layerpin_last_layer == 26 && layer == 1) {
        /* n=0 on dense L0 is only an activation sentinel: it lazily allocates
         * pinned_of without protecting any record. The routed layers populate
         * their two pins as the first decode forward walks L1..L26. */
        if (kvl_expert_cache_pin_layer(cache, 0, NULL, 0) != 0) return -1;
        fprintf(stderr,
                "kvl_layerpin: activated_after_prefill=yes slots=%d policy=%s\n",
                cache->n_slots, use_hysteresis ? "hysteresis" : "topweight");
    }
    g_layerpin_last_layer = layer;
    return 0;
}

static size_t q8_0_matrix_bytes(int in, int out) {
    if (in <= 0 || out <= 0 || (in % (int)Q8_0_BLOCK) != 0) return 0;
    return (size_t)out * ((size_t)in / Q8_0_BLOCK) * Q8_0_BYTES;
}

static int layerpin_is_pinned(const KvlExpertCache *cache, int layer, int expert) {
    if (!cache || !cache->store || !cache->pinned_of ||
        layer < 0 || expert < 0 ||
        layer >= (int)cache->store->hdr.n_layers ||
        expert >= (int)cache->store->hdr.n_experts)
        return 0;
    const size_t key = (size_t)layer * cache->store->hdr.n_experts + (size_t)expert;
    return cache->pinned_of[key] != 0;
}

static int expert_already_selected(const int *pins, int n, int expert) {
    for (int i = 0; i < n; ++i)
        if (pins[i] == expert) return 1;
    return 0;
}

static int pin_routes(KvlExpertCache *cache, int layer,
                      const KvlRouterConfig *router_cfg,
                      const int *top_ids, const float *top_weights,
                      int use_hysteresis, int *retained_out) {
    if (!cache || !cache->store || !router_cfg || !top_ids || !top_weights)
        return -1;

    const int n_pin = router_cfg->top_k < LAYER_PIN_COUNT ? router_cfg->top_k : LAYER_PIN_COUNT;
    if (retained_out) *retained_out = 0;
    if (n_pin <= 0) return kvl_expert_cache_pin_layer(cache, layer, NULL, 0);

    /* This pilot intentionally reserves two experts for every routed layer and
     * leaves one full top-k batch transient. Kimi has one dense layer (L0), so
     * 26*2 + 6 = 58 slots, exactly the 512 MiB direct-GGUF cache. Refuse smaller
     * caches instead of silently degrading into a partially pinned policy. */
    const int routed_layers = (int)cache->store->hdr.n_layers - 1;
    const int needed_slots = routed_layers * LAYER_PIN_COUNT + router_cfg->top_k;
    if (routed_layers <= 0 || cache->n_slots < needed_slots) return -1;

    int pins[LAYER_PIN_COUNT] = {-1, -1};
    int n_selected = 0;
    int retained = 0;

    if (use_hysteresis && cache->pinned_of) {
        /* Hysteresis is deliberately routing-safe: an old pin is retained only
         * if the CURRENT router still selected that exact expert in top-k. No
         * expert outside current top-k is kept merely because it was hot before.
         * If both old pins survive, both win regardless of their current rank;
         * this trades pin churn for temporal reuse without changing MoE math. */
        int keep_idx[LAYER_PIN_COUNT] = {-1, -1};
        for (int j = 0; j < router_cfg->top_k; ++j) {
            if (!layerpin_is_pinned(cache, layer, top_ids[j])) continue;
            for (int p = 0; p < n_pin; ++p) {
                if (keep_idx[p] < 0 || top_weights[j] > top_weights[keep_idx[p]]) {
                    for (int q = n_pin - 1; q > p; --q) keep_idx[q] = keep_idx[q - 1];
                    keep_idx[p] = j;
                    break;
                }
            }
        }
        for (int p = 0; p < n_pin; ++p) {
            if (keep_idx[p] < 0) continue;
            const int e = top_ids[keep_idx[p]];
            if (expert_already_selected(pins, n_selected, e)) continue;
            pins[n_selected++] = e;
            retained++;
        }
    }

    /* Fill remaining pin slots with the heaviest current routes. With
     * hysteresis disabled this is byte-for-byte the old top-weight policy's
     * selection rule, including stable first-occurrence behavior on ties. */
    while (n_selected < n_pin) {
        int best = -1;
        for (int j = 0; j < router_cfg->top_k; ++j) {
            if (expert_already_selected(pins, n_selected, top_ids[j])) continue;
            if (best < 0 || top_weights[j] > top_weights[best]) best = j;
        }
        if (best < 0) return -1;
        pins[n_selected++] = top_ids[best];
    }

    if (retained_out) *retained_out = retained;
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

    const int use_hysteresis = use_layer_pins && layerpin_hysteresis_enabled();
    if (use_layer_pins &&
        layerpin_phase_update(cache, layer, use_hysteresis) != 0)
        return -1;

    const int H = router_cfg->hidden_size;
    const int I = expert_intermediate_size;
    const int maxI = (shared && shared->intermediate_size > I) ? shared->intermediate_size : I;
    if (kvl_router_noaux_tc(router_cfg, x, router_weight, correction_bias,
                            top_ids, top_weights) != 0)
        return -1;

    /* pinned_of is NULL for all prefill calls. It becomes non-NULL only at the
     * L26->L1 transition above, so prefill remains the exact baseline cache path. */
    if (use_layer_pins && cache->pinned_of) {
        int retained = 0;
        if (pin_routes(cache, layer, router_cfg, top_ids, top_weights,
                       use_hysteresis, &retained) != 0)
            return -1;
        if (use_hysteresis) {
            g_layerpin_hyst_retained += (uint64_t)retained;
            if (layer == 26) {
                g_layerpin_hyst_decode_passes++;
                fprintf(stderr,
                        "kvl_layerpin_hyst: decode_pass=%llu retained_total=%llu\n",
                        (unsigned long long)g_layerpin_hyst_decode_passes,
                        (unsigned long long)g_layerpin_hyst_retained);
            }
        }
    }

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
