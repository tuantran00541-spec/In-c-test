#define _POSIX_C_SOURCE 200809L
#define main kvl_generate_text_entry_unused
#include "generate.c"
#undef main

#include "kvl/mla_compressed_q8_state.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double lab_wall_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double lab_wall_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
#endif

static int argmax_lab(const float *logits) {
    int best = 0;
    for (int i = 1; i < V; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

static uint16_t f32_to_bf16_rne_lab(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    if ((u & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000))
        u += UINT32_C(0x00007fff) + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static int clone_f32_states_lab(KvlMlaCompressedState *dst,
                                const KvlMlaCompressedState *src,
                                const KvlMlaConfig *cfg,
                                int capacity) {
    for (int layer = 0; layer < LN; ++layer) {
        if (kvl_mla_compressed_state_init(&dst[layer], cfg, capacity) != 0)
            return -1;
        if (src[layer].len < 0 || src[layer].len > capacity ||
            src[layer].kv_lora_rank != R || src[layer].qk_rope_dim != DR)
            return -1;
        memcpy(dst[layer].latent, src[layer].latent,
               (size_t)src[layer].len * R * sizeof(float));
        memcpy(dst[layer].rope, src[layer].rope,
               (size_t)src[layer].len * DR * sizeof(float));
        dst[layer].len = src[layer].len;
    }
    return 0;
}

static void free_f32_states_lab(KvlMlaCompressedState *states) {
    for (int layer = 0; layer < LN; ++layer)
        kvl_mla_compressed_state_free(&states[layer]);
}

static double f32_state_max_abs_lab(const KvlMlaCompressedState *a,
                                    const KvlMlaCompressedState *b) {
    double m = 0.0;
    for (int layer = 0; layer < LN; ++layer) {
        if (a[layer].len != b[layer].len) return INFINITY;
        for (int t = 0; t < a[layer].len; ++t) {
            for (int r = 0; r < R; ++r) {
                const double d = fabs((double)a[layer].latent[(size_t)t * R + r] -
                                      (double)b[layer].latent[(size_t)t * R + r]);
                if (d > m) m = d;
            }
            for (int d = 0; d < DR; ++d) {
                const double e = fabs((double)a[layer].rope[(size_t)t * DR + d] -
                                      (double)b[layer].rope[(size_t)t * DR + d]);
                if (e > m) m = e;
            }
        }
    }
    return m;
}

static void quantize_latent_row_lab(int8_t *dst, float *scale_out,
                                    const float *src, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    if (!(amax > 0.0f) || !isfinite(amax)) {
        memset(dst, 0, (size_t)n);
        *scale_out = 1.0f;
        return;
    }
    const float scale = amax / 127.0f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        float q = roundf(src[i] * inv);
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        dst[i] = (int8_t)q;
    }
    *scale_out = scale;
}

static int clone_q8_from_f32_lab(KvlMlaCompressedQ8State *dst,
                                 const KvlMlaCompressedState *src,
                                 const KvlMlaConfig *cfg,
                                 int capacity) {
    for (int layer = 0; layer < LN; ++layer) {
        if (kvl_mla_compressed_q8_state_init(&dst[layer], cfg, capacity) != 0)
            return -1;
        if (src[layer].len < 0 || src[layer].len > capacity) return -1;
        for (int t = 0; t < src[layer].len; ++t) {
            quantize_latent_row_lab(dst[layer].latent_q8 + (size_t)t * R,
                                    dst[layer].latent_scale + t,
                                    src[layer].latent + (size_t)t * R, R);
            for (int d = 0; d < DR; ++d)
                dst[layer].rope_bf16[(size_t)t * DR + d] =
                    f32_to_bf16_rne_lab(src[layer].rope[(size_t)t * DR + d]);
        }
        dst[layer].len = src[layer].len;
    }
    return 0;
}

static void free_q8_states_lab(KvlMlaCompressedQ8State *states) {
    for (int layer = 0; layer < LN; ++layer)
        kvl_mla_compressed_q8_state_free(&states[layer]);
}

static int attention_q8_lab(KvlTrunkStore *ts, int layer, const float *x,
                            int position, KvlMlaCompressedQ8State *state,
                            float *r1, float *n2, float *n1, float *attn) {
    KvlTrunkTensor in = {0}, q = {0}, kva = {0}, kvan = {0};
    KvlTrunkTensor kvb = {0}, o = {0}, pn = {0};
    int rc = -1;
    if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_INPUT_NORM, &in) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_Q_PROJ, &q) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_KV_A_PROJ, &kva) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_KV_A_NORM, &kvan) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_KV_B_PROJ, &kvb) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_O_PROJ, &o) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_POST_ATTN_NORM, &pn))
        goto done;
    kvl_rmsnorm_bf16(n1, x, (const uint16_t *)in.data, H, RMS_EPS);
    KvlMlaConfig cfg = {H, NH, DN, DR, DV, R, RMS_EPS, ROPE_THETA};
    KvlMlaBF16 aw = {
        (const uint16_t *)q.data, (const uint16_t *)kva.data,
        (const uint16_t *)kvan.data, (const uint16_t *)kvb.data,
        (const uint16_t *)o.data
    };
    if (kvl_mla_decode_compressed_q8_bf16(attn, n1, position, &cfg, &aw, state) != 0)
        goto done;
    for (int i = 0; i < H; ++i) r1[i] = x[i] + attn[i];
    kvl_rmsnorm_bf16(n2, r1, (const uint16_t *)pn.data, H, RMS_EPS);
    rc = 0;
done:
    kvl_trunk_tensor_free(&in); kvl_trunk_tensor_free(&q);
    kvl_trunk_tensor_free(&kva); kvl_trunk_tensor_free(&kvan);
    kvl_trunk_tensor_free(&kvb); kvl_trunk_tensor_free(&o);
    kvl_trunk_tensor_free(&pn);
    return rc;
}

static int shared_only_mlp_lab(KvlTrunkStore *ts, int layer,
                               const float *n, float *out, float *scratch) {
    KvlTrunkTensor sg = {0}, su = {0}, sd = {0};
    int rc = -1;
    if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_GATE, &sg) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_UP, &su) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_DOWN, &sd))
        goto done;
    KvlMlpBF16 shared = {
        (const uint16_t *)sg.data, (const uint16_t *)su.data,
        (const uint16_t *)sd.data, SHARED_I
    };
    rc = kvl_mlp_bf16(out, n, &shared, H, scratch);
done:
    kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
    kvl_trunk_tensor_free(&sd);
    return rc;
}

static int draft_forward_q8_lab(KvlTrunkStore *ts, KvlExpertCache *cache,
                                KvlMlaCompressedQ8State *states,
                                const KvlTrunkTensor *emb,
                                const KvlTrunkTensor *final_norm,
                                const KvlTrunkTensor *lm_head,
                                int token_id, int position, uint32_t skip_mask,
                                float *logits, float *x, float *r1, float *n2,
                                float *y, float *n1, float *attn,
                                float *router, float *bias, int *ids,
                                float *weights, float *scratch, float *z) {
    if (token_id < 0 || token_id >= V) return -1;
    expand(x, (const uint16_t *)emb->data + (size_t)token_id * H, H);
    for (int layer = 0; layer < LN; ++layer) {
        if (attention_q8_lab(ts, layer, x, position, &states[layer],
                             r1, n2, n1, attn) != 0)
            return -1;
        const int skip_routed = layer > 0 && ((skip_mask >> layer) & 1u);
        if (skip_routed) {
            if (shared_only_mlp_lab(ts, layer, n2, y, scratch) != 0) return -1;
        } else {
            if (mlp_token(ts, cache, layer, n2, y, router, bias,
                          ids, weights, scratch) != 0)
                return -1;
        }
        for (int i = 0; i < H; ++i) x[i] = r1[i] + y[i];
    }
    kvl_rmsnorm_bf16(z, x, (const uint16_t *)final_norm->data, H, RMS_EPS);
    kvl_matvec_bf16(logits, z, (const uint16_t *)lm_head->data, H, V);
    return 0;
}

static int exact_block_lab(KvlTrunkStore *ts, KvlExpertCache *cache,
                           KvlMlaCompressedState *states,
                           const KvlTrunkTensor *emb,
                           const KvlTrunkTensor *final_norm,
                           const KvlTrunkTensor *lm_head,
                           const int *token_ids, int count, int start_position,
                           int *next_ids, float *last_logits) {
    if (!ts || !cache || !states || !emb || !final_norm || !lm_head ||
        !token_ids || count <= 0 || start_position < 0 || !next_ids || !last_logits)
        return -1;
    for (int layer = 0; layer < LN; ++layer)
        if (states[layer].len != start_position ||
            count > states[layer].capacity - start_position)
            return -1;

    const size_t elems = (size_t)count * H;
    float *x = (float *)malloc(elems * sizeof(float));
    float *a = (float *)malloc(elems * sizeof(float));
    float *b = (float *)malloc(elems * sizeof(float));
    float *router = (float *)malloc((size_t)E * H * sizeof(float));
    float *bias = (float *)malloc((size_t)E * sizeof(float));
    int *ids = (int *)malloc((size_t)TOPK * sizeof(int));
    float *weights = (float *)malloc((size_t)TOPK * sizeof(float));
    float *scratch = (float *)malloc((size_t)(3 * DENSE_I + H) * sizeof(float));
    float *z = (float *)malloc((size_t)H * sizeof(float));
    int rc = -1;
    if (!x || !a || !b || !router || !bias || !ids || !weights || !scratch || !z)
        goto done;
    for (int t = 0; t < count; ++t) {
        if (token_ids[t] < 0 || token_ids[t] >= V) goto done;
        expand(x + (size_t)t * H,
               (const uint16_t *)emb->data + (size_t)token_ids[t] * H, H);
    }
    KvlMlaConfig cfg = {H, NH, DN, DR, DV, R, RMS_EPS, ROPE_THETA};
    KvlRouterConfig router_cfg = {H, E, TOPK, 1, 1, 1, 2.446f};
    for (int layer = 0; layer < LN; ++layer) {
        KvlTrunkTensor in = {0}, q = {0}, kva = {0}, kvan = {0};
        KvlTrunkTensor kvb = {0}, o = {0}, pn = {0};
        if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_INPUT_NORM, &in) ||
            load_kind(ts, (uint32_t)layer, KVL_TENSOR_Q_PROJ, &q) ||
            load_kind(ts, (uint32_t)layer, KVL_TENSOR_KV_A_PROJ, &kva) ||
            load_kind(ts, (uint32_t)layer, KVL_TENSOR_KV_A_NORM, &kvan) ||
            load_kind(ts, (uint32_t)layer, KVL_TENSOR_KV_B_PROJ, &kvb) ||
            load_kind(ts, (uint32_t)layer, KVL_TENSOR_O_PROJ, &o) ||
            load_kind(ts, (uint32_t)layer, KVL_TENSOR_POST_ATTN_NORM, &pn)) {
            kvl_trunk_tensor_free(&in); kvl_trunk_tensor_free(&q);
            kvl_trunk_tensor_free(&kva); kvl_trunk_tensor_free(&kvan);
            kvl_trunk_tensor_free(&kvb); kvl_trunk_tensor_free(&o);
            kvl_trunk_tensor_free(&pn);
            goto done;
        }
        for (int t = 0; t < count; ++t)
            kvl_rmsnorm_bf16(a + (size_t)t * H, x + (size_t)t * H,
                             (const uint16_t *)in.data, H, RMS_EPS);
        KvlMlaBF16 aw = {
            (const uint16_t *)q.data, (const uint16_t *)kva.data,
            (const uint16_t *)kvan.data, (const uint16_t *)kvb.data,
            (const uint16_t *)o.data
        };
        if (kvl_mla_decode_compressed_block_bf16(b, a, count, start_position,
                                                  &cfg, &aw, &states[layer]) != 0) {
            kvl_trunk_tensor_free(&in); kvl_trunk_tensor_free(&q);
            kvl_trunk_tensor_free(&kva); kvl_trunk_tensor_free(&kvan);
            kvl_trunk_tensor_free(&kvb); kvl_trunk_tensor_free(&o);
            kvl_trunk_tensor_free(&pn);
            goto done;
        }
        for (int t = 0; t < count; ++t) {
            const size_t base = (size_t)t * H;
            for (int i = 0; i < H; ++i) b[base + i] = x[base + i] + b[base + i];
            kvl_rmsnorm_bf16(a + base, b + base,
                             (const uint16_t *)pn.data, H, RMS_EPS);
        }
        kvl_trunk_tensor_free(&in); kvl_trunk_tensor_free(&q);
        kvl_trunk_tensor_free(&kva); kvl_trunk_tensor_free(&kvan);
        kvl_trunk_tensor_free(&kvb); kvl_trunk_tensor_free(&o);
        kvl_trunk_tensor_free(&pn);

        if (layer == 0) {
            KvlTrunkTensor g = {0}, u = {0}, d = {0};
            if (load_kind(ts, 0, KVL_TENSOR_DENSE_GATE, &g) ||
                load_kind(ts, 0, KVL_TENSOR_DENSE_UP, &u) ||
                load_kind(ts, 0, KVL_TENSOR_DENSE_DOWN, &d)) {
                kvl_trunk_tensor_free(&g); kvl_trunk_tensor_free(&u);
                kvl_trunk_tensor_free(&d); goto done;
            }
            KvlMlpBF16 dense = {
                (const uint16_t *)g.data, (const uint16_t *)u.data,
                (const uint16_t *)d.data, DENSE_I
            };
            for (int t = 0; t < count; ++t)
                if (kvl_mlp_bf16(x + (size_t)t * H, a + (size_t)t * H,
                                 &dense, H, scratch) != 0) {
                    kvl_trunk_tensor_free(&g); kvl_trunk_tensor_free(&u);
                    kvl_trunk_tensor_free(&d); goto done;
                }
            kvl_trunk_tensor_free(&g); kvl_trunk_tensor_free(&u);
            kvl_trunk_tensor_free(&d);
        } else {
            KvlTrunkTensor rt = {0}, rb = {0}, sg = {0}, su = {0}, sd = {0};
            if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_WEIGHT, &rt) ||
                load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_BIAS, &rb) ||
                load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_GATE, &sg) ||
                load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_UP, &su) ||
                load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_DOWN, &sd)) {
                kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
                kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
                kvl_trunk_tensor_free(&sd); goto done;
            }
            expand(router, (const uint16_t *)rt.data, (size_t)E * H);
            expand(bias, (const uint16_t *)rb.data, E);
            KvlMlpBF16 shared = {
                (const uint16_t *)sg.data, (const uint16_t *)su.data,
                (const uint16_t *)sd.data, SHARED_I
            };
            for (int t = 0; t < count; ++t)
                if (kvl_moe_token_bf16(cache, layer, &router_cfg, a + (size_t)t * H,
                                       router, bias, EXP_I, &shared,
                                       x + (size_t)t * H, ids, weights, scratch) != 0) {
                    kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
                    kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
                    kvl_trunk_tensor_free(&sd); goto done;
                }
            kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
            kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
            kvl_trunk_tensor_free(&sd);
        }
        for (size_t i = 0; i < elems; ++i) x[i] = b[i] + x[i];
    }
    for (int t = 0; t < count; ++t) {
        kvl_rmsnorm_bf16(z, x + (size_t)t * H,
                         (const uint16_t *)final_norm->data, H, RMS_EPS);
        kvl_matvec_bf16(last_logits, z, (const uint16_t *)lm_head->data, H, V);
        next_ids[t] = argmax_lab(last_logits);
    }
    rc = 0;
done:
    free(x); free(a); free(b); free(router); free(bias);
    free(ids); free(weights); free(scratch); free(z);
    return rc;
}

static uint32_t bit_layers_lab(const int *layers, int n) {
    uint32_t m = 0;
    for (int i = 0; i < n; ++i)
        if (layers[i] > 0 && layers[i] < LN) m |= UINT32_C(1) << layers[i];
    return m;
}

static int popcount_mask_lab(uint32_t m) {
    int n = 0;
    while (m) { n += (int)(m & 1u); m >>= 1; }
    return n;
}

typedef struct {
    const char *name;
    uint32_t mask;
    int focused;
} DraftMask;

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr,
                "usage: %s trunk.bin trunk.idx experts.bin experts.idx prompt.ids "
                "cache_bytes block_tokens\n", argv[0]);
        return 2;
    }
    const size_t cache_bytes = (size_t)strtoull(argv[6], NULL, 10);
    const int K = atoi(argv[7]);
    if (cache_bytes == 0 || K < 2 || K > 8) return 2;

    int *prompt = NULL, prompt_n = 0;
    if (read_prompt_ids(argv[5], &prompt, &prompt_n) != 0) return 2;
    const int capacity = prompt_n + K + 1;

    KvlTrunkStore ts;
    KvlExpertStore es;
    if (kvl_trunk_store_open(&ts, argv[1], argv[2], 1) != 0 ||
        kvl_expert_store_open(&es, argv[3], argv[4], 1) != 0) {
        free(prompt); return 2;
    }
    KvlTrunkTensor emb = {0}, fn = {0}, lm = {0};
    if (load_kind(&ts, KVL_TRUNK_GLOBAL_LAYER, KVL_TENSOR_EMBED_TOKENS, &emb) ||
        load_kind(&ts, KVL_TRUNK_GLOBAL_LAYER, KVL_TENSOR_FINAL_NORM, &fn) ||
        load_kind(&ts, KVL_TRUNK_GLOBAL_LAYER, KVL_TENSOR_LM_HEAD, &lm))
        return 2;

    KvlMlaConfig mc = {H, NH, DN, DR, DV, R, RMS_EPS, ROPE_THETA};
    KvlMlaCompressedState base[LN], baseline_states[LN];
    memset(base, 0, sizeof base); memset(baseline_states, 0, sizeof baseline_states);
    for (int l = 0; l < LN; ++l)
        if (kvl_mla_compressed_state_init(&base[l], &mc, capacity) != 0) return 2;

    float *prefill_logits = (float *)malloc((size_t)V * sizeof(float));
    float *logits = (float *)malloc((size_t)V * sizeof(float));
    float *x = (float *)malloc((size_t)H * sizeof(float));
    float *r1 = (float *)malloc((size_t)H * sizeof(float));
    float *n2 = (float *)malloc((size_t)H * sizeof(float));
    float *y = (float *)malloc((size_t)H * sizeof(float));
    float *n1 = (float *)malloc((size_t)H * sizeof(float));
    float *attn = (float *)malloc((size_t)H * sizeof(float));
    float *z = (float *)malloc((size_t)H * sizeof(float));
    float *router = (float *)malloc((size_t)E * H * sizeof(float));
    float *bias = (float *)malloc((size_t)E * sizeof(float));
    int *ids = (int *)malloc((size_t)TOPK * sizeof(int));
    float *weights = (float *)malloc((size_t)TOPK * sizeof(float));
    float *scratch = (float *)malloc((size_t)(3 * DENSE_I + H) * sizeof(float));
    int *baseline_tokens = (int *)malloc((size_t)(K + 1) * sizeof(int));
    double *baseline_wall = (double *)calloc((size_t)(K + 2), sizeof(double));
    uint64_t *baseline_bytes = (uint64_t *)calloc((size_t)(K + 2), sizeof(uint64_t));
    int *draft = (int *)malloc((size_t)K * sizeof(int));
    int *target_next = (int *)malloc((size_t)K * sizeof(int));
    if (!prefill_logits || !logits || !x || !r1 || !n2 || !y || !n1 || !attn ||
        !z || !router || !bias || !ids || !weights || !scratch || !baseline_tokens ||
        !baseline_wall || !baseline_bytes || !draft || !target_next)
        return 2;

    KvlExpertCache cache;
    memset(&cache, 0, sizeof cache);
    if (kvl_expert_cache_init(&cache, &es, cache_bytes) != 0) return 2;
    if (prefill_prompt(&ts, &cache, base, &emb, &fn, &lm, prompt, prompt_n,
                       prefill_logits, router, bias, ids, weights, scratch, z) != 0)
        return 1;
    kvl_expert_cache_close(&cache);

    if (clone_f32_states_lab(baseline_states, base, &mc, capacity) != 0) return 1;
    memcpy(logits, prefill_logits, (size_t)V * sizeof(float));
    if (kvl_expert_cache_init(&cache, &es, cache_bytes) != 0) return 2;
    const double baseline_t0 = lab_wall_s();
    for (int t = 0; t < K + 1; ++t) {
        baseline_tokens[t] = argmax_lab(logits);
        if (forward_token(&ts, &cache, baseline_states, &emb, &fn, &lm,
                          baseline_tokens[t], prompt_n + t, 1, logits,
                          x, r1, n2, y, router, bias, ids, weights, scratch, z) != 0)
            return 1;
        baseline_wall[t + 1] = lab_wall_s() - baseline_t0;
        baseline_bytes[t + 1] = cache.bytes_read;
    }
    fprintf(stderr, "BASELINE_CACHE ");
    kvl_expert_cache_report(&cache);
    kvl_expert_cache_close(&cache);

    const int s4[] = {6, 12, 18, 24};
    const int s7[] = {4, 8, 12, 16, 20, 24, 26};
    const int s13[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26};
    const uint32_t m4 = bit_layers_lab(s4, (int)(sizeof s4 / sizeof s4[0]));
    const uint32_t m7 = bit_layers_lab(s7, (int)(sizeof s7 / sizeof s7[0]));
    const uint32_t m13 = bit_layers_lab(s13, (int)(sizeof s13 / sizeof s13[0]));
    DraftMask masks[] = {
        {"q8_only", 0, 0},
        {"skip4", m4, 0},
        {"skip7", m7, 0},
        {"skip13", m13, 0},
        {"skip12_keep2",  m13 & ~(UINT32_C(1) << 2), 1},
        {"skip12_keep6",  m13 & ~(UINT32_C(1) << 6), 1},
        {"skip12_keep10", m13 & ~(UINT32_C(1) << 10), 1},
        {"skip12_keep14", m13 & ~(UINT32_C(1) << 14), 1},
        {"skip12_keep18", m13 & ~(UINT32_C(1) << 18), 1},
        {"skip12_keep22", m13 & ~(UINT32_C(1) << 22), 1}
    };
    const int nmasks = (int)(sizeof masks / sizeof masks[0]);

    KvlMlaCompressedQ8State q8_probe[LN];
    memset(q8_probe, 0, sizeof q8_probe);
    if (clone_q8_from_f32_lab(q8_probe, base, &mc, capacity) != 0) return 1;
    size_t f32_bytes = 0, q8_bytes = 0;
    for (int l = 0; l < LN; ++l) {
        f32_bytes += kvl_mla_compressed_state_bytes(&base[l]);
        q8_bytes += kvl_mla_compressed_q8_state_bytes(&q8_probe[l]);
    }
    free_q8_states_lab(q8_probe);
    printf("SELF_SPEC_LAYOUT prompt=%d block=%d target_state_mib=%.3f draft_q8_state_mib=%.3f ratio=%.3fx\n",
           prompt_n, K, f32_bytes / 1048576.0, q8_bytes / 1048576.0,
           q8_bytes ? (double)f32_bytes / (double)q8_bytes : 0.0);

    int failures = 0;
    int focus_exact = 0;
    int focus_full_accept = 0;
    for (int mi = 0; mi < nmasks; ++mi) {
        KvlMlaCompressedQ8State draft_states[LN];
        KvlMlaCompressedState target_states[LN], cmp_states[LN];
        memset(draft_states, 0, sizeof draft_states);
        memset(target_states, 0, sizeof target_states);
        memset(cmp_states, 0, sizeof cmp_states);
        if (clone_q8_from_f32_lab(draft_states, base, &mc, capacity) != 0 ||
            clone_f32_states_lab(target_states, base, &mc, capacity) != 0)
            return 1;

        draft[0] = argmax_lab(prefill_logits);
        if (kvl_expert_cache_init(&cache, &es, cache_bytes) != 0) return 2;
        const double d0 = lab_wall_s();
        for (int t = 0; t < K - 1; ++t) {
            if (draft_forward_q8_lab(&ts, &cache, draft_states, &emb, &fn, &lm,
                                     draft[t], prompt_n + t, masks[mi].mask,
                                     logits, x, r1, n2, y, n1, attn,
                                     router, bias, ids, weights, scratch, z) != 0)
                return 1;
            draft[t + 1] = argmax_lab(logits);
        }
        const double draft_s = lab_wall_s() - d0;
        const uint64_t draft_bytes = cache.bytes_read;
        const uint64_t draft_reads = cache.read_ops;
        kvl_expert_cache_close(&cache);

        if (kvl_expert_cache_init(&cache, &es, cache_bytes) != 0) return 2;
        const double v0 = lab_wall_s();
        if (exact_block_lab(&ts, &cache, target_states, &emb, &fn, &lm,
                            draft, K, prompt_n, target_next, logits) != 0)
            return 1;
        const double verify_s = lab_wall_s() - v0;

        int accept = 1;
        for (int t = 1; t < K; ++t) {
            if (draft[t] != target_next[t - 1]) break;
            ++accept;
        }
        const int correction = target_next[accept - 1];
        const int resolved = accept + 1;
        if (accept < K) {
            for (int l = 0; l < LN; ++l)
                if (kvl_mla_compressed_state_truncate(&target_states[l], prompt_n + accept) != 0)
                    return 1;
        }
        const double c0 = lab_wall_s();
        if (forward_token(&ts, &cache, target_states, &emb, &fn, &lm,
                          correction, prompt_n + accept, 1, logits,
                          x, r1, n2, y, router, bias, ids, weights, scratch, z) != 0)
            return 1;
        const double commit_s = lab_wall_s() - c0;
        const uint64_t target_bytes = cache.bytes_read;
        const uint64_t target_reads = cache.read_ops;
        kvl_expert_cache_close(&cache);

        int seq_ok = 1;
        for (int t = 0; t < accept; ++t)
            if (draft[t] != baseline_tokens[t]) seq_ok = 0;
        if (correction != baseline_tokens[accept]) seq_ok = 0;

        if (clone_f32_states_lab(cmp_states, baseline_states, &mc, capacity) != 0)
            return 1;
        for (int l = 0; l < LN; ++l)
            if (kvl_mla_compressed_state_truncate(&cmp_states[l], prompt_n + resolved) != 0)
                return 1;
        const double commit_state_max = f32_state_max_abs_lab(target_states, cmp_states);

        const double cycle_s = draft_s + verify_s + commit_s;
        const double base_s = baseline_wall[resolved];
        const double speedup = cycle_s > 0.0 ? base_s / cycle_s : 0.0;
        const double tok_s = cycle_s > 0.0 ? (double)resolved / cycle_s : 0.0;
        const double acc = (double)accept / (double)K;
        if (!masks[mi].focused) {
            printf("SELF_SPEC mask=%s skipped=%d acceptance=%d/%d(%.3f) resolved=%d "
                   "draft_s=%.3f verify_s=%.3f commit_s=%.3f cycle_s=%.3f "
                   "baseline_s=%.3f speedup=%.3fx resolved_tok_s=%.4f "
                   "draft_bytes_mib=%.2f target_bytes_mib=%.2f baseline_bytes_mib=%.2f "
                   "draft_reads=%" PRIu64 " target_reads=%" PRIu64 " "
                   "seq_ok=%s commit_state_max=%.9g\n",
                   masks[mi].name, popcount_mask_lab(masks[mi].mask), accept, K, acc,
                   resolved, draft_s, verify_s, commit_s, cycle_s, base_s, speedup, tok_s,
                   draft_bytes / 1048576.0, target_bytes / 1048576.0,
                   baseline_bytes[resolved] / 1048576.0,
                   draft_reads, target_reads, seq_ok ? "yes" : "no", commit_state_max);
            printf("DRAFT_IDS mask=%s", masks[mi].name);
        } else {
            printf("FOCUS_SELF_SPEC mask=%s skipped=%d acceptance=%d/%d(%.3f) resolved=%d "
                   "draft_s=%.3f verify_s=%.3f commit_s=%.3f cycle_s=%.3f "
                   "baseline_s=%.3f speedup=%.3fx resolved_tok_s=%.4f "
                   "draft_bytes_mib=%.2f target_bytes_mib=%.2f baseline_bytes_mib=%.2f "
                   "draft_reads=%" PRIu64 " target_reads=%" PRIu64 " "
                   "focus_seq_ok=%s focus_commit_state_max=%.9g\n",
                   masks[mi].name, popcount_mask_lab(masks[mi].mask), accept, K, acc,
                   resolved, draft_s, verify_s, commit_s, cycle_s, base_s, speedup, tok_s,
                   draft_bytes / 1048576.0, target_bytes / 1048576.0,
                   baseline_bytes[resolved] / 1048576.0,
                   draft_reads, target_reads, seq_ok ? "yes" : "no", commit_state_max);
            printf("FOCUS_DRAFT_IDS mask=%s", masks[mi].name);
            if (seq_ok && commit_state_max == 0.0) focus_exact++;
            if (accept == K) focus_full_accept++;
        }
        for (int t = 0; t < K; ++t) printf(" %d", draft[t]);
        printf("\n");
        if (!seq_ok || commit_state_max != 0.0) failures++;

        free_q8_states_lab(draft_states);
        free_f32_states_lab(target_states);
        free_f32_states_lab(cmp_states);
    }

    printf("FOCUS_SUMMARY masks=6 exact_commit=%d full_accept=%d\n",
           focus_exact, focus_full_accept);

    free_f32_states_lab(base); free_f32_states_lab(baseline_states);
    kvl_trunk_tensor_free(&emb); kvl_trunk_tensor_free(&fn); kvl_trunk_tensor_free(&lm);
    kvl_expert_store_close(&es); kvl_trunk_store_close(&ts);
    free(prompt); free(prefill_logits); free(logits); free(x); free(r1); free(n2);
    free(y); free(n1); free(attn); free(z); free(router); free(bias); free(ids);
    free(weights); free(scratch); free(baseline_tokens); free(baseline_wall);
    free(baseline_bytes); free(draft); free(target_next);

    if (failures) {
        fprintf(stderr, "FAIL: self-spec target commit diverged from exact serial baseline\n");
        return 1;
    }
    puts("PASS: self-spec target correction/bonus commits remain exact");
    return 0;
}
