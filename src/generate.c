#include "kvl/mla_compressed_state.h"
#include "kvl/ops.h"
#include "kvl/trunk_store.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    H = 2048, NH = 16, DN = 128, DR = 64, DV = 128, R = 512,
    LN = 27, V = 163840, E = 64, TOPK = 6,
    DENSE_I = 11264, EXP_I = 1408, SHARED_I = 2816
};

static const float RMS_EPS = 1.0e-5f;
static const float ROPE_THETA = 800000.0f;
static const int EOS_ID = 163585;
static const int IM_END_ID = 163586;

static int load_kind(KvlTrunkStore *ts, uint32_t layer, uint32_t kind, KvlTrunkTensor *t) {
    if (kvl_trunk_load(ts, layer, kind, t) != 0) {
        fprintf(stderr, "trunk load failed layer=%u kind=%u\n", layer, kind);
        return -1;
    }
    return 0;
}

static void expand(float *dst, const uint16_t *src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = kvl_bf16_to_f32(src[i]);
}

static int attention_token(KvlTrunkStore *ts, int layer, const float *x, int position,
                           KvlMlaCompressedState *state, float *r1, float *n2) {
    KvlTrunkTensor in = {0}, q = {0}, kva = {0}, kvan = {0}, kvb = {0}, o = {0}, pn = {0};
    float *n1 = NULL, *attn = NULL;
    int rc = -1;

    if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_INPUT_NORM, &in) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_Q_PROJ, &q) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_KV_A_PROJ, &kva) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_KV_A_NORM, &kvan) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_KV_B_PROJ, &kvb) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_O_PROJ, &o) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_POST_ATTN_NORM, &pn)) goto done;

    n1 = (float *)malloc((size_t)H * sizeof(float));
    attn = (float *)malloc((size_t)H * sizeof(float));
    if (!n1 || !attn) goto done;

    kvl_rmsnorm_bf16(n1, x, (const uint16_t *)in.data, H, RMS_EPS);
    KvlMlaConfig cfg = {H, NH, DN, DR, DV, R, RMS_EPS, ROPE_THETA};
    KvlMlaBF16 w = {
        (const uint16_t *)q.data,
        (const uint16_t *)kva.data,
        (const uint16_t *)kvan.data,
        (const uint16_t *)kvb.data,
        (const uint16_t *)o.data
    };
    if (kvl_mla_decode_compressed_bf16(attn, n1, position, &cfg, &w, state) != 0) goto done;

    for (int i = 0; i < H; ++i) r1[i] = x[i] + attn[i];
    kvl_rmsnorm_bf16(n2, r1, (const uint16_t *)pn.data, H, RMS_EPS);
    rc = 0;

done:
    free(n1);
    free(attn);
    kvl_trunk_tensor_free(&in);
    kvl_trunk_tensor_free(&q);
    kvl_trunk_tensor_free(&kva);
    kvl_trunk_tensor_free(&kvan);
    kvl_trunk_tensor_free(&kvb);
    kvl_trunk_tensor_free(&o);
    kvl_trunk_tensor_free(&pn);
    return rc;
}

static int mlp_token(KvlTrunkStore *ts, KvlExpertCache *cache, int layer,
                     const float *n, float *y, float *router, float *bias,
                     int *ids, float *weights, float *scratch) {
    if (layer == 0) {
        KvlTrunkTensor g = {0}, u = {0}, d = {0};
        int rc = -1;
        if (load_kind(ts, 0, KVL_TENSOR_DENSE_GATE, &g) ||
            load_kind(ts, 0, KVL_TENSOR_DENSE_UP, &u) ||
            load_kind(ts, 0, KVL_TENSOR_DENSE_DOWN, &d)) goto done_dense;
        KvlMlpBF16 dense = {
            (const uint16_t *)g.data, (const uint16_t *)u.data,
            (const uint16_t *)d.data, DENSE_I
        };
        rc = kvl_mlp_bf16(y, n, &dense, H, scratch);
    done_dense:
        kvl_trunk_tensor_free(&g);
        kvl_trunk_tensor_free(&u);
        kvl_trunk_tensor_free(&d);
        return rc;
    }

    KvlTrunkTensor rt = {0}, rb = {0}, sg = {0}, su = {0}, sd = {0};
    int rc = -1;
    if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_WEIGHT, &rt) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_BIAS, &rb) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_GATE, &sg) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_UP, &su) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_SHARED_DOWN, &sd)) goto done_moe;

    expand(router, (const uint16_t *)rt.data, (size_t)E * H);
    expand(bias, (const uint16_t *)rb.data, E);
    KvlRouterConfig r = {H, E, TOPK, 1, 1, 1, 2.446f};
    KvlMlpBF16 shared = {
        (const uint16_t *)sg.data, (const uint16_t *)su.data,
        (const uint16_t *)sd.data, SHARED_I
    };
    rc = kvl_moe_token_bf16(cache, layer, &r, n, router, bias, EXP_I,
                            &shared, y, ids, weights, scratch);

done_moe:
    kvl_trunk_tensor_free(&rt);
    kvl_trunk_tensor_free(&rb);
    kvl_trunk_tensor_free(&sg);
    kvl_trunk_tensor_free(&su);
    kvl_trunk_tensor_free(&sd);
    return rc;
}

static int forward_token(KvlTrunkStore *ts, KvlExpertCache *cache,
                         KvlMlaCompressedState *states, const KvlTrunkTensor *emb,
                         const KvlTrunkTensor *final_norm, const KvlTrunkTensor *lm_head,
                         int token_id, int position, int need_logits, float *logits,
                         float *x, float *r1, float *n2, float *y,
                         float *router, float *bias, int *ids, float *weights,
                         float *scratch, float *z) {
    if (token_id < 0 || token_id >= V) return -1;
    expand(x, (const uint16_t *)emb->data + (size_t)token_id * H, H);

    for (int layer = 0; layer < LN; ++layer) {
        if (attention_token(ts, layer, x, position, &states[layer], r1, n2) != 0) return -1;
        if (mlp_token(ts, cache, layer, n2, y, router, bias, ids, weights, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) x[i] = r1[i] + y[i];
    }

    if (need_logits) {
        kvl_rmsnorm_bf16(z, x, (const uint16_t *)final_norm->data, H, RMS_EPS);
        kvl_matvec_bf16(logits, z, (const uint16_t *)lm_head->data, H, V);
    }
    return 0;
}

/* V8 prompt prefill is layer-major rather than token-major. Every trunk tensor is loaded
 * once per layer for the complete prompt, causal MLA runs as a batch, and routed experts are
 * evaluated token-by-token while staying within the same layer so the LRU can reuse recurring
 * experts. The resulting compressed MLA histories are then used by the unchanged V6 decode
 * path for newly generated tokens. */
static int prefill_prompt(KvlTrunkStore *ts, KvlExpertCache *cache,
                          KvlMlaCompressedState *states,
                          const KvlTrunkTensor *emb,
                          const KvlTrunkTensor *final_norm,
                          const KvlTrunkTensor *lm_head,
                          const int *prompt, int seq_len, float *logits,
                          float *router, float *bias, int *ids, float *weights,
                          float *scratch, float *z) {
    if (!ts || !cache || !states || !emb || !final_norm || !lm_head ||
        !prompt || seq_len <= 0 || !logits || !router || !bias || !ids ||
        !weights || !scratch || !z)
        return -1;

    const size_t elems = (size_t)seq_len * H;
    float *x = (float *)malloc(elems * sizeof(float));
    float *n1 = (float *)malloc(elems * sizeof(float));
    float *attn = (float *)malloc(elems * sizeof(float));
    float *r1 = (float *)malloc(elems * sizeof(float));
    float *n2 = (float *)malloc(elems * sizeof(float));
    float *y = (float *)malloc(elems * sizeof(float));
    int rc = -1;
    if (!x || !n1 || !attn || !r1 || !n2 || !y) goto done;

    for (int t = 0; t < seq_len; ++t) {
        if (prompt[t] < 0 || prompt[t] >= V) goto done;
        expand(x + (size_t)t * H,
               (const uint16_t *)emb->data + (size_t)prompt[t] * H, H);
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

        for (int t = 0; t < seq_len; ++t)
            kvl_rmsnorm_bf16(n1 + (size_t)t * H, x + (size_t)t * H,
                              (const uint16_t *)in.data, H, RMS_EPS);

        KvlMlaBF16 aw = {
            (const uint16_t *)q.data,
            (const uint16_t *)kva.data,
            (const uint16_t *)kvan.data,
            (const uint16_t *)kvb.data,
            (const uint16_t *)o.data
        };
        if (kvl_mla_prefill_bf16(attn, n1, seq_len, &cfg, &aw) != 0 ||
            kvl_mla_compressed_state_prefill_bf16(n1, seq_len, &cfg, &aw,
                                                   &states[layer]) != 0) {
            kvl_trunk_tensor_free(&in); kvl_trunk_tensor_free(&q);
            kvl_trunk_tensor_free(&kva); kvl_trunk_tensor_free(&kvan);
            kvl_trunk_tensor_free(&kvb); kvl_trunk_tensor_free(&o);
            kvl_trunk_tensor_free(&pn);
            goto done;
        }

        for (int t = 0; t < seq_len; ++t) {
            const size_t base = (size_t)t * H;
            for (int i = 0; i < H; ++i) r1[base + i] = x[base + i] + attn[base + i];
            kvl_rmsnorm_bf16(n2 + base, r1 + base,
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
                kvl_trunk_tensor_free(&d);
                goto done;
            }
            KvlMlpBF16 dense = {
                (const uint16_t *)g.data, (const uint16_t *)u.data,
                (const uint16_t *)d.data, DENSE_I
            };
            for (int t = 0; t < seq_len; ++t) {
                const size_t base = (size_t)t * H;
                if (kvl_mlp_bf16(y + base, n2 + base, &dense, H, scratch) != 0) {
                    kvl_trunk_tensor_free(&g); kvl_trunk_tensor_free(&u);
                    kvl_trunk_tensor_free(&d);
                    goto done;
                }
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
                kvl_trunk_tensor_free(&sd);
                goto done;
            }
            expand(router, (const uint16_t *)rt.data, (size_t)E * H);
            expand(bias, (const uint16_t *)rb.data, E);
            KvlMlpBF16 shared = {
                (const uint16_t *)sg.data, (const uint16_t *)su.data,
                (const uint16_t *)sd.data, SHARED_I
            };
            for (int t = 0; t < seq_len; ++t) {
                const size_t base = (size_t)t * H;
                if (kvl_moe_token_bf16(cache, layer, &router_cfg, n2 + base,
                                       router, bias, EXP_I, &shared, y + base,
                                       ids, weights, scratch) != 0) {
                    kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
                    kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
                    kvl_trunk_tensor_free(&sd);
                    goto done;
                }
            }
            kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
            kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
            kvl_trunk_tensor_free(&sd);
        }

        for (size_t i = 0; i < elems; ++i) x[i] = r1[i] + y[i];
    }

    kvl_rmsnorm_bf16(z, x + (size_t)(seq_len - 1) * H,
                      (const uint16_t *)final_norm->data, H, RMS_EPS);
    kvl_matvec_bf16(logits, z, (const uint16_t *)lm_head->data, H, V);
    rc = 0;

done:
    free(x); free(n1); free(attn); free(r1); free(n2); free(y);
    return rc;
}

static uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * UINT64_C(2685821657736338717);
}

static double rng_unit(uint64_t *s) {
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

static int sample_token(const float *logits, float temperature, uint64_t *rng) {
    int argmax = 0;
    for (int i = 1; i < V; ++i) if (logits[i] > logits[argmax]) argmax = i;
    if (temperature <= 0.0f) return argmax;

    float maxv = logits[argmax];
    double sum = 0.0;
    for (int i = 0; i < V; ++i) sum += exp(((double)logits[i] - maxv) / temperature);
    double target = rng_unit(rng) * sum;
    double acc = 0.0;
    for (int i = 0; i < V; ++i) {
        acc += exp(((double)logits[i] - maxv) / temperature);
        if (acc >= target) return i;
    }
    return argmax;
}

static int read_prompt_ids(const char *path, int **out_ids, int *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int cap = 64, n = 0;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { fclose(f); return -1; }
    for (;;) {
        int id;
        int r = fscanf(f, "%d", &id);
        if (r == EOF) break;
        if (r != 1 || id < 0 || id >= V) { free(ids); fclose(f); return -1; }
        if (n == cap) {
            cap *= 2;
            int *p = (int *)realloc(ids, (size_t)cap * sizeof(int));
            if (!p) { free(ids); fclose(f); return -1; }
            ids = p;
        }
        ids[n++] = id;
    }
    fclose(f);
    if (n == 0) { free(ids); return -1; }
    *out_ids = ids;
    *out_n = n;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 10) {
        fprintf(stderr,
                "usage: %s trunk.bin trunk.idx experts.bin experts.idx prompt.ids "
                "cache_bytes max_new temperature seed\n", argv[0]);
        return 2;
    }

    const size_t cache_bytes = (size_t)strtoull(argv[6], NULL, 10);
    const int max_new = atoi(argv[7]);
    const float temperature = strtof(argv[8], NULL);
    uint64_t rng = (uint64_t)strtoull(argv[9], NULL, 10);
    if (max_new < 1 || temperature < 0.0f) return 2;
    if (rng == 0) rng = UINT64_C(0x9e3779b97f4a7c15);

    int *prompt = NULL, prompt_n = 0;
    if (read_prompt_ids(argv[5], &prompt, &prompt_n) != 0) {
        fprintf(stderr, "failed to read prompt token ids\n");
        return 2;
    }
    const int capacity = prompt_n + max_new;

    KvlTrunkStore ts;
    KvlExpertStore es;
    KvlExpertCache cache;
    if (kvl_trunk_store_open(&ts, argv[1], argv[2], 1) != 0 ||
        kvl_expert_store_open(&es, argv[3], argv[4], 1) != 0 ||
        kvl_expert_cache_init(&cache, &es, cache_bytes) != 0) {
        free(prompt);
        return 2;
    }

    KvlTrunkTensor emb = {0}, fn = {0}, lm = {0};
    if (load_kind(&ts, KVL_TRUNK_GLOBAL_LAYER, KVL_TENSOR_EMBED_TOKENS, &emb) ||
        load_kind(&ts, KVL_TRUNK_GLOBAL_LAYER, KVL_TENSOR_FINAL_NORM, &fn) ||
        load_kind(&ts, KVL_TRUNK_GLOBAL_LAYER, KVL_TENSOR_LM_HEAD, &lm)) return 2;

    KvlMlaConfig mc = {H, NH, DN, DR, DV, R, RMS_EPS, ROPE_THETA};
    KvlMlaCompressedState states[LN];
    memset(states, 0, sizeof states);
    size_t state_bytes = 0;
    for (int layer = 0; layer < LN; ++layer) {
        if (kvl_mla_compressed_state_init(&states[layer], &mc, capacity) != 0) return 2;
        state_bytes += kvl_mla_compressed_state_bytes(&states[layer]);
    }

    float *logits = (float *)malloc((size_t)V * sizeof(float));
    float *x = (float *)malloc((size_t)H * sizeof(float));
    float *r1 = (float *)malloc((size_t)H * sizeof(float));
    float *n2 = (float *)malloc((size_t)H * sizeof(float));
    float *y = (float *)malloc((size_t)H * sizeof(float));
    float *z = (float *)malloc((size_t)H * sizeof(float));
    float *router = (float *)malloc((size_t)E * H * sizeof(float));
    float *bias = (float *)malloc((size_t)E * sizeof(float));
    int *top_ids = (int *)malloc((size_t)TOPK * sizeof(int));
    float *top_weights = (float *)malloc((size_t)TOPK * sizeof(float));
    float *scratch = (float *)malloc((size_t)(3 * DENSE_I + H) * sizeof(float));
    if (!logits || !x || !r1 || !n2 || !y || !z || !router || !bias ||
        !top_ids || !top_weights || !scratch) return 2;

    const double batch_mib = (double)((size_t)6 * (size_t)prompt_n * H * sizeof(float)) / 1048576.0;
    fprintf(stderr,
            "kvl_generate: prompt=%d max_new=%d temperature=%.4g rms_eps=1e-5 "
            "state=%.2f MiB cache=%.2f MiB prefill=batch layer-major batch_ws=%.2f MiB\n",
            prompt_n, max_new, temperature, state_bytes / 1048576.0,
            cache_bytes / 1048576.0, batch_mib);

    if (prefill_prompt(&ts, &cache, states, &emb, &fn, &lm, prompt, prompt_n,
                       logits, router, bias, top_ids, top_weights, scratch, z) != 0)
        return 1;

    int generated = 0;
    int position = prompt_n;
    while (generated < max_new) {
        int token = sample_token(logits, temperature, &rng);
        printf("TOKEN %d\n", token);
        fflush(stdout);
        ++generated;
        if (token == EOS_ID || token == IM_END_ID) break;
        if (generated == max_new) break;
        if (forward_token(&ts, &cache, states, &emb, &fn, &lm, token, position,
                          1, logits, x, r1, n2, y, router, bias,
                          top_ids, top_weights, scratch, z) != 0) return 1;
        ++position;
    }

    fprintf(stderr, "kvl_generate: generated=%d context=%d trunk_direct_io=%s expert_direct_io=%s\n",
            generated, position, ts.direct_io ? "yes" : "no", es.direct_io ? "yes" : "no");
    kvl_expert_cache_report(&cache);

    for (int layer = 0; layer < LN; ++layer) kvl_mla_compressed_state_free(&states[layer]);
    kvl_trunk_tensor_free(&emb);
    kvl_trunk_tensor_free(&fn);
    kvl_trunk_tensor_free(&lm);
    free(prompt);
    free(logits); free(x); free(r1); free(n2); free(y); free(z);
    free(router); free(bias); free(top_ids); free(top_weights); free(scratch);
    kvl_expert_cache_close(&cache);
    kvl_expert_store_close(&es);
    kvl_trunk_store_close(&ts);
    return 0;
}
