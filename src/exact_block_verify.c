/* Experimental exact multi-token target verifier for Draft & Verify.
 *
 * This translation unit deliberately includes the proven text generator so the lab
 * verifier reuses exactly the same constants, store access, MoE dispatch and prompt
 * prefill.  It is not wired into production generation yet.
 */
#define main kvl_generate_text_entry_unused
#include "generate.c"
#undef main

static int argmax_logits_lab(const float *logits) {
    int best = 0;
    for (int i = 1; i < V; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

static int clone_states_lab(KvlMlaCompressedState *dst,
                            const KvlMlaCompressedState *src,
                            const KvlMlaConfig *cfg,
                            int capacity) {
    for (int layer = 0; layer < LN; ++layer) {
        if (kvl_mla_compressed_state_init(&dst[layer], cfg, capacity) != 0)
            return -1;
        if (src[layer].len < 0 || src[layer].len > capacity ||
            src[layer].kv_lora_rank != R || src[layer].qk_rope_dim != DR)
            return -1;
        const size_t latent_n = (size_t)src[layer].len * R;
        const size_t rope_n = (size_t)src[layer].len * DR;
        memcpy(dst[layer].latent, src[layer].latent, latent_n * sizeof(float));
        memcpy(dst[layer].rope, src[layer].rope, rope_n * sizeof(float));
        dst[layer].len = src[layer].len;
    }
    return 0;
}

static void free_states_lab(KvlMlaCompressedState *states) {
    for (int layer = 0; layer < LN; ++layer)
        kvl_mla_compressed_state_free(&states[layer]);
}

static double state_max_abs_lab(const KvlMlaCompressedState *a,
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

static double vector_max_abs_lab(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* Evaluate `count` already-proposed tokens with the exact target model, but traverse
 * the model layer-major.  Each trunk/router/shared tensor is loaded once per layer for
 * the whole block instead of once per (token, layer).  Causal MLA appends the block in
 * token order inside each layer, so semantics match serial forward_token().
 *
 * next_ids[t] is the target argmax after consuming token_ids[t].  Thus for a draft
 * d[0..K-1], existing prefix logits verify d[0], next_ids[0] verifies d[1], and
 * next_ids[K-1] is the verifier's bonus prediction when all drafts are accepted.
 */
static int forward_block_exact_lab(KvlTrunkStore *ts, KvlExpertCache *cache,
                                   KvlMlaCompressedState *states,
                                   const KvlTrunkTensor *emb,
                                   const KvlTrunkTensor *final_norm,
                                   const KvlTrunkTensor *lm_head,
                                   const int *token_ids, int count,
                                   int start_position, int *next_ids,
                                   float *last_logits) {
    if (!ts || !cache || !states || !emb || !final_norm || !lm_head ||
        !token_ids || count <= 0 || start_position < 0 || !next_ids || !last_logits)
        return -1;
    for (int t = 0; t < count; ++t)
        if (token_ids[t] < 0 || token_ids[t] >= V) return -1;
    for (int layer = 0; layer < LN; ++layer)
        if (states[layer].len != start_position ||
            count > states[layer].capacity - start_position)
            return -1;

    const size_t elems = (size_t)count * H;
    float *x = (float *)malloc(elems * sizeof(float));
    float *work_a = (float *)malloc(elems * sizeof(float));
    float *work_b = (float *)malloc(elems * sizeof(float));
    float *router = (float *)malloc((size_t)E * H * sizeof(float));
    float *bias = (float *)malloc((size_t)E * sizeof(float));
    int *top_ids = (int *)malloc((size_t)TOPK * sizeof(int));
    float *top_weights = (float *)malloc((size_t)TOPK * sizeof(float));
    float *scratch = (float *)malloc((size_t)(3 * DENSE_I + H) * sizeof(float));
    float *z = (float *)malloc((size_t)H * sizeof(float));
    int rc = -1;
    if (!x || !work_a || !work_b || !router || !bias || !top_ids ||
        !top_weights || !scratch || !z)
        goto done;

    for (int t = 0; t < count; ++t)
        expand(x + (size_t)t * H,
               (const uint16_t *)emb->data + (size_t)token_ids[t] * H, H);

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
            kvl_rmsnorm_bf16(work_a + (size_t)t * H, x + (size_t)t * H,
                             (const uint16_t *)in.data, H, RMS_EPS);

        KvlMlaBF16 aw = {
            (const uint16_t *)q.data,
            (const uint16_t *)kva.data,
            (const uint16_t *)kvan.data,
            (const uint16_t *)kvb.data,
            (const uint16_t *)o.data
        };
        if (kvl_mla_decode_compressed_block_bf16(work_b, work_a, count,
                                                  start_position, &cfg, &aw,
                                                  &states[layer]) != 0) {
            kvl_trunk_tensor_free(&in); kvl_trunk_tensor_free(&q);
            kvl_trunk_tensor_free(&kva); kvl_trunk_tensor_free(&kvan);
            kvl_trunk_tensor_free(&kvb); kvl_trunk_tensor_free(&o);
            kvl_trunk_tensor_free(&pn);
            goto done;
        }

        for (int t = 0; t < count; ++t) {
            const size_t base = (size_t)t * H;
            for (int i = 0; i < H; ++i)
                work_b[base + i] = x[base + i] + work_b[base + i];
            kvl_rmsnorm_bf16(work_a + base, work_b + base,
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
            for (int t = 0; t < count; ++t) {
                const size_t base = (size_t)t * H;
                if (kvl_mlp_bf16(x + base, work_a + base, &dense, H, scratch) != 0) {
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
            for (int t = 0; t < count; ++t) {
                const size_t base = (size_t)t * H;
                if (kvl_moe_token_bf16(cache, layer, &router_cfg, work_a + base,
                                       router, bias, EXP_I, &shared, x + base,
                                       top_ids, top_weights, scratch) != 0) {
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

        for (size_t i = 0; i < elems; ++i)
            x[i] = work_b[i] + x[i];
    }

    for (int t = 0; t < count; ++t) {
        kvl_rmsnorm_bf16(z, x + (size_t)t * H,
                         (const uint16_t *)final_norm->data, H, RMS_EPS);
        kvl_matvec_bf16(last_logits, z, (const uint16_t *)lm_head->data, H, V);
        next_ids[t] = argmax_logits_lab(last_logits);
    }
    rc = 0;

done:
    free(x); free(work_a); free(work_b); free(router); free(bias);
    free(top_ids); free(top_weights); free(scratch); free(z);
    return rc;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr,
                "usage: %s trunk.bin trunk.idx experts.bin experts.idx prompt.ids "
                "cache_bytes block_tokens\n", argv[0]);
        return 2;
    }
    const size_t cache_bytes = (size_t)strtoull(argv[6], NULL, 10);
    const int block_tokens = atoi(argv[7]);
    if (cache_bytes == 0 || block_tokens < 2 || block_tokens > 16) return 2;

    int *prompt = NULL, prompt_n = 0;
    if (read_prompt_ids(argv[5], &prompt, &prompt_n) != 0) {
        fprintf(stderr, "failed to read prompt token ids\n");
        return 2;
    }
    const int capacity = prompt_n + block_tokens + 1;

    KvlTrunkStore ts;
    KvlExpertStore es;
    if (kvl_trunk_store_open(&ts, argv[1], argv[2], 1) != 0 ||
        kvl_expert_store_open(&es, argv[3], argv[4], 1) != 0) {
        free(prompt);
        return 2;
    }

    KvlTrunkTensor emb = {0}, fn = {0}, lm = {0};
    if (load_kind(&ts, KVL_TRUNK_GLOBAL_LAYER, KVL_TENSOR_EMBED_TOKENS, &emb) ||
        load_kind(&ts, KVL_TRUNK_GLOBAL_LAYER, KVL_TENSOR_FINAL_NORM, &fn) ||
        load_kind(&ts, KVL_TRUNK_GLOBAL_LAYER, KVL_TENSOR_LM_HEAD, &lm))
        return 2;

    KvlMlaConfig mc = {H, NH, DN, DR, DV, R, RMS_EPS, ROPE_THETA};
    KvlMlaCompressedState serial_states[LN], block_states[LN];
    memset(serial_states, 0, sizeof serial_states);
    memset(block_states, 0, sizeof block_states);
    for (int layer = 0; layer < LN; ++layer)
        if (kvl_mla_compressed_state_init(&serial_states[layer], &mc, capacity) != 0)
            return 2;

    float *prefill_logits = (float *)malloc((size_t)V * sizeof(float));
    float *serial_logits = (float *)malloc((size_t)V * sizeof(float));
    float *block_logits = (float *)malloc((size_t)V * sizeof(float));
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
    int *draft = (int *)malloc((size_t)block_tokens * sizeof(int));
    int *serial_next = (int *)malloc((size_t)block_tokens * sizeof(int));
    int *block_next = (int *)malloc((size_t)block_tokens * sizeof(int));
    if (!prefill_logits || !serial_logits || !block_logits || !x || !r1 || !n2 ||
        !y || !z || !router || !bias || !top_ids || !top_weights || !scratch ||
        !draft || !serial_next || !block_next)
        return 2;

    KvlExpertCache cache;
    memset(&cache, 0, sizeof cache);
    if (kvl_expert_cache_init(&cache, &es, cache_bytes) != 0) return 2;
    if (prefill_prompt(&ts, &cache, serial_states, &emb, &fn, &lm,
                       prompt, prompt_n, prefill_logits,
                       router, bias, top_ids, top_weights, scratch, z) != 0)
        return 1;
    kvl_expert_cache_close(&cache);

    if (clone_states_lab(block_states, serial_states, &mc, capacity) != 0)
        return 1;
    memcpy(serial_logits, prefill_logits, (size_t)V * sizeof(float));

    if (kvl_expert_cache_init(&cache, &es, cache_bytes) != 0) return 2;
    for (int t = 0; t < block_tokens; ++t) {
        draft[t] = argmax_logits_lab(serial_logits);
        if (forward_token(&ts, &cache, serial_states, &emb, &fn, &lm,
                          draft[t], prompt_n + t, 1, serial_logits,
                          x, r1, n2, y, router, bias,
                          top_ids, top_weights, scratch, z) != 0)
            return 1;
        serial_next[t] = argmax_logits_lab(serial_logits);
    }
    fprintf(stderr, "SERIAL_CACHE ");
    kvl_expert_cache_report(&cache);
    kvl_expert_cache_close(&cache);

    if (kvl_expert_cache_init(&cache, &es, cache_bytes) != 0) return 2;
    if (forward_block_exact_lab(&ts, &cache, block_states, &emb, &fn, &lm,
                                draft, block_tokens, prompt_n,
                                block_next, block_logits) != 0)
        return 1;
    fprintf(stderr, "BLOCK_CACHE ");
    kvl_expert_cache_report(&cache);
    kvl_expert_cache_close(&cache);

    int token_mismatch = 0;
    for (int t = 0; t < block_tokens; ++t) {
        printf("VERIFY t=%d draft=%d serial_next=%d block_next=%d\n",
               t, draft[t], serial_next[t], block_next[t]);
        if (serial_next[t] != block_next[t]) token_mismatch = 1;
    }
    const double logits_max = vector_max_abs_lab(serial_logits, block_logits, V);
    const double state_max = state_max_abs_lab(serial_states, block_states);
    printf("FULL_BLOCK prompt=%d block=%d logits_max=%.9g state_max=%.9g "
           "trunk_direct_io=%s expert_direct_io=%s\n",
           prompt_n, block_tokens, logits_max, state_max,
           ts.direct_io ? "yes" : "no", es.direct_io ? "yes" : "no");

    const int accept = block_tokens - 1;
    for (int layer = 0; layer < LN; ++layer) {
        if (kvl_mla_compressed_state_truncate(&serial_states[layer], prompt_n + accept) != 0 ||
            kvl_mla_compressed_state_truncate(&block_states[layer], prompt_n + accept) != 0)
            return 1;
    }
    const double rollback_state_max = state_max_abs_lab(serial_states, block_states);
    printf("FULL_BLOCK_ROLLBACK accepted=%d state_max=%.9g\n",
           accept, rollback_state_max);

    free_states_lab(serial_states);
    free_states_lab(block_states);
    kvl_trunk_tensor_free(&emb); kvl_trunk_tensor_free(&fn); kvl_trunk_tensor_free(&lm);
    kvl_expert_store_close(&es); kvl_trunk_store_close(&ts);
    free(prompt); free(prefill_logits); free(serial_logits); free(block_logits);
    free(x); free(r1); free(n2); free(y); free(z); free(router); free(bias);
    free(top_ids); free(top_weights); free(scratch); free(draft); free(serial_next); free(block_next);

    if (token_mismatch || logits_max != 0.0 || state_max != 0.0 || rollback_state_max != 0.0) {
        fprintf(stderr, "FAIL: full-model block verifier diverged from serial target\n");
        return 1;
    }
    puts("PASS: full-model exact block verifier matches serial target");
    return 0;
}
