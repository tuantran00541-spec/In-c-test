/* V9 multimodal generator.
 *
 * Reuse the proven V8 decoder implementation, but initialize the prompt hidden-state matrix
 * from a mixture of BF16 token embeddings and one projected image embedding per media-pad.
 * The complete multimodal prompt then follows the same layer-major batch prefill as text-only
 * V8, so each trunk tensor is loaded once per layer rather than once per token.
 */
#define main kvl_generate_text_entry_unused
#include "generate.c"
#undef main

static const int MEDIA_PAD_ID_V9 = 163605;

static int read_media_f32(const char *path, float **out, int count) {
    if (!path || !out || count <= 0) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    const size_t n = (size_t)count * H;
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fclose(f); return -1; }
    const size_t got = fread(p, sizeof(float), n, f);
    const int extra = fgetc(f);
    fclose(f);
    if (got != n || extra != EOF) { free(p); return -1; }
    *out = p;
    return 0;
}

/* Released Kimi-VL uses BF16 image features and casts text embeddings to that dtype before
 * replacing media-pad positions. Keep FP32 runtime activations, but round projector output to
 * an IEEE BF16 value at exactly that multimodal boundary. */
static float v9_round_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    const uint32_t exp = u & UINT32_C(0x7f800000);
    if (exp != UINT32_C(0x7f800000))
        u += UINT32_C(0x00007fff) + ((u >> 16) & 1u);
    u &= UINT32_C(0xffff0000);
    memcpy(&x, &u, sizeof x);
    return x;
}

static int prefill_multimodal_v9(KvlTrunkStore *ts, KvlExpertCache *cache,
                                 KvlMlaCompressedState *states,
                                 const KvlTrunkTensor *emb,
                                 const KvlTrunkTensor *final_norm,
                                 const KvlTrunkTensor *lm_head,
                                 const int *prompt, int seq_len,
                                 const float *media, int media_n,
                                 float *logits, float *router, float *bias,
                                 int *ids, float *weights, float *scratch, float *z) {
    if (!ts || !cache || !states || !emb || !final_norm || !lm_head ||
        !prompt || seq_len <= 0 || !media || media_n <= 0 || !logits ||
        !router || !bias || !ids || !weights || !scratch || !z)
        return -1;

    const size_t elems = (size_t)seq_len * H;
    float *x = (float *)malloc(elems * sizeof(float));
    float *work_a = (float *)malloc(elems * sizeof(float));
    float *work_b = (float *)malloc(elems * sizeof(float));
    int rc = -1;
    if (!x || !work_a || !work_b) goto done;

    int mi = 0;
    for (int t = 0; t < seq_len; ++t) {
        float *dst = x + (size_t)t * H;
        if (prompt[t] == MEDIA_PAD_ID_V9) {
            if (mi >= media_n) goto done;
            const float *src = media + (size_t)mi * H;
            for (int i = 0; i < H; ++i) dst[i] = v9_round_bf16(src[i]);
            ++mi;
        } else {
            if (prompt[t] < 0 || prompt[t] >= V) goto done;
            expand(dst, (const uint16_t *)emb->data + (size_t)prompt[t] * H, H);
        }
    }
    if (mi != media_n) goto done;

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
            kvl_rmsnorm_bf16(work_a + (size_t)t * H, x + (size_t)t * H,
                             (const uint16_t *)in.data, H, RMS_EPS);

        KvlMlaBF16 aw = {
            (const uint16_t *)q.data,
            (const uint16_t *)kva.data,
            (const uint16_t *)kvan.data,
            (const uint16_t *)kvb.data,
            (const uint16_t *)o.data
        };
        if (kvl_mla_prefill_bf16(work_b, work_a, seq_len, &cfg, &aw) != 0 ||
            kvl_mla_compressed_state_prefill_bf16(work_a, seq_len, &cfg, &aw,
                                                   &states[layer]) != 0) {
            kvl_trunk_tensor_free(&in); kvl_trunk_tensor_free(&q);
            kvl_trunk_tensor_free(&kva); kvl_trunk_tensor_free(&kvan);
            kvl_trunk_tensor_free(&kvb); kvl_trunk_tensor_free(&o);
            kvl_trunk_tensor_free(&pn);
            goto done;
        }

        for (int t = 0; t < seq_len; ++t) {
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
            for (int t = 0; t < seq_len; ++t) {
                const size_t base = (size_t)t * H;
                if (kvl_mlp_bf16(x + base, work_a + base, &dense, H, scratch) != 0) {
                    kvl_trunk_tensor_free(&g); kvl_trunk_tensor_free(&u);
                    kvl_trunk_tensor_free(&d); goto done;
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
                kvl_trunk_tensor_free(&sd); goto done;
            }
            expand(router, (const uint16_t *)rt.data, (size_t)E * H);
            expand(bias, (const uint16_t *)rb.data, E);
            KvlMlpBF16 shared = {
                (const uint16_t *)sg.data, (const uint16_t *)su.data,
                (const uint16_t *)sd.data, SHARED_I
            };
            for (int t = 0; t < seq_len; ++t) {
                const size_t base = (size_t)t * H;
                if (kvl_moe_token_bf16(cache, layer, &router_cfg, work_a + base,
                                       router, bias, EXP_I, &shared, x + base,
                                       ids, weights, scratch) != 0) {
                    kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
                    kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
                    kvl_trunk_tensor_free(&sd); goto done;
                }
            }
            kvl_trunk_tensor_free(&rt); kvl_trunk_tensor_free(&rb);
            kvl_trunk_tensor_free(&sg); kvl_trunk_tensor_free(&su);
            kvl_trunk_tensor_free(&sd);
        }

        for (size_t i = 0; i < elems; ++i) x[i] = work_b[i] + x[i];
    }

    kvl_rmsnorm_bf16(z, x + (size_t)(seq_len - 1) * H,
                     (const uint16_t *)final_norm->data, H, RMS_EPS);
    kvl_matvec_bf16(logits, z, (const uint16_t *)lm_head->data, H, V);
    rc = 0;

done:
    free(x); free(work_a); free(work_b);
    return rc;
}

int main(int argc, char **argv) {
    if (argc != 11) {
        fprintf(stderr,
                "usage: %s trunk.bin trunk.idx experts.bin experts.idx prompt.ids media.f32 "
                "cache_bytes max_new temperature seed\n", argv[0]);
        return 2;
    }

    const size_t cache_bytes = (size_t)strtoull(argv[7], NULL, 10);
    const int max_new = atoi(argv[8]);
    const float temperature = strtof(argv[9], NULL);
    uint64_t rng = (uint64_t)strtoull(argv[10], NULL, 10);
    if (max_new < 1 || temperature < 0.0f) return 2;
    if (rng == 0) rng = UINT64_C(0x9e3779b97f4a7c15);

    int *prompt = NULL, prompt_n = 0;
    if (read_prompt_ids(argv[5], &prompt, &prompt_n) != 0) return 2;
    int media_n = 0;
    for (int i = 0; i < prompt_n; ++i) if (prompt[i] == MEDIA_PAD_ID_V9) ++media_n;
    if (media_n <= 0) { free(prompt); return 2; }
    float *media = NULL;
    if (read_media_f32(argv[6], &media, media_n) != 0) { free(prompt); return 2; }

    const int capacity = prompt_n + max_new;
    KvlTrunkStore ts;
    KvlExpertStore es;
    KvlExpertCache cache;
    if (kvl_trunk_store_open(&ts, argv[1], argv[2], 1) != 0 ||
        kvl_expert_store_open(&es, argv[3], argv[4], 1) != 0 ||
        kvl_expert_cache_init(&cache, &es, cache_bytes) != 0) {
        free(prompt); free(media); return 2;
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

    const double batch_mib = (double)((size_t)3 * (size_t)prompt_n * H * sizeof(float)) / 1048576.0;
    fprintf(stderr,
            "kvl_generate_vl: prompt=%d media=%d max_new=%d temperature=%.4g "
            "state=%.2f MiB cache=%.2f MiB prefill=batch-layer-major-media "
            "buffers=3 batch_ws=%.2f MiB media_boundary=bf16\n",
            prompt_n, media_n, max_new, temperature, state_bytes / 1048576.0,
            cache_bytes / 1048576.0, batch_mib);

    if (prefill_multimodal_v9(&ts, &cache, states, &emb, &fn, &lm,
                              prompt, prompt_n, media, media_n, logits,
                              router, bias, top_ids, top_weights, scratch, z) != 0)
        return 1;

    int generated = 0;
    int position = prompt_n;
    while (generated < max_new) {
        const int token = sample_token(logits, temperature, &rng);
        printf("TOKEN %d\n", token);
        fflush(stdout);
        ++generated;
        if (token == EOS_ID || token == IM_END_ID || generated == max_new) break;
        if (forward_token(&ts, &cache, states, &emb, &fn, &lm, token, position,
                          1, logits, x, r1, n2, y, router, bias,
                          top_ids, top_weights, scratch, z) != 0) return 1;
        ++position;
    }

    fprintf(stderr,
            "kvl_generate_vl: generated=%d context=%d trunk_direct_io=%s expert_direct_io=%s\n",
            generated, position, ts.direct_io ? "yes" : "no", es.direct_io ? "yes" : "no");
    kvl_expert_cache_report(&cache);

    for (int layer = 0; layer < LN; ++layer) kvl_mla_compressed_state_free(&states[layer]);
    kvl_trunk_tensor_free(&emb); kvl_trunk_tensor_free(&fn); kvl_trunk_tensor_free(&lm);
    free(prompt); free(media); free(logits); free(x); free(r1); free(n2); free(y); free(z);
    free(router); free(bias); free(top_ids); free(top_weights); free(scratch);
    kvl_expert_cache_close(&cache); kvl_expert_store_close(&es); kvl_trunk_store_close(&ts);
    return 0;
}
