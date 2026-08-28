/* V9 multimodal generator.
 *
 * Reuse the already-proven V8 decoder implementation in this translation unit while exposing
 * a prompt-prefill path that can inject one distinct 2048-float projected image embedding at
 * every <|media_pad|> token position. Keeping this as a separate executable prevents the V9
 * correctness bridge from regressing the faster layer-major text-only prefill path.
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

static int forward_input_v9(KvlTrunkStore *ts, KvlExpertCache *cache,
                            KvlMlaCompressedState *states,
                            const KvlTrunkTensor *final_norm,
                            const KvlTrunkTensor *lm_head,
                            const float *input, int position, int need_logits,
                            float *logits, float *x, float *r1, float *n2, float *y,
                            float *router, float *bias, int *ids, float *weights,
                            float *scratch, float *z) {
    memcpy(x, input, (size_t)H * sizeof(float));
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
    if (read_prompt_ids(argv[5], &prompt, &prompt_n) != 0) {
        fprintf(stderr, "failed to read prompt token ids\n");
        return 2;
    }
    int media_n = 0;
    for (int i = 0; i < prompt_n; ++i) if (prompt[i] == MEDIA_PAD_ID_V9) ++media_n;
    if (media_n <= 0) {
        fprintf(stderr, "multimodal prompt contains no media_pad token id %d\n", MEDIA_PAD_ID_V9);
        free(prompt);
        return 2;
    }
    float *media = NULL;
    if (read_media_f32(argv[6], &media, media_n) != 0) {
        fprintf(stderr, "failed to read %d projected media embeddings\n", media_n);
        free(prompt);
        return 2;
    }

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
    float *input = (float *)malloc((size_t)H * sizeof(float));
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
    if (!logits || !input || !x || !r1 || !n2 || !y || !z || !router || !bias ||
        !top_ids || !top_weights || !scratch) return 2;

    fprintf(stderr,
            "kvl_generate_vl: prompt=%d media=%d max_new=%d temperature=%.4g "
            "state=%.2f MiB cache=%.2f MiB prefill=token-major-media\n",
            prompt_n, media_n, max_new, temperature, state_bytes / 1048576.0,
            cache_bytes / 1048576.0);

    int mi = 0;
    for (int position = 0; position < prompt_n; ++position) {
        if (prompt[position] == MEDIA_PAD_ID_V9) {
            memcpy(input, media + (size_t)mi * H, (size_t)H * sizeof(float));
            ++mi;
        } else {
            const int token = prompt[position];
            if (token < 0 || token >= V) return 2;
            expand(input, (const uint16_t *)emb.data + (size_t)token * H, H);
        }
        if (forward_input_v9(&ts, &cache, states, &fn, &lm, input, position,
                             position == prompt_n - 1, logits, x, r1, n2, y,
                             router, bias, top_ids, top_weights, scratch, z) != 0)
            return 1;
    }
    if (mi != media_n) return 1;

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
    free(prompt); free(media); free(logits); free(input); free(x); free(r1); free(n2);
    free(y); free(z); free(router); free(bias); free(top_ids); free(top_weights); free(scratch);
    kvl_expert_cache_close(&cache); kvl_expert_store_close(&es); kvl_trunk_store_close(&ts);
    return 0;
}
