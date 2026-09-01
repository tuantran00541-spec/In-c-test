#include "kvl/ops.h"
#include "kvl/format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int kvl_moe_token_q8_expert_parallel_auto(KvlExpertCache *cache, int layer,
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

enum { H = 32, I = 20, E = 64, K = 6, LAYER = 1 };

static size_t align_up(size_t n, size_t a) {
    return (n + a - 1u) / a * a;
}

static void fill_q8(void *dst, int out, int in, int seed) {
    float *scales = (float *)dst;
    signed char *q = (signed char *)(scales + out);
    for (int o = 0; o < out; ++o) {
        scales[o] = 0.001f * (float)(1 + ((seed + o * 7) % 23));
        for (int i = 0; i < in; ++i) {
            int v = ((seed * 17 + o * 13 + i * 5) % 251) - 125;
            if (v == 0) v = 1;
            q[(size_t)o * (size_t)in + (size_t)i] = (signed char)v;
        }
    }
}

static int write_store(const char *bin_path, const char *idx_path) {
    const size_t gate_bytes = (size_t)I * sizeof(float) + (size_t)I * H;
    const size_t up_bytes = gate_bytes;
    const size_t down_bytes = (size_t)H * sizeof(float) + (size_t)H * I;
    const size_t payload = gate_bytes + up_bytes + down_bytes;
    const size_t record_bytes = align_up(payload, KVL_EXPERT_ALIGN);

    FILE *bin = fopen(bin_path, "wb");
    if (!bin) return -1;
    unsigned char *record = (unsigned char *)calloc(record_bytes, 1);
    if (!record) { fclose(bin); return -1; }

    KvlExpertRecord recs[E];
    memset(recs, 0, sizeof recs);
    for (int e = 0; e < E; ++e) {
        memset(record, 0, record_bytes);
        const size_t gate_off = 0;
        const size_t up_off = gate_off + gate_bytes;
        const size_t down_off = up_off + up_bytes;
        fill_q8(record + gate_off, I, H, 100 + e);
        fill_q8(record + up_off, I, H, 500 + e);
        fill_q8(record + down_off, H, I, 900 + e);
        if (fwrite(record, 1, record_bytes, bin) != record_bytes) {
            free(record); fclose(bin); return -1;
        }
        recs[e].layer = LAYER;
        recs[e].expert = e;
        recs[e].file_offset = (uint64_t)e * record_bytes;
        recs[e].read_bytes = record_bytes;
        recs[e].payload_bytes = payload;
        recs[e].gate_off = gate_off;
        recs[e].gate_bytes = gate_bytes;
        recs[e].up_off = up_off;
        recs[e].up_bytes = up_bytes;
        recs[e].down_off = down_off;
        recs[e].down_bytes = down_bytes;
    }
    free(record);
    fclose(bin);

    KvlExpertIndexHeader hdr;
    memset(&hdr, 0, sizeof hdr);
    memcpy(hdr.magic, KVL_EXPERT_MAGIC, 8);
    hdr.version = KVL_EXPERT_VERSION;
    hdr.align = KVL_EXPERT_ALIGN;
    hdr.n_layers = 2;
    hdr.n_experts = E;
    hdr.n_records = E;
    hdr.dtype = KVL_DTYPE_Q8_ROW;
    hdr.records_offset = sizeof hdr;
    hdr.data_file_bytes = (uint64_t)E * record_bytes;

    FILE *idx = fopen(idx_path, "wb");
    if (!idx) return -1;
    if (fwrite(&hdr, 1, sizeof hdr, idx) != sizeof hdr ||
        fwrite(recs, sizeof recs[0], E, idx) != E) {
        fclose(idx); return -1;
    }
    fclose(idx);
    return 0;
}

int main(void) {
    const char *bin_path = "q8_parallel_probe_experts.bin";
    const char *idx_path = "q8_parallel_probe_experts.idx";
    if (write_store(bin_path, idx_path) != 0) {
        fprintf(stderr, "failed to write Q8 probe store\n");
        return 2;
    }

    KvlExpertStore store;
    if (kvl_expert_store_open(&store, bin_path, idx_path, 0) != 0) {
        fprintf(stderr, "failed to open Q8 probe store\n");
        return 2;
    }
    KvlExpertCache base_cache, par_cache;
    const size_t budget = (size_t)K * KVL_EXPERT_ALIGN;
    if (kvl_expert_cache_init(&base_cache, &store, budget) != 0 ||
        kvl_expert_cache_init(&par_cache, &store, budget) != 0) {
        fprintf(stderr, "cache init failed\n");
        return 2;
    }

    float x[H], router[E * H], bias[E];
    for (int i = 0; i < H; ++i) x[i] = ((i * 11) % 29 - 14) * 0.03125f;
    for (int e = 0; e < E; ++e) {
        bias[e] = ((e * 7) % 19 - 9) * 0.003f;
        for (int i = 0; i < H; ++i)
            router[(size_t)e * H + i] =
                ((e * 17 + i * 3) % 37 - 18) * 0.006f + (float)e * 0.0002f;
    }

    KvlRouterConfig rc = {H, E, K, 1, 1, 1, 2.446f};
    int base_ids[K], par_ids[K];
    float base_w[K], par_w[K], base_out[H], par_out[H];
    const size_t baseline_scratch_n = (size_t)3 * I + H;
    const size_t parallel_scratch_n = (size_t)K * ((size_t)2 * I + H);
    const size_t scratch_n = baseline_scratch_n > parallel_scratch_n
        ? baseline_scratch_n : parallel_scratch_n;
    float *base_scratch = (float *)calloc(scratch_n, sizeof(float));
    float *par_scratch = (float *)calloc(scratch_n, sizeof(float));
    if (!base_scratch || !par_scratch) return 2;

    if (kvl_moe_token_auto(&base_cache, LAYER, &rc, x, router, bias, I, NULL,
                           base_out, base_ids, base_w, base_scratch) != 0 ||
        kvl_moe_token_q8_expert_parallel_auto(&par_cache, LAYER, &rc, x, router, bias,
                                              I, NULL, par_out, par_ids, par_w,
                                              par_scratch) != 0) {
        fprintf(stderr, "MoE probe forward failed\n");
        return 1;
    }

    const int ids_exact = memcmp(base_ids, par_ids, sizeof base_ids) == 0;
    const int weights_exact = memcmp(base_w, par_w, sizeof base_w) == 0;
    const int out_exact = memcmp(base_out, par_out, sizeof base_out) == 0;
    printf("ids_exact=%s weights_exact=%s out_exact=%s\n",
           ids_exact ? "yes" : "no",
           weights_exact ? "yes" : "no",
           out_exact ? "yes" : "no");
    if (!out_exact) {
        for (int i = 0; i < H; ++i) {
            if (memcmp(&base_out[i], &par_out[i], sizeof(float)) != 0) {
                printf("first_output_mismatch=%d baseline=%.9g candidate=%.9g\n",
                       i, base_out[i], par_out[i]);
                break;
            }
        }
    }

    free(base_scratch); free(par_scratch);
    kvl_expert_cache_close(&base_cache);
    kvl_expert_cache_close(&par_cache);
    kvl_expert_store_close(&store);
    remove(bin_path); remove(idx_path);

    if (!ids_exact || !weights_exact || !out_exact) return 1;
    puts("PACKED_Q8_EXPERT_PARALLEL_BIT_EXACT_PASS");
    return 0;
}
