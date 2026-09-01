#include "shared_q8_runtime.h"

#include "kvl/format.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KVL_SHARED_I 2816
#define KVL_SHARED_DEFAULT_CACHE_MIB 432u

static KvlExpertStore g_shared_store;
static KvlExpertCache g_shared_cache;
static int g_shared_state; /* 0 unopened, 1 open, -1 failed */
static int g_shared_atexit;

static void shared_cleanup_atexit(void) {
    if (g_shared_state == 1) {
        fprintf(stderr, "kvl_shared_q8: final cache stats\n");
        kvl_expert_cache_report(&g_shared_cache);
        kvl_expert_cache_close(&g_shared_cache);
        kvl_expert_store_close(&g_shared_store);
        g_shared_state = 0;
    }
}

static int parse_cache_mib(size_t *bytes) {
    unsigned long long mib = KVL_SHARED_DEFAULT_CACHE_MIB;
    const char *raw = getenv("KVL_SHARED_Q8_CACHE_MIB");
    if (raw && raw[0]) {
        errno = 0;
        char *end = NULL;
        const unsigned long long v = strtoull(raw, &end, 10);
        if (errno || !end || *end != '\0' || v == 0 ||
            v > (unsigned long long)(SIZE_MAX / 1048576u)) {
            fprintf(stderr, "kvl_shared_q8: invalid KVL_SHARED_Q8_CACHE_MIB=%s\n", raw);
            return -1;
        }
        mib = v;
    }
    *bytes = (size_t)mib * 1048576u;
    return 0;
}

static int shared_open(void) {
    if (g_shared_state != 0) return g_shared_state == 1 ? 0 : -1;
    const char *bin = getenv("KVL_SHARED_Q8_BIN");
    const char *idx = getenv("KVL_SHARED_Q8_IDX");
    if (!bin || !bin[0] || !idx || !idx[0]) {
        fprintf(stderr,
                "kvl_shared_q8: set KVL_SHARED_Q8_BIN and KVL_SHARED_Q8_IDX\n");
        g_shared_state = -1;
        return -1;
    }
    memset(&g_shared_store, 0, sizeof g_shared_store);
    memset(&g_shared_cache, 0, sizeof g_shared_cache);
    if (kvl_expert_store_open(&g_shared_store, bin, idx, 1) != 0) {
        fprintf(stderr, "kvl_shared_q8: cannot open sidecar store\n");
        g_shared_state = -1;
        return -1;
    }
    if (g_shared_store.hdr.dtype != KVL_DTYPE_Q8_ROW ||
        g_shared_store.hdr.n_experts != 1) {
        fprintf(stderr, "kvl_shared_q8: incompatible store dtype=%u experts=%u\n",
                g_shared_store.hdr.dtype, g_shared_store.hdr.n_experts);
        kvl_expert_store_close(&g_shared_store);
        g_shared_state = -1;
        return -1;
    }
    size_t budget = 0;
    if (parse_cache_mib(&budget) != 0 ||
        kvl_expert_cache_init(&g_shared_cache, &g_shared_store, budget) != 0) {
        kvl_expert_store_close(&g_shared_store);
        g_shared_state = -1;
        return -1;
    }
    g_shared_state = 1;
    if (!g_shared_atexit) {
        atexit(shared_cleanup_atexit);
        g_shared_atexit = 1;
    }
    fprintf(stderr,
            "kvl_shared_q8: opened records=%u cache=%.2f MiB slots=%d direct_io=%s\n",
            g_shared_store.hdr.n_records, budget / 1048576.0,
            g_shared_cache.n_slots, g_shared_store.direct_io ? "yes" : "no");
    return 0;
}

static int shared_mlp(int layer, const float *x, int hidden_size,
                      float *out, float *scratch) {
    if (shared_open() != 0 || !x || !out || !scratch || hidden_size <= 0)
        return -1;
    KvlCachedExpert q;
    if (kvl_expert_cache_get(&g_shared_cache, layer, 0, &q) != 0) {
        fprintf(stderr, "kvl_shared_q8: missing layer=%d\n", layer);
        return -1;
    }
    const size_t gate_need = (size_t)KVL_SHARED_I * sizeof(float) +
                             (size_t)KVL_SHARED_I * hidden_size;
    const size_t down_need = (size_t)hidden_size * sizeof(float) +
                             (size_t)hidden_size * KVL_SHARED_I;
    if (q.record->gate_bytes != gate_need || q.record->up_bytes != gate_need ||
        q.record->down_bytes != down_need) {
        fprintf(stderr,
                "kvl_shared_q8: L%d shape mismatch gate=%llu up=%llu down=%llu\n",
                layer,
                (unsigned long long)q.record->gate_bytes,
                (unsigned long long)q.record->up_bytes,
                (unsigned long long)q.record->down_bytes);
        return -1;
    }

    float *gate = scratch;
    float *up = gate + KVL_SHARED_I;
    float *act = up + KVL_SHARED_I;
    kvl_matvec_q8_rowwise(gate, x, q.gate, hidden_size, KVL_SHARED_I);
    kvl_matvec_q8_rowwise(up, x, q.up, hidden_size, KVL_SHARED_I);
    kvl_silu_mul(act, gate, up, KVL_SHARED_I);
    kvl_matvec_q8_rowwise(out, act, q.down, KVL_SHARED_I, hidden_size);
    return 0;
}

int kvl_moe_token_q8_shared_sidecar_auto(KvlExpertCache *cache, int layer,
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
    (void)shared;
    if (!router_cfg || !scratch) return -1;
    if (kvl_moe_token_auto(cache, layer, router_cfg, x, router_weight,
                           correction_bias, expert_intermediate_size, NULL,
                           out, top_ids, top_weights, scratch) != 0)
        return -1;

    /* Routed path is finished before this scratch region is reused. The normal
     * generator allocates 3*DENSE_I+H floats, so [3*SHARED_I, +H) fits safely. */
    float *shared_out = scratch + (size_t)3 * KVL_SHARED_I;
    if (shared_mlp(layer, x, router_cfg->hidden_size, shared_out, scratch) != 0)
        return -1;
    for (int i = 0; i < router_cfg->hidden_size; ++i)
        out[i] += shared_out[i];
    return 0;
}

void kvl_shared_q8_sidecar_report(void) {
    if (g_shared_state == 1) kvl_expert_cache_report(&g_shared_cache);
}

void kvl_shared_q8_sidecar_close(void) {
    if (g_shared_state != 1) return;
    kvl_expert_cache_close(&g_shared_cache);
    kvl_expert_store_close(&g_shared_store);
    g_shared_state = 0;
}
