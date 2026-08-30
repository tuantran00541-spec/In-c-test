#include "kvl/ops.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Research-only activation trace.
 *
 * Enabled only when KVL_MOE_ACT_TRACE points to a file. The wrapper delegates
 * computation to kvl_moe_token_auto first, then appends one compact record per
 * routed token/layer: one expert-input vector plus the selected expert ids and
 * weights. The six experts share the same input vector, so we never duplicate
 * x six times.
 *
 * Binary format, native little-endian x86 target:
 *   header:
 *     char magic[8] = "KVLACT01"
 *     uint32 version = 1
 *     uint32 hidden
 *     uint32 top_k
 *     uint32 endian_marker = 0x01020304
 *   repeated records:
 *     uint64 event
 *     int32 layer
 *     int32 top_ids[top_k]
 *     float top_weights[top_k]
 *     float x[hidden]
 *
 * The file is calibration evidence, never a model-weight artifact.
 */

static FILE *g_act_trace;
static int g_act_trace_state;
static uint64_t g_act_event;
static int g_act_hidden;
static int g_act_top_k;

static int write_u32(FILE *f, uint32_t v) { return fwrite(&v, sizeof v, 1, f) == 1 ? 0 : -1; }

static int act_trace_open(int hidden, int top_k) {
    if (g_act_trace_state) {
        if (!g_act_trace) return 0;
        return (hidden == g_act_hidden && top_k == g_act_top_k) ? 1 : -1;
    }
    g_act_trace_state = 1;
    const char *path = getenv("KVL_MOE_ACT_TRACE");
    if (!path || !path[0]) return 0;
    if (hidden <= 0 || top_k <= 0 || top_k > 64) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "kvl: cannot open KVL_MOE_ACT_TRACE=%s\n", path);
        return -1;
    }
    static const char magic[8] = {'K','V','L','A','C','T','0','1'};
    if (fwrite(magic, 1, sizeof magic, f) != sizeof magic ||
        write_u32(f, 1u) != 0 ||
        write_u32(f, (uint32_t)hidden) != 0 ||
        write_u32(f, (uint32_t)top_k) != 0 ||
        write_u32(f, UINT32_C(0x01020304)) != 0) {
        fclose(f);
        fprintf(stderr, "kvl: failed to write activation trace header\n");
        return -1;
    }
    g_act_trace = f;
    g_act_hidden = hidden;
    g_act_top_k = top_k;
    return 1;
}

static int act_trace_record(int layer, const float *x, int hidden,
                            const int *top_ids, const float *top_weights, int top_k) {
    const int state = act_trace_open(hidden, top_k);
    if (state <= 0) return state;
    if (!x || !top_ids || !top_weights) return -1;
    const uint64_t event = ++g_act_event;
    const int32_t layer32 = (int32_t)layer;
    if (fwrite(&event, sizeof event, 1, g_act_trace) != 1 ||
        fwrite(&layer32, sizeof layer32, 1, g_act_trace) != 1)
        return -1;
    for (int i = 0; i < top_k; ++i) {
        const int32_t id = (int32_t)top_ids[i];
        if (fwrite(&id, sizeof id, 1, g_act_trace) != 1) return -1;
    }
    if (fwrite(top_weights, sizeof(float), (size_t)top_k, g_act_trace) != (size_t)top_k ||
        fwrite(x, sizeof(float), (size_t)hidden, g_act_trace) != (size_t)hidden)
        return -1;
    return fflush(g_act_trace) == 0 ? 1 : -1;
}

int kvl_moe_token_trace_auto(KvlExpertCache *cache, int layer,
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
    const int rc = kvl_moe_token_auto(cache, layer, router_cfg, x, router_weight,
                                      correction_bias, expert_intermediate_size,
                                      shared, out, top_ids, top_weights, scratch);
    if (rc != 0) return rc;
    if (!router_cfg) return -1;
    const int tr = act_trace_record(layer, x, router_cfg->hidden_size,
                                    top_ids, top_weights, router_cfg->top_k);
    if (tr < 0) {
        fprintf(stderr, "kvl: activation trace write failed\n");
        return -1;
    }
    return 0;
}
