#include "kvl/moe_dynamic_skip.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KVL_DYNSKIP_MAX_LAYERS 256
#define KVL_DYNSKIP_MAX_TOPK 256
#define KVL_DYNSKIP_FIRST_MOE_LAYER 1
#define KVL_DYNSKIP_LAST_MOE_LAYER 26

#define KIMI_IM_END_ID 163586
#define KIMI_IM_USER_ID 163587
#define KIMI_IM_ASSISTANT_ID 163588
#define KIMI_IM_SYSTEM_ID 163594
#define KIMI_IM_MIDDLE_ID 163601
#define KIMI_MEDIA_START_ID 163602
#define KIMI_MEDIA_END_ID 163604
#define KIMI_MEDIA_PAD_ID 163605

typedef struct {
    float threshold;
    int min_keep;
} KvlDynskipPolicy;

typedef struct {
    unsigned long long events;
    unsigned long long routed;
    unsigned long long skipped;
} KvlDynskipStats;

static KvlDynskipPolicy g_policy[KVL_MOE_FAMILY_COUNT][KVL_DYNSKIP_MAX_LAYERS];
static KvlDynskipStats g_stats[KVL_MOE_FAMILY_COUNT][KVL_DYNSKIP_MAX_LAYERS];
static int g_policy_enabled;
static int g_runtime_state; /* 0=uninitialised, 1=ready, -1=error */
static int g_report_registered;
static unsigned char *g_prompt_family;
static int g_prompt_n;
static int g_prefill_layer;
static int g_prefill_pos;
static int g_prefill_active;
static char g_policy_path[1024];
static char g_stats_path[1024];

static const char *family_name(int family) {
    if (family == KVL_MOE_FAMILY_CONTENT) return "content";
    if (family == KVL_MOE_FAMILY_MEDIA) return "media";
    return "control";
}

static int parse_family(const char *s) {
    if (!strcmp(s, "content")) return KVL_MOE_FAMILY_CONTENT;
    if (!strcmp(s, "media")) return KVL_MOE_FAMILY_MEDIA;
    if (!strcmp(s, "control")) return KVL_MOE_FAMILY_CONTROL;
    return -1;
}

void kvl_moe_dynskip_reset_policy(void) {
    memset(g_policy, 0, sizeof g_policy);
    memset(g_stats, 0, sizeof g_stats);
    g_policy_enabled = 0;
}

int kvl_moe_dynskip_set_policy(int family, int layer,
                               float threshold, int min_keep) {
    if ((family != KVL_MOE_FAMILY_CONTENT && family != KVL_MOE_FAMILY_MEDIA) ||
        layer < KVL_DYNSKIP_FIRST_MOE_LAYER || layer >= KVL_DYNSKIP_MAX_LAYERS ||
        !isfinite(threshold) || threshold < 0.0f || threshold > 1.0f ||
        min_keep < 1 || min_keep > KVL_DYNSKIP_MAX_TOPK)
        return -1;
    g_policy[family][layer].threshold = threshold;
    g_policy[family][layer].min_keep = min_keep;
    if (threshold > 0.0f) g_policy_enabled = 1;
    return 0;
}

int kvl_moe_dynskip_load_policy(const char *path) {
    if (!path || !path[0]) return -1;
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    kvl_moe_dynskip_reset_policy();
    char line[512];
    int lineno = 0;
    int entries = 0;
    while (fgets(line, sizeof line, f)) {
        ++lineno;
        char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (!*p || *p == '#') continue;
        char family_text[32];
        int layer = -1, min_keep = -1;
        float threshold = -1.0f;
        char extra = 0;
        if (sscanf(p, "%31s %d %f %d %c", family_text, &layer,
                   &threshold, &min_keep, &extra) != 4) {
            fprintf(stderr, "kvl dynskip: bad policy line %d: %s", lineno, line);
            fclose(f);
            kvl_moe_dynskip_reset_policy();
            return -1;
        }
        const int family = parse_family(family_text);
        if (kvl_moe_dynskip_set_policy(family, layer, threshold, min_keep) != 0) {
            fprintf(stderr, "kvl dynskip: invalid policy line %d: %s", lineno, line);
            fclose(f);
            kvl_moe_dynskip_reset_policy();
            return -1;
        }
        ++entries;
    }
    fclose(f);
    if (!entries || !g_policy_enabled) {
        fprintf(stderr, "kvl dynskip: policy has no active threshold\n");
        kvl_moe_dynskip_reset_policy();
        return -1;
    }
    return 0;
}

int kvl_moe_dynskip_classify_prompt(const int *prompt, int n,
                                    unsigned char *out_family) {
    if (!prompt || n <= 0 || !out_family) return -1;
    enum { OUTSIDE, SYSTEM, USER_HEADER, USER_BODY, MEDIA, ASSISTANT } state = OUTSIDE;
    for (int i = 0; i < n; ++i) {
        const int token = prompt[i];
        int family = KVL_MOE_FAMILY_CONTROL;
        if (token == KIMI_IM_SYSTEM_ID) {
            state = SYSTEM;
        } else if (state == SYSTEM) {
            if (token == KIMI_IM_END_ID) state = OUTSIDE;
        } else if (token == KIMI_IM_USER_ID) {
            state = USER_HEADER;
        } else if (state == USER_HEADER) {
            if (token == KIMI_IM_MIDDLE_ID) state = USER_BODY;
        } else if (state == MEDIA) {
            if (token == KIMI_MEDIA_PAD_ID) family = KVL_MOE_FAMILY_MEDIA;
            if (token == KIMI_MEDIA_END_ID) state = USER_BODY;
        } else if (state == USER_BODY) {
            if (token == KIMI_IM_END_ID) {
                state = OUTSIDE;
            } else if (token == KIMI_MEDIA_START_ID) {
                state = MEDIA;
            } else if (token == KIMI_MEDIA_PAD_ID) {
                family = KVL_MOE_FAMILY_MEDIA;
            } else {
                family = KVL_MOE_FAMILY_CONTENT;
            }
        } else if (token == KIMI_IM_ASSISTANT_ID) {
            state = ASSISTANT;
        } else if (state == ASSISTANT) {
            family = KVL_MOE_FAMILY_CONTROL;
        } else if (token == KIMI_MEDIA_PAD_ID) {
            family = KVL_MOE_FAMILY_MEDIA;
        }
        out_family[i] = (unsigned char)family;
    }
    return 0;
}

int kvl_moe_dynskip_apply_policy(int family, int layer, int top_k,
                                 const float *top_weights,
                                 unsigned char *keep,
                                 int *out_skipped) {
    if (!top_weights || !keep || top_k <= 0 || top_k > KVL_DYNSKIP_MAX_TOPK ||
        layer < 0 || layer >= KVL_DYNSKIP_MAX_LAYERS)
        return -1;
    for (int i = 0; i < top_k; ++i) keep[i] = 1;
    if (out_skipped) *out_skipped = 0;
    if (family != KVL_MOE_FAMILY_CONTENT && family != KVL_MOE_FAMILY_MEDIA)
        return 0;

    const KvlDynskipPolicy policy = g_policy[family][layer];
    if (policy.threshold <= 0.0f) return 0;

    double sum = 0.0;
    for (int i = 0; i < top_k; ++i) {
        if (!isfinite(top_weights[i]) || top_weights[i] < 0.0f) return -1;
        sum += top_weights[i];
    }
    if (!(sum > 0.0)) return 0;

    float mass[KVL_DYNSKIP_MAX_TOPK];
    int kept = 0;
    for (int i = 0; i < top_k; ++i) {
        mass[i] = (float)((double)top_weights[i] / sum);
        keep[i] = (unsigned char)(mass[i] >= policy.threshold);
        kept += keep[i] != 0;
    }

    int min_keep = policy.min_keep;
    if (min_keep < 1) min_keep = 1;
    if (min_keep > top_k) min_keep = top_k;
    while (kept < min_keep) {
        int best = -1;
        for (int i = 0; i < top_k; ++i) {
            if (keep[i]) continue;
            if (best < 0 || mass[i] > mass[best]) best = i;
        }
        if (best < 0) break;
        keep[best] = 1;
        ++kept;
    }

    if (out_skipped) *out_skipped = top_k - kept;
    return 0;
}

static int read_prompt_ids_file(const char *path, int **out_ids, int *out_n) {
    if (!path || !out_ids || !out_n) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int cap = 64, n = 0;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { fclose(f); return -1; }
    for (;;) {
        int id;
        const int rc = fscanf(f, "%d", &id);
        if (rc == EOF) break;
        if (rc != 1 || id < 0) { free(ids); fclose(f); return -1; }
        if (n == cap) {
            cap *= 2;
            int *tmp = (int *)realloc(ids, (size_t)cap * sizeof(int));
            if (!tmp) { free(ids); fclose(f); return -1; }
            ids = tmp;
        }
        ids[n++] = id;
    }
    fclose(f);
    if (!n) { free(ids); return -1; }
    *out_ids = ids;
    *out_n = n;
    return 0;
}

static void write_stats(FILE *f) {
    if (!f) return;
    fputs("family\tlayer\tevents\trouted\tskipped\tskip_fraction\tthreshold\tmin_keep\n", f);
    for (int family = KVL_MOE_FAMILY_CONTROL; family < KVL_MOE_FAMILY_COUNT; ++family) {
        for (int layer = KVL_DYNSKIP_FIRST_MOE_LAYER; layer <= KVL_DYNSKIP_LAST_MOE_LAYER; ++layer) {
            const KvlDynskipStats s = g_stats[family][layer];
            if (!s.events) continue;
            const double frac = s.routed ? (double)s.skipped / (double)s.routed : 0.0;
            fprintf(f, "%s\t%d\t%llu\t%llu\t%llu\t%.9g\t%.9g\t%d\n",
                    family_name(family), layer,
                    s.events, s.routed, s.skipped, frac,
                    g_policy[family][layer].threshold,
                    g_policy[family][layer].min_keep);
        }
    }
}

static void dynskip_report_at_exit(void) {
    if (g_runtime_state <= 0 || !g_policy_enabled) return;
    unsigned long long routed = 0, skipped = 0;
    for (int family = 0; family < KVL_MOE_FAMILY_COUNT; ++family)
        for (int layer = KVL_DYNSKIP_FIRST_MOE_LAYER; layer <= KVL_DYNSKIP_LAST_MOE_LAYER; ++layer) {
            routed += g_stats[family][layer].routed;
            skipped += g_stats[family][layer].skipped;
        }
    fprintf(stderr,
            "KIMI_DYNSKIP_SUMMARY routed=%llu skipped=%llu skip_fraction=%.9g policy=%s\n",
            routed, skipped, routed ? (double)skipped / (double)routed : 0.0,
            g_policy_path[0] ? g_policy_path : "<none>");
    if (g_stats_path[0]) {
        FILE *f = fopen(g_stats_path, "w");
        if (!f) {
            fprintf(stderr, "kvl dynskip: cannot write stats %s\n", g_stats_path);
        } else {
            write_stats(f);
            fclose(f);
        }
    }
}

static int runtime_init(void) {
    if (g_runtime_state) return g_runtime_state > 0 ? 0 : -1;
    g_runtime_state = -1;
    const char *policy = getenv("KVL_MOE_DYNSKIP_POLICY");
    if (!policy || !policy[0]) {
        g_runtime_state = 1;
        return 0;
    }
    const char *static_mask = getenv("KVL_MOE_MASK");
    if (static_mask && static_mask[0]) {
        fprintf(stderr, "kvl dynskip: refusing to combine KVL_MOE_DYNSKIP_POLICY with KVL_MOE_MASK\n");
        return -1;
    }
    if (kvl_moe_dynskip_load_policy(policy) != 0) {
        fprintf(stderr, "kvl dynskip: failed to load policy %s\n", policy);
        return -1;
    }
    snprintf(g_policy_path, sizeof g_policy_path, "%s", policy);
    const char *stats = getenv("KVL_MOE_DYNSKIP_STATS");
    if (stats && stats[0]) snprintf(g_stats_path, sizeof g_stats_path, "%s", stats);

    const char *ids_path = getenv("KVL_MOE_DYNSKIP_PROMPT_IDS");
    int *ids = NULL, n = 0;
    if (!ids_path || !ids_path[0] || read_prompt_ids_file(ids_path, &ids, &n) != 0) {
        fprintf(stderr,
                "kvl dynskip: active policy requires readable KVL_MOE_DYNSKIP_PROMPT_IDS\n");
        free(ids);
        return -1;
    }
    g_prompt_family = (unsigned char *)malloc((size_t)n);
    if (!g_prompt_family || kvl_moe_dynskip_classify_prompt(ids, n, g_prompt_family) != 0) {
        free(ids); free(g_prompt_family); g_prompt_family = NULL;
        return -1;
    }
    free(ids);
    g_prompt_n = n;
    g_prefill_layer = KVL_DYNSKIP_FIRST_MOE_LAYER;
    g_prefill_pos = 0;
    g_prefill_active = 1;
    if (!g_report_registered) {
        if (atexit(dynskip_report_at_exit) != 0) return -1;
        g_report_registered = 1;
    }
    fprintf(stderr,
            "kvl dynskip: enabled prompt=%d policy=%s semantics=no-reroute,no-renorm control/decode=protected\n",
            g_prompt_n, g_policy_path);
    g_runtime_state = 1;
    return 0;
}

static int next_family(int layer, int *out_family) {
    if (!out_family) return -1;
    *out_family = KVL_MOE_FAMILY_CONTROL;
    if (!g_policy_enabled || !g_prefill_active) return 0;
    if (layer != g_prefill_layer || g_prefill_pos < 0 || g_prefill_pos >= g_prompt_n) {
        fprintf(stderr,
                "kvl dynskip: unexpected MoE call order layer=%d expected_layer=%d pos=%d prompt=%d\n",
                layer, g_prefill_layer, g_prefill_pos, g_prompt_n);
        return -1;
    }
    *out_family = g_prompt_family[g_prefill_pos];
    ++g_prefill_pos;
    if (g_prefill_pos == g_prompt_n) {
        g_prefill_pos = 0;
        ++g_prefill_layer;
        if (g_prefill_layer > KVL_DYNSKIP_LAST_MOE_LAYER)
            g_prefill_active = 0; /* decode is intentionally protected */
    }
    return 0;
}

int kvl_moe_token_q8_dynskip_auto(KvlExpertCache *cache, int layer,
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
    if (runtime_init() != 0) return -1;
    if (!g_policy_enabled)
        return kvl_moe_token_auto(cache, layer, router_cfg, x, router_weight,
                                  correction_bias, expert_intermediate_size, shared,
                                  out, top_ids, top_weights, scratch);
    if (!cache || !cache->store || cache->store->hdr.dtype != KVL_DTYPE_Q8_ROW ||
        !router_cfg || !x || !router_weight || !correction_bias || !out ||
        !top_ids || !top_weights || !scratch || expert_intermediate_size <= 0 ||
        router_cfg->top_k <= 0 || router_cfg->top_k > KVL_DYNSKIP_MAX_TOPK)
        return -1;

    int family = KVL_MOE_FAMILY_CONTROL;
    if (next_family(layer, &family) != 0) return -1;
    if (kvl_router_noaux_tc(router_cfg, x, router_weight, correction_bias,
                            top_ids, top_weights) != 0)
        return -1;

    unsigned char keep[KVL_DYNSKIP_MAX_TOPK];
    int skipped = 0;
    if (kvl_moe_dynskip_apply_policy(family, layer, router_cfg->top_k,
                                     top_weights, keep, &skipped) != 0)
        return -1;
    KvlDynskipStats *stats = &g_stats[family][layer];
    ++stats->events;
    stats->routed += (unsigned long long)router_cfg->top_k;
    stats->skipped += (unsigned long long)skipped;

    int kept_ids[KVL_DYNSKIP_MAX_TOPK];
    int kept_n = 0;
    for (int j = 0; j < router_cfg->top_k; ++j) {
        if (keep[j]) kept_ids[kept_n++] = top_ids[j];
        else top_weights[j] = 0.0f;
    }
    if (kept_n <= 0) return -1;
    (void)kvl_expert_cache_getmany(cache, layer, kept_ids, kept_n);

    const int H = router_cfg->hidden_size;
    const int I = expert_intermediate_size;
    const int maxI = (shared && shared->intermediate_size > I) ? shared->intermediate_size : I;
    for (int i = 0; i < H; ++i) out[i] = 0.0f;
    float *gate = scratch;
    float *up = gate + maxI;
    float *act = up + maxI;
    float *tmp = act + maxI;
    const size_t need_gu = (size_t)I * sizeof(float) + (size_t)I * (size_t)H;
    const size_t need_dn = (size_t)H * sizeof(float) + (size_t)H * (size_t)I;

    for (int j = 0; j < router_cfg->top_k; ++j) {
        if (!keep[j]) continue;
        KvlCachedExpert q;
        if (kvl_expert_cache_get(cache, layer, top_ids[j], &q) != 0) return -1;
        if (q.record->gate_bytes != need_gu || q.record->up_bytes != need_gu ||
            q.record->down_bytes != need_dn)
            return -1;
        kvl_matvec_q8_rowwise(gate, x, q.gate, H, I);
        kvl_matvec_q8_rowwise(up, x, q.up, H, I);
        kvl_silu_mul(act, gate, up, I);
        kvl_matvec_q8_rowwise(tmp, act, q.down, I, H);
        const float w = top_weights[j];
        for (int i = 0; i < H; ++i) out[i] += w * tmp[i];
    }

    if (shared) {
        if (kvl_mlp_bf16(tmp, x, shared, H, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) out[i] += tmp[i];
    }
    return 0;
}
