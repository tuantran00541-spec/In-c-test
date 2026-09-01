#include "kvl/ops.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

enum { KVL_ROUTER_STACK_MAX = 256 };
typedef struct { float score; int id; } StackChoice;

static int better(float a, int ia, float b, int ib) {
    return (a > b) || (a == b && ia < ib);
}

static void insert_top(StackChoice *best, int k, float score, int id) {
    int pos = k;
    for (int i = 0; i < k; ++i) {
        if (best[i].id < 0 || better(score, id, best[i].score, best[i].id)) {
            pos = i;
            break;
        }
    }
    if (pos == k) return;
    for (int i = k - 1; i > pos; --i) best[i] = best[i - 1];
    best[pos].score = score;
    best[pos].id = id;
}

int kvl_router_noaux_tc_stack(const KvlRouterConfig *cfg, const float *x,
                              const float *router_weight,
                              const float *correction_bias,
                              int *top_ids, float *top_weights) {
    if (!cfg || !x || !router_weight || !correction_bias || !top_ids || !top_weights ||
        cfg->hidden_size <= 0 || cfg->n_experts <= 0 || cfg->top_k <= 0 ||
        cfg->top_k > cfg->n_experts || cfg->n_group <= 0 ||
        cfg->n_experts % cfg->n_group != 0 || cfg->topk_group <= 0 ||
        cfg->topk_group > cfg->n_group)
        return -1;

    /* Keep this research path bounded and generic. Larger future models retain
     * the production heap-backed implementation byte-for-byte. */
    if (cfg->n_experts > KVL_ROUTER_STACK_MAX ||
        cfg->n_group > KVL_ROUTER_STACK_MAX ||
        cfg->top_k > KVL_ROUTER_STACK_MAX ||
        cfg->topk_group > KVL_ROUTER_STACK_MAX)
        return kvl_router_noaux_tc(cfg, x, router_weight, correction_bias,
                                   top_ids, top_weights);

    const int E = cfg->n_experts, H = cfg->hidden_size;
    const int per_group = E / cfg->n_group;
    float scores[KVL_ROUTER_STACK_MAX];
    float choice[KVL_ROUTER_STACK_MAX];
    float group_scores[KVL_ROUTER_STACK_MAX];
    unsigned char group_keep[KVL_ROUTER_STACK_MAX];
    StackChoice gbest[KVL_ROUTER_STACK_MAX];
    StackChoice ebest[KVL_ROUTER_STACK_MAX];
    memset(group_keep, 0, sizeof group_keep);

    for (int e = 0; e < E; ++e) {
        const float *row = router_weight + (size_t)e * H;
        double z = 0.0;
        for (int i = 0; i < H; ++i) z += (double)row[i] * (double)x[i];
        const float s = 1.0f / (1.0f + expf(-(float)z));
        scores[e] = s;
        choice[e] = s + correction_bias[e];
    }

    for (int g = 0; g < cfg->n_group; ++g) {
        float a = -INFINITY, b = -INFINITY;
        const int begin = g * per_group;
        for (int j = 0; j < per_group; ++j) {
            const float v = choice[begin + j];
            if (v > a) { b = a; a = v; }
            else if (v > b) b = v;
        }
        if (per_group == 1) b = 0.0f;
        group_scores[g] = a + b;
    }

    for (int i = 0; i < cfg->topk_group; ++i)
        gbest[i] = (StackChoice){0.0f, -1};
    for (int g = 0; g < cfg->n_group; ++g)
        insert_top(gbest, cfg->topk_group, group_scores[g], g);
    for (int i = 0; i < cfg->topk_group; ++i)
        if (gbest[i].id >= 0) group_keep[gbest[i].id] = 1;

    for (int i = 0; i < cfg->top_k; ++i)
        ebest[i] = (StackChoice){0.0f, -1};
    for (int e = 0; e < E; ++e) {
        const int g = e / per_group;
        const float v = group_keep[g] ? choice[e] : 0.0f;
        insert_top(ebest, cfg->top_k, v, e);
    }

    double sum = 0.0;
    for (int i = 0; i < cfg->top_k; ++i) {
        top_ids[i] = ebest[i].id;
        top_weights[i] = scores[ebest[i].id];
        sum += top_weights[i];
    }
    if (cfg->top_k > 1 && cfg->norm_topk_prob) {
        const float denom = (float)(sum + 1e-20);
        for (int i = 0; i < cfg->top_k; ++i)
            top_weights[i] = top_weights[i] / denom * cfg->routed_scaling_factor;
    } else {
        for (int i = 0; i < cfg->top_k; ++i)
            top_weights[i] *= cfg->routed_scaling_factor;
    }
    return 0;
}
