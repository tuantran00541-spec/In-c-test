#include "kvl/ops.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef KVL_USE_AVX2
#include <immintrin.h>
#endif

#define KVL_MOE_LAB_MAX_LAYERS 256
#define KVL_MOE_LAB_MAX_EXPERTS 256

typedef struct { float score; int id; } Q8Choice;

static unsigned char g_moe_disabled[KVL_MOE_LAB_MAX_LAYERS][KVL_MOE_LAB_MAX_EXPERTS];
/* 0=uninitialised, 1=no mask requested, 2=loaded, -1=load error. */
static int g_moe_mask_state;
static FILE *g_moe_trace;
static int g_moe_trace_state;
static unsigned long long g_moe_trace_event;

static int q8_better(float a, int ia, float b, int ib) {
    return (a > b) || (a == b && ia < ib);
}

static void q8_insert_top(Q8Choice *best, int k, float score, int id) {
    int pos = k;
    for (int i = 0; i < k; ++i) {
        if (best[i].id < 0 || q8_better(score, id, best[i].score, best[i].id)) {
            pos = i;
            break;
        }
    }
    if (pos == k) return;
    for (int i = k - 1; i > pos; --i) best[i] = best[i - 1];
    best[pos].score = score;
    best[pos].id = id;
}

static int moe_mask_load(void) {
    if (g_moe_mask_state) return g_moe_mask_state < 0 ? -1 : g_moe_mask_state - 1;
    const char *path = getenv("KVL_MOE_MASK");
    if (!path || !path[0]) {
        g_moe_mask_state = 1;
        return 0;
    }
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "kvl: cannot open KVL_MOE_MASK=%s\n", path);
        g_moe_mask_state = -1;
        return -1;
    }
    char line[256];
    int lineno = 0;
    while (fgets(line, sizeof line, f)) {
        ++lineno;
        char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (!*p || *p == '#') continue;
        int layer = -1, expert = -1;
        if (sscanf(p, "%d %d", &layer, &expert) != 2 ||
            layer < 0 || layer >= KVL_MOE_LAB_MAX_LAYERS ||
            expert < 0 || expert >= KVL_MOE_LAB_MAX_EXPERTS) {
            fprintf(stderr, "kvl: bad KVL_MOE_MASK line %d: %s", lineno, line);
            fclose(f);
            g_moe_mask_state = -1;
            return -1;
        }
        g_moe_disabled[layer][expert] = 1;
    }
    fclose(f);
    g_moe_mask_state = 2;
    return 1;
}

static int moe_is_disabled(int layer, int expert) {
    return layer >= 0 && layer < KVL_MOE_LAB_MAX_LAYERS &&
           expert >= 0 && expert < KVL_MOE_LAB_MAX_EXPERTS &&
           g_moe_disabled[layer][expert] != 0;
}

/* The baseline path deliberately calls the production router unchanged. Only a
 * requested mask enters this copy, where disabled experts are removed before
 * group scoring and before final top-k selection. */
static int q8_router_for_layer(int layer, const KvlRouterConfig *cfg, const float *x,
                               const float *router_weight, const float *correction_bias,
                               int *top_ids, float *top_weights) {
    const int mask = moe_mask_load();
    if (mask < 0) return -1;
    if (!mask)
        return kvl_router_noaux_tc(cfg, x, router_weight, correction_bias,
                                   top_ids, top_weights);
    if (!cfg || !x || !router_weight || !correction_bias || !top_ids || !top_weights ||
        cfg->hidden_size <= 0 || cfg->n_experts <= 0 || cfg->top_k <= 0 ||
        cfg->top_k > cfg->n_experts || cfg->n_group <= 0 ||
        cfg->n_experts % cfg->n_group != 0 || cfg->topk_group <= 0 ||
        cfg->topk_group > cfg->n_group || cfg->n_experts > KVL_MOE_LAB_MAX_EXPERTS)
        return -1;

    const int E = cfg->n_experts, H = cfg->hidden_size;
    const int per_group = E / cfg->n_group;
    float *scores = (float *)malloc((size_t)E * sizeof(float));
    float *choice = (float *)malloc((size_t)E * sizeof(float));
    float *group_scores = (float *)malloc((size_t)cfg->n_group * sizeof(float));
    unsigned char *group_keep = (unsigned char *)calloc((size_t)cfg->n_group, 1);
    Q8Choice *gbest = (Q8Choice *)malloc((size_t)cfg->topk_group * sizeof(Q8Choice));
    Q8Choice *ebest = (Q8Choice *)malloc((size_t)cfg->top_k * sizeof(Q8Choice));
    if (!scores || !choice || !group_scores || !group_keep || !gbest || !ebest) {
        free(scores); free(choice); free(group_scores); free(group_keep); free(gbest); free(ebest);
        return -1;
    }

    int enabled = 0;
    for (int e = 0; e < E; ++e) {
        const float *row = router_weight + (size_t)e * H;
        double z = 0.0;
        for (int i = 0; i < H; ++i) z += (double)row[i] * (double)x[i];
        const float s = 1.0f / (1.0f + expf(-(float)z));
        scores[e] = s;
        choice[e] = s + correction_bias[e];
        if (!moe_is_disabled(layer, e)) ++enabled;
    }
    if (enabled < cfg->top_k) {
        free(scores); free(choice); free(group_scores); free(group_keep); free(gbest); free(ebest);
        return -1;
    }

    for (int g = 0; g < cfg->n_group; ++g) {
        float a = -INFINITY, b = -INFINITY;
        int count = 0;
        const int begin = g * per_group;
        for (int j = 0; j < per_group; ++j) {
            const int e = begin + j;
            if (moe_is_disabled(layer, e)) continue;
            ++count;
            const float v = choice[e];
            if (v > a) { b = a; a = v; }
            else if (v > b) b = v;
        }
        if (count == 0) group_scores[g] = -INFINITY;
        else if (count == 1) group_scores[g] = a;
        else group_scores[g] = a + b;
    }
    for (int i = 0; i < cfg->topk_group; ++i) gbest[i] = (Q8Choice){0.0f, -1};
    for (int g = 0; g < cfg->n_group; ++g)
        q8_insert_top(gbest, cfg->topk_group, group_scores[g], g);
    for (int i = 0; i < cfg->topk_group; ++i)
        if (gbest[i].id >= 0 && isfinite(gbest[i].score)) group_keep[gbest[i].id] = 1;

    for (int i = 0; i < cfg->top_k; ++i) ebest[i] = (Q8Choice){0.0f, -1};
    for (int e = 0; e < E; ++e) {
        if (moe_is_disabled(layer, e)) continue;
        const int g = e / per_group;
        const float v = group_keep[g] ? choice[e] : 0.0f;
        q8_insert_top(ebest, cfg->top_k, v, e);
    }

    double sum = 0.0;
    for (int i = 0; i < cfg->top_k; ++i) {
        if (ebest[i].id < 0) {
            free(scores); free(choice); free(group_scores); free(group_keep); free(gbest); free(ebest);
            return -1;
        }
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

    free(scores); free(choice); free(group_scores); free(group_keep); free(gbest); free(ebest);
    return 0;
}

static void moe_trace_open(void) {
    if (g_moe_trace_state) return;
    g_moe_trace_state = 1;
    const char *path = getenv("KVL_MOE_TRACE");
    if (!path || !path[0]) return;
    g_moe_trace = fopen(path, "a+");
    if (!g_moe_trace) {
        fprintf(stderr, "kvl: cannot open KVL_MOE_TRACE=%s\n", path);
        return;
    }
    if (fseek(g_moe_trace, 0, SEEK_END) == 0 && ftell(g_moe_trace) == 0)
        fputs("# event\tlayer\texpert\trouter_weight\toutput_l2\tsaliency\n", g_moe_trace);
}

static unsigned long long moe_trace_begin(void) {
    unsigned long long event = 0;
#ifdef _OPENMP
#pragma omp critical(kvl_moe_trace_io)
#endif
    {
        moe_trace_open();
        if (g_moe_trace) event = ++g_moe_trace_event;
    }
    return event;
}

static void moe_trace_expert(unsigned long long event, int layer, int expert,
                             float router_weight, const float *output, int n) {
    if (!event || !output || n <= 0) return;
    double sq = 0.0;
    for (int i = 0; i < n; ++i) sq += (double)output[i] * (double)output[i];
    const double l2 = sqrt(sq);
    const double saliency = fabs((double)router_weight) * l2;
#ifdef _OPENMP
#pragma omp critical(kvl_moe_trace_io)
#endif
    {
        if (g_moe_trace)
            fprintf(g_moe_trace, "%llu\t%d\t%d\t%.9g\t%.12g\t%.12g\n",
                    event, layer, expert, router_weight, l2, saliency);
    }
}

static void moe_trace_flush(void) {
#ifdef _OPENMP
#pragma omp critical(kvl_moe_trace_io)
#endif
    {
        if (g_moe_trace) fflush(g_moe_trace);
    }
}

static float dot_q8_ref(const int8_t *row, const float *x, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double)row[i] * (double)x[i];
    return (float)acc;
}

#ifdef KVL_USE_AVX2
static float dot_q8_avx2(const int8_t *row, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i q8 = _mm_loadl_epi64((const __m128i *)(row + i));
        const __m256i q32 = _mm256_cvtepi8_epi32(q8);
        const __m256 qf = _mm256_cvtepi32_ps(q32);
        const __m256 xv = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(qf, xv));
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float out = _mm_cvtss_f32(sum);
    for (; i < n; ++i) out += (float)row[i] * x[i];
    return out;
}
#endif

void kvl_matvec_q8_rowwise(float *y, const float *x, const void *blob,
                           int in, int out) {
    if (!y || !x || !blob || in <= 0 || out <= 0) return;
    const float *scales = (const float *)blob;
    const int8_t *q = (const int8_t *)(scales + out);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(out > 64)
#endif
    for (int o = 0; o < out; ++o) {
        const int8_t *row = q + (size_t)o * (size_t)in;
#ifdef KVL_USE_AVX2
        const float dot = dot_q8_avx2(row, x, in);
#else
        const float dot = dot_q8_ref(row, x, in);
#endif
        y[o] = scales[o] * dot;
    }
}

int kvl_moe_token_auto(KvlExpertCache *cache, int layer,
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
    if (!cache || !cache->store) return -1;
    if (cache->store->hdr.dtype == KVL_DTYPE_BF16)
        return kvl_moe_token_bf16(cache, layer, router_cfg, x, router_weight,
                                  correction_bias, expert_intermediate_size, shared,
                                  out, top_ids, top_weights, scratch);
    if (cache->store->hdr.dtype != KVL_DTYPE_Q8_ROW || !router_cfg || !x ||
        !router_weight || !correction_bias || !out || !top_ids || !top_weights ||
        !scratch || expert_intermediate_size <= 0)
        return -1;

    const int H = router_cfg->hidden_size;
    const int I = expert_intermediate_size;
    const int maxI = (shared && shared->intermediate_size > I) ? shared->intermediate_size : I;
    if (q8_router_for_layer(layer, router_cfg, x, router_weight, correction_bias,
                            top_ids, top_weights) != 0)
        return -1;

    (void)kvl_expert_cache_getmany(cache, layer, top_ids, router_cfg->top_k);
    for (int i = 0; i < H; ++i) out[i] = 0.0f;

    float *gate = scratch;
    float *up = gate + maxI;
    float *act = up + maxI;
    float *tmp = act + maxI;
    const unsigned long long trace_event = moe_trace_begin();

    const size_t need_gu = (size_t)I * sizeof(float) + (size_t)I * (size_t)H;
    const size_t need_dn = (size_t)H * sizeof(float) + (size_t)H * (size_t)I;
    for (int j = 0; j < router_cfg->top_k; ++j) {
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
        moe_trace_expert(trace_event, layer, top_ids[j], w, tmp, H);
        for (int i = 0; i < H; ++i) out[i] += w * tmp[i];
    }
    moe_trace_flush();

    if (shared) {
        if (kvl_mlp_bf16(tmp, x, shared, H, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) out[i] += tmp[i];
    }
    return 0;
}
