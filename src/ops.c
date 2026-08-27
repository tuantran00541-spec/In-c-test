#include "kvl/ops.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

float kvl_bf16_to_f32(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

void kvl_matvec_bf16(float *y, const float *x, const uint16_t *w,
                     int in, int out) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(out > 64)
#endif
    for (int o = 0; o < out; ++o) {
        const uint16_t *row = w + (size_t)o * (size_t)in;
        double acc = 0.0;
        for (int i = 0; i < in; ++i)
            acc += (double)kvl_bf16_to_f32(row[i]) * (double)x[i];
        y[o] = (float)acc;
    }
}

void kvl_silu_mul(float *y, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; ++i) {
        const float g = gate[i];
        const float silu = g / (1.0f + expf(-g));
        y[i] = silu * up[i];
    }
}

void kvl_rmsnorm_bf16(float *y, const float *x, const uint16_t *weight,
                      int n, float eps) {
    double ms = 0.0;
    for (int i = 0; i < n; ++i) ms += (double)x[i] * (double)x[i];
    const float inv = 1.0f / sqrtf((float)(ms / (double)n) + eps);
    for (int i = 0; i < n; ++i)
        y[i] = x[i] * inv * kvl_bf16_to_f32(weight[i]);
}

/* Kimi-VL's apply_rotary_pos_emb first reshapes [D] -> [D/2,2], transposes
 * the final two dimensions, then flattens. For a raw projection laid out
 * [e0,o0,e1,o1,...], that produces [all evens, all odds]. rotate_half then
 * forms the usual complex pairs. Compute that exact result directly. */
static void rope_interleaved(float *dst, const float *raw, int dim,
                             int position, float theta) {
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const double exponent = (double)(2 * i) / (double)dim;
        const double inv_freq = pow((double)theta, -exponent);
        const double angle = (double)position * inv_freq;
        const float c = (float)cos(angle);
        const float si = (float)sin(angle);
        const float a = raw[2 * i];
        const float b = raw[2 * i + 1];
        dst[i] = a * c - b * si;
        dst[half + i] = b * c + a * si;
    }
}

int kvl_mla_prefill_bf16(float *out, const float *x, int seq_len,
                         const KvlMlaConfig *cfg, const KvlMlaBF16 *w) {
    if (!out || !x || !cfg || !w || !w->q_proj || !w->kv_a_proj ||
        !w->kv_a_norm || !w->kv_b_proj || !w->o_proj || seq_len <= 0 ||
        cfg->hidden_size <= 0 || cfg->num_heads <= 0 || cfg->qk_nope_dim <= 0 ||
        cfg->qk_rope_dim <= 0 || (cfg->qk_rope_dim & 1) || cfg->v_head_dim <= 0 ||
        cfg->kv_lora_rank <= 0 || cfg->rope_theta <= 0.0f)
        return -1;

    const int S = seq_len, H = cfg->hidden_size, NH = cfg->num_heads;
    const int DN = cfg->qk_nope_dim, DR = cfg->qk_rope_dim, DV = cfg->v_head_dim;
    const int QD = DN + DR, R = cfg->kv_lora_rank;
    const int QO = NH * QD, KVO = R + DR, KVB = NH * (DN + DV);

    float *q_states = (float *)malloc((size_t)S * NH * QD * sizeof(float));
    float *k_states = (float *)malloc((size_t)S * NH * QD * sizeof(float));
    float *v_states = (float *)malloc((size_t)S * NH * DV * sizeof(float));
    float *qtmp = (float *)malloc((size_t)QO * sizeof(float));
    float *katmp = (float *)malloc((size_t)KVO * sizeof(float));
    float *latent = (float *)malloc((size_t)R * sizeof(float));
    float *kvtmp = (float *)malloc((size_t)KVB * sizeof(float));
    float *rope = (float *)malloc((size_t)DR * sizeof(float));
    float *head_out = (float *)malloc((size_t)NH * DV * sizeof(float));
    float *scores = (float *)malloc((size_t)S * sizeof(float));
    if (!q_states || !k_states || !v_states || !qtmp || !katmp || !latent ||
        !kvtmp || !rope || !head_out || !scores) {
        free(q_states); free(k_states); free(v_states); free(qtmp); free(katmp);
        free(latent); free(kvtmp); free(rope); free(head_out); free(scores);
        return -1;
    }

    for (int t = 0; t < S; ++t) {
        const float *xt = x + (size_t)t * H;
        kvl_matvec_bf16(qtmp, xt, w->q_proj, H, QO);
        kvl_matvec_bf16(katmp, xt, w->kv_a_proj, H, KVO);
        kvl_rmsnorm_bf16(latent, katmp, w->kv_a_norm, R, cfg->rms_eps);
        kvl_matvec_bf16(kvtmp, latent, w->kv_b_proj, R, KVB);
        rope_interleaved(rope, katmp + R, DR, t, cfg->rope_theta);

        for (int h = 0; h < NH; ++h) {
            const float *qh = qtmp + (size_t)h * QD;
            float *qd = q_states + ((size_t)t * NH + h) * QD;
            memcpy(qd, qh, (size_t)DN * sizeof(float));
            rope_interleaved(qd + DN, qh + DN, DR, t, cfg->rope_theta);

            const float *kvh = kvtmp + (size_t)h * (DN + DV);
            float *kd = k_states + ((size_t)t * NH + h) * QD;
            memcpy(kd, kvh, (size_t)DN * sizeof(float));
            memcpy(kd + DN, rope, (size_t)DR * sizeof(float));
            memcpy(v_states + ((size_t)t * NH + h) * DV,
                   kvh + DN, (size_t)DV * sizeof(float));
        }
    }

    const float scale = 1.0f / sqrtf((float)QD);
    for (int t = 0; t < S; ++t) {
        for (int h = 0; h < NH; ++h) {
            const float *q = q_states + ((size_t)t * NH + h) * QD;
            float max_score = -INFINITY;
            for (int j = 0; j <= t; ++j) {
                const float *k = k_states + ((size_t)j * NH + h) * QD;
                double dot = 0.0;
                for (int d = 0; d < QD; ++d) dot += (double)q[d] * (double)k[d];
                scores[j] = (float)dot * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }
            double denom = 0.0;
            for (int j = 0; j <= t; ++j) {
                scores[j] = expf(scores[j] - max_score);
                denom += scores[j];
            }
            float *ho = head_out + (size_t)h * DV;
            for (int d = 0; d < DV; ++d) {
                double acc = 0.0;
                for (int j = 0; j <= t; ++j) {
                    const float *v = v_states + ((size_t)j * NH + h) * DV;
                    acc += ((double)scores[j] / denom) * (double)v[d];
                }
                ho[d] = (float)acc;
            }
        }
        kvl_matvec_bf16(out + (size_t)t * H, head_out, w->o_proj, NH * DV, H);
    }

    free(q_states); free(k_states); free(v_states); free(qtmp); free(katmp);
    free(latent); free(kvtmp); free(rope); free(head_out); free(scores);
    return 0;
}

int kvl_mlp_bf16(float *y, const float *x, const KvlMlpBF16 *mlp,
                 int hidden_size, float *scratch) {
    if (!y || !x || !mlp || !mlp->gate || !mlp->up || !mlp->down ||
        hidden_size <= 0 || mlp->intermediate_size <= 0 || !scratch)
        return -1;
    const int I = mlp->intermediate_size;
    float *gate = scratch;
    float *up = gate + I;
    float *act = up + I;
    kvl_matvec_bf16(gate, x, mlp->gate, hidden_size, I);
    kvl_matvec_bf16(up, x, mlp->up, hidden_size, I);
    kvl_silu_mul(act, gate, up, I);
    kvl_matvec_bf16(y, act, mlp->down, I, hidden_size);
    return 0;
}

typedef struct { float score; int id; } Choice;

static int better(float a, int ia, float b, int ib) {
    return (a > b) || (a == b && ia < ib);
}

static void insert_top(Choice *best, int k, float score, int id) {
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

int kvl_router_noaux_tc(const KvlRouterConfig *cfg, const float *x,
                        const float *router_weight, const float *correction_bias,
                        int *top_ids, float *top_weights) {
    if (!cfg || !x || !router_weight || !correction_bias || !top_ids || !top_weights ||
        cfg->hidden_size <= 0 || cfg->n_experts <= 0 || cfg->top_k <= 0 ||
        cfg->top_k > cfg->n_experts || cfg->n_group <= 0 ||
        cfg->n_experts % cfg->n_group != 0 || cfg->topk_group <= 0 ||
        cfg->topk_group > cfg->n_group)
        return -1;

    const int E = cfg->n_experts, H = cfg->hidden_size;
    const int per_group = E / cfg->n_group;
    float *scores = (float *)malloc((size_t)E * sizeof(float));
    float *choice = (float *)malloc((size_t)E * sizeof(float));
    float *group_scores = (float *)malloc((size_t)cfg->n_group * sizeof(float));
    unsigned char *group_keep = (unsigned char *)calloc((size_t)cfg->n_group, 1);
    Choice *gbest = (Choice *)malloc((size_t)cfg->topk_group * sizeof(Choice));
    Choice *ebest = (Choice *)malloc((size_t)cfg->top_k * sizeof(Choice));
    if (!scores || !choice || !group_scores || !group_keep || !gbest || !ebest) {
        free(scores); free(choice); free(group_scores); free(group_keep); free(gbest); free(ebest);
        return -1;
    }

    for (int e = 0; e < E; ++e) {
        const float *row = router_weight + (size_t)e * H;
        double z = 0.0;
        for (int i = 0; i < H; ++i) z += (double)row[i] * (double)x[i];
        const float s = 1.0f / (1.0f + expf(-(float)z));
        scores[e] = s;
        choice[e] = s + correction_bias[e];
    }

    /* Official noaux_tc group score = sum of the two largest choice scores in a group. */
    for (int g = 0; g < cfg->n_group; ++g) {
        float a = -INFINITY, b = -INFINITY;
        const int begin = g * per_group;
        for (int j = 0; j < per_group; ++j) {
            float v = choice[begin + j];
            if (v > a) { b = a; a = v; }
            else if (v > b) b = v;
        }
        if (per_group == 1) b = 0.0f;
        group_scores[g] = a + b;
    }
    for (int i = 0; i < cfg->topk_group; ++i) gbest[i] = (Choice){0.0f, -1};
    for (int g = 0; g < cfg->n_group; ++g) insert_top(gbest, cfg->topk_group, group_scores[g], g);
    for (int i = 0; i < cfg->topk_group; ++i) if (gbest[i].id >= 0) group_keep[gbest[i].id] = 1;

    for (int i = 0; i < cfg->top_k; ++i) ebest[i] = (Choice){0.0f, -1};
    for (int e = 0; e < E; ++e) {
        const int g = e / per_group;
        /* Match the model's masked_fill(..., 0.0) semantics exactly. */
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

    free(scores); free(choice); free(group_scores); free(group_keep); free(gbest); free(ebest);
    return 0;
}

int kvl_moe_token_bf16(KvlExpertCache *cache, int layer,
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
    if (!cache || !router_cfg || !x || !router_weight || !correction_bias || !out ||
        !top_ids || !top_weights || !scratch || expert_intermediate_size <= 0)
        return -1;
    const int H = router_cfg->hidden_size;
    const int I = expert_intermediate_size;
    const int maxI = (shared && shared->intermediate_size > I) ? shared->intermediate_size : I;
    if (kvl_router_noaux_tc(router_cfg, x, router_weight, correction_bias,
                            top_ids, top_weights) != 0)
        return -1;

    (void)kvl_expert_cache_getmany(cache, layer, top_ids, router_cfg->top_k);
    for (int i = 0; i < H; ++i) out[i] = 0.0f;

    float *gate = scratch;
    float *up = gate + maxI;
    float *act = up + maxI;
    float *tmp = act + maxI;

    for (int j = 0; j < router_cfg->top_k; ++j) {
        KvlCachedExpert q;
        if (kvl_expert_cache_get(cache, layer, top_ids[j], &q) != 0) return -1;
        const size_t need_gu = (size_t)H * I * sizeof(uint16_t);
        const size_t need_dn = (size_t)I * H * sizeof(uint16_t);
        if (q.record->gate_bytes != need_gu || q.record->up_bytes != need_gu ||
            q.record->down_bytes != need_dn)
            return -1;
        kvl_matvec_bf16(gate, x, (const uint16_t *)q.gate, H, I);
        kvl_matvec_bf16(up, x, (const uint16_t *)q.up, H, I);
        kvl_silu_mul(act, gate, up, I);
        kvl_matvec_bf16(tmp, act, (const uint16_t *)q.down, I, H);
        const float w = top_weights[j];
        for (int i = 0; i < H; ++i) out[i] += w * tmp[i];
    }

    if (shared) {
        if (kvl_mlp_bf16(tmp, x, shared, H, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) out[i] += tmp[i];
    }
    return 0;
}
