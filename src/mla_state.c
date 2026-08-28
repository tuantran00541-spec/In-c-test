#include "kvl/mla_state.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static void rope_interleaved_v6(float *dst, const float *raw, int dim,
                                int position, float theta) {
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const double exponent = (double)(2 * i) / (double)dim;
        const double inv_freq = pow((double)theta, -exponent);
        const double angle = (double)position * inv_freq;
        const float c = (float)cos(angle);
        const float s = (float)sin(angle);
        const float a = raw[2 * i];
        const float b = raw[2 * i + 1];
        dst[i] = a * c - b * s;
        dst[half + i] = b * c + a * s;
    }
}

int kvl_mla_state_init(KvlMlaState *state, const KvlMlaConfig *cfg, int capacity) {
    if (!state || !cfg || capacity <= 0 || cfg->num_heads <= 0 ||
        cfg->qk_nope_dim <= 0 || cfg->qk_rope_dim <= 0 || cfg->v_head_dim <= 0)
        return -1;
    memset(state, 0, sizeof(*state));
    const size_t qd = (size_t)cfg->qk_nope_dim + (size_t)cfg->qk_rope_dim;
    const size_t nk = (size_t)capacity * (size_t)cfg->num_heads * qd;
    const size_t nv = (size_t)capacity * (size_t)cfg->num_heads * (size_t)cfg->v_head_dim;
    state->keys = (float *)calloc(nk, sizeof(float));
    state->values = (float *)calloc(nv, sizeof(float));
    if (!state->keys || !state->values) {
        kvl_mla_state_free(state);
        return -1;
    }
    state->capacity = capacity;
    state->num_heads = cfg->num_heads;
    state->qk_nope_dim = cfg->qk_nope_dim;
    state->qk_rope_dim = cfg->qk_rope_dim;
    state->v_head_dim = cfg->v_head_dim;
    return 0;
}

void kvl_mla_state_reset(KvlMlaState *state) {
    if (state) state->len = 0;
}

void kvl_mla_state_free(KvlMlaState *state) {
    if (!state) return;
    free(state->keys);
    free(state->values);
    memset(state, 0, sizeof(*state));
}

size_t kvl_mla_state_bytes(const KvlMlaState *state) {
    if (!state || state->capacity <= 0) return 0;
    const size_t qd = (size_t)state->qk_nope_dim + (size_t)state->qk_rope_dim;
    return sizeof(*state) +
           (size_t)state->capacity * (size_t)state->num_heads * qd * sizeof(float) +
           (size_t)state->capacity * (size_t)state->num_heads *
               (size_t)state->v_head_dim * sizeof(float);
}

int kvl_mla_decode_bf16(float *out,
                        const float *x,
                        int position,
                        const KvlMlaConfig *cfg,
                        const KvlMlaBF16 *w,
                        KvlMlaState *state) {
    if (!out || !x || !cfg || !w || !state || !w->q_proj || !w->kv_a_proj ||
        !w->kv_a_norm || !w->kv_b_proj || !w->o_proj || position != state->len ||
        position < 0 || position >= state->capacity ||
        state->num_heads != cfg->num_heads || state->qk_nope_dim != cfg->qk_nope_dim ||
        state->qk_rope_dim != cfg->qk_rope_dim || state->v_head_dim != cfg->v_head_dim)
        return -1;

    const int H = cfg->hidden_size, NH = cfg->num_heads;
    const int DN = cfg->qk_nope_dim, DR = cfg->qk_rope_dim, DV = cfg->v_head_dim;
    const int QD = DN + DR, R = cfg->kv_lora_rank;
    const int QO = NH * QD, KVO = R + DR, KVB = NH * (DN + DV);
    if (H <= 0 || R <= 0 || DR <= 0 || (DR & 1) || cfg->rope_theta <= 0.0f)
        return -1;

    float *qtmp = (float *)malloc((size_t)QO * sizeof(float));
    float *katmp = (float *)malloc((size_t)KVO * sizeof(float));
    float *latent = (float *)malloc((size_t)R * sizeof(float));
    float *kvtmp = (float *)malloc((size_t)KVB * sizeof(float));
    float *rope = (float *)malloc((size_t)DR * sizeof(float));
    float *q = (float *)malloc((size_t)QD * sizeof(float));
    float *head_out = (float *)malloc((size_t)NH * DV * sizeof(float));
    float *scores = (float *)malloc((size_t)(position + 1) * sizeof(float));
    if (!qtmp || !katmp || !latent || !kvtmp || !rope || !q || !head_out || !scores) {
        free(qtmp); free(katmp); free(latent); free(kvtmp); free(rope);
        free(q); free(head_out); free(scores);
        return -1;
    }

    kvl_matvec_bf16(qtmp, x, w->q_proj, H, QO);
    kvl_matvec_bf16(katmp, x, w->kv_a_proj, H, KVO);
    kvl_rmsnorm_bf16(latent, katmp, w->kv_a_norm, R, cfg->rms_eps);
    kvl_matvec_bf16(kvtmp, latent, w->kv_b_proj, R, KVB);
    rope_interleaved_v6(rope, katmp + R, DR, position, cfg->rope_theta);

    for (int h = 0; h < NH; ++h) {
        const float *qh = qtmp + (size_t)h * QD;
        memcpy(q, qh, (size_t)DN * sizeof(float));
        rope_interleaved_v6(q + DN, qh + DN, DR, position, cfg->rope_theta);

        const float *kvh = kvtmp + (size_t)h * (DN + DV);
        float *kcur = state->keys + ((size_t)position * NH + h) * QD;
        float *vcur = state->values + ((size_t)position * NH + h) * DV;
        memcpy(kcur, kvh, (size_t)DN * sizeof(float));
        memcpy(kcur + DN, rope, (size_t)DR * sizeof(float));
        memcpy(vcur, kvh + DN, (size_t)DV * sizeof(float));

        const float scale = 1.0f / sqrtf((float)QD);
        float max_score = -INFINITY;
        for (int j = 0; j <= position; ++j) {
            const float *kj = state->keys + ((size_t)j * NH + h) * QD;
            double dot = 0.0;
            for (int d = 0; d < QD; ++d) dot += (double)q[d] * (double)kj[d];
            scores[j] = (float)dot * scale;
            if (scores[j] > max_score) max_score = scores[j];
        }
        double denom = 0.0;
        for (int j = 0; j <= position; ++j) {
            scores[j] = expf(scores[j] - max_score);
            denom += scores[j];
        }
        float *ho = head_out + (size_t)h * DV;
        for (int d = 0; d < DV; ++d) {
            double acc = 0.0;
            for (int j = 0; j <= position; ++j) {
                const float *vj = state->values + ((size_t)j * NH + h) * DV;
                acc += ((double)scores[j] / denom) * (double)vj[d];
            }
            ho[d] = (float)acc;
        }
    }

    kvl_matvec_bf16(out, head_out, w->o_proj, NH * DV, H);
    state->len = position + 1;

    free(qtmp); free(katmp); free(latent); free(kvtmp); free(rope);
    free(q); free(head_out); free(scores);
    return 0;
}
