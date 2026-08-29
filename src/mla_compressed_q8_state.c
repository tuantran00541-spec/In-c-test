#include "kvl/mla_compressed_q8_state.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_bf16_rne_q8(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    const uint32_t exp = u & UINT32_C(0x7f800000);
    if (exp != UINT32_C(0x7f800000))
        u += UINT32_C(0x00007fff) + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static void rope_interleaved_q8(float *dst, const float *raw, int dim,
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

static void quantize_latent_token(int8_t *dst, float *scale_out,
                                  const float *src, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    if (!(amax > 0.0f) || !isfinite(amax)) {
        memset(dst, 0, (size_t)n * sizeof(*dst));
        *scale_out = 1.0f;
        return;
    }
    const float scale = amax / 127.0f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        float qf = roundf(src[i] * inv);
        if (qf > 127.0f) qf = 127.0f;
        if (qf < -127.0f) qf = -127.0f;
        dst[i] = (int8_t)qf;
    }
    *scale_out = scale;
}

static float dequant_latent(const KvlMlaCompressedQ8State *state,
                            int token, int r) {
    return (float)state->latent_q8[(size_t)token * state->kv_lora_rank + r] *
           state->latent_scale[token];
}

int kvl_mla_compressed_q8_state_init(KvlMlaCompressedQ8State *state,
                                     const KvlMlaConfig *cfg,
                                     int capacity) {
    if (!state || !cfg || capacity <= 0 || cfg->kv_lora_rank <= 0 ||
        cfg->qk_rope_dim <= 0 || (cfg->qk_rope_dim & 1))
        return -1;
    memset(state, 0, sizeof(*state));
    state->latent_q8 = (int8_t *)calloc((size_t)capacity * cfg->kv_lora_rank,
                                        sizeof(int8_t));
    state->latent_scale = (float *)calloc((size_t)capacity, sizeof(float));
    state->rope_bf16 = (uint16_t *)calloc((size_t)capacity * cfg->qk_rope_dim,
                                          sizeof(uint16_t));
    if (!state->latent_q8 || !state->latent_scale || !state->rope_bf16) {
        kvl_mla_compressed_q8_state_free(state);
        return -1;
    }
    state->capacity = capacity;
    state->kv_lora_rank = cfg->kv_lora_rank;
    state->qk_rope_dim = cfg->qk_rope_dim;
    return 0;
}

void kvl_mla_compressed_q8_state_reset(KvlMlaCompressedQ8State *state) {
    if (state) state->len = 0;
}

int kvl_mla_compressed_q8_state_truncate(KvlMlaCompressedQ8State *state,
                                         int new_len) {
    if (!state || new_len < 0 || new_len > state->len) return -1;
    state->len = new_len;
    return 0;
}

void kvl_mla_compressed_q8_state_free(KvlMlaCompressedQ8State *state) {
    if (!state) return;
    free(state->latent_q8);
    free(state->latent_scale);
    free(state->rope_bf16);
    memset(state, 0, sizeof(*state));
}

size_t kvl_mla_compressed_q8_state_bytes(const KvlMlaCompressedQ8State *state) {
    if (!state || state->capacity <= 0) return 0;
    return sizeof(*state) + (size_t)state->capacity *
        ((size_t)state->kv_lora_rank * sizeof(int8_t) + sizeof(float) +
         (size_t)state->qk_rope_dim * sizeof(uint16_t));
}

static int append_q8_entry(const float *x, int position,
                           const KvlMlaConfig *cfg, const KvlMlaBF16 *w,
                           KvlMlaCompressedQ8State *state,
                           float *katmp, float *latent_tmp, float *rope_tmp) {
    const int H = cfg->hidden_size;
    const int R = cfg->kv_lora_rank;
    const int DR = cfg->qk_rope_dim;
    const int KVO = R + DR;
    kvl_matvec_bf16(katmp, x, w->kv_a_proj, H, KVO);
    kvl_rmsnorm_bf16(latent_tmp, katmp, w->kv_a_norm, R, cfg->rms_eps);
    quantize_latent_token(state->latent_q8 + (size_t)position * R,
                          state->latent_scale + position, latent_tmp, R);
    rope_interleaved_q8(rope_tmp, katmp + R, DR, position, cfg->rope_theta);
    for (int d = 0; d < DR; ++d)
        state->rope_bf16[(size_t)position * DR + d] = f32_to_bf16_rne_q8(rope_tmp[d]);
    return 0;
}

int kvl_mla_compressed_q8_state_prefill_bf16(const float *x,
                                             int seq_len,
                                             const KvlMlaConfig *cfg,
                                             const KvlMlaBF16 *w,
                                             KvlMlaCompressedQ8State *state) {
    if (!x || !cfg || !w || !state || !w->kv_a_proj || !w->kv_a_norm ||
        seq_len <= 0 || state->len != 0 || seq_len > state->capacity ||
        state->kv_lora_rank != cfg->kv_lora_rank ||
        state->qk_rope_dim != cfg->qk_rope_dim)
        return -1;
    const int H = cfg->hidden_size;
    const int R = cfg->kv_lora_rank;
    const int DR = cfg->qk_rope_dim;
    const int KVO = R + DR;
    if (H <= 0 || R <= 0 || DR <= 0 || (DR & 1) || cfg->rope_theta <= 0.0f)
        return -1;

    float *katmp = (float *)malloc((size_t)KVO * sizeof(float));
    float *latent_tmp = (float *)malloc((size_t)R * sizeof(float));
    float *rope_tmp = (float *)malloc((size_t)DR * sizeof(float));
    if (!katmp || !latent_tmp || !rope_tmp) {
        free(katmp); free(latent_tmp); free(rope_tmp);
        return -1;
    }
    for (int t = 0; t < seq_len; ++t)
        append_q8_entry(x + (size_t)t * H, t, cfg, w, state,
                        katmp, latent_tmp, rope_tmp);
    state->len = seq_len;
    free(katmp); free(latent_tmp); free(rope_tmp);
    return 0;
}

int kvl_mla_decode_compressed_q8_bf16(float *out,
                                      const float *x,
                                      int position,
                                      const KvlMlaConfig *cfg,
                                      const KvlMlaBF16 *w,
                                      KvlMlaCompressedQ8State *state) {
    if (!out || !x || !cfg || !w || !state || !w->q_proj || !w->kv_a_proj ||
        !w->kv_a_norm || !w->kv_b_proj || !w->o_proj || position != state->len ||
        position < 0 || position >= state->capacity ||
        state->kv_lora_rank != cfg->kv_lora_rank ||
        state->qk_rope_dim != cfg->qk_rope_dim)
        return -1;

    const int H = cfg->hidden_size, NH = cfg->num_heads;
    const int DN = cfg->qk_nope_dim, DR = cfg->qk_rope_dim;
    const int DV = cfg->v_head_dim, R = cfg->kv_lora_rank;
    const int QD = DN + DR, QO = NH * QD, KVO = R + DR;
    if (H <= 0 || NH <= 0 || DN <= 0 || DR <= 0 || (DR & 1) ||
        DV <= 0 || R <= 0 || cfg->rope_theta <= 0.0f)
        return -1;

    float *qtmp = (float *)malloc((size_t)QO * sizeof(float));
    float *katmp = (float *)malloc((size_t)KVO * sizeof(float));
    float *qrope = (float *)malloc((size_t)DR * sizeof(float));
    float *q_lat = (float *)malloc((size_t)R * sizeof(float));
    float *mix = (float *)malloc((size_t)R * sizeof(float));
    float *scores = (float *)malloc((size_t)(position + 1) * sizeof(float));
    float *head_out = (float *)malloc((size_t)NH * DV * sizeof(float));
    float *latent_tmp = (float *)malloc((size_t)R * sizeof(float));
    float *rope_tmp = (float *)malloc((size_t)DR * sizeof(float));
    if (!qtmp || !katmp || !qrope || !q_lat || !mix || !scores || !head_out ||
        !latent_tmp || !rope_tmp) {
        free(qtmp); free(katmp); free(qrope); free(q_lat); free(mix); free(scores);
        free(head_out); free(latent_tmp); free(rope_tmp);
        return -1;
    }

    kvl_matvec_bf16(qtmp, x, w->q_proj, H, QO);
    append_q8_entry(x, position, cfg, w, state, katmp, latent_tmp, rope_tmp);

    const float scale = 1.0f / sqrtf((float)QD);
    for (int h = 0; h < NH; ++h) {
        const float *qh = qtmp + (size_t)h * QD;
        rope_interleaved_q8(qrope, qh + DN, DR, position, cfg->rope_theta);

        for (int r = 0; r < R; ++r) {
            double acc = 0.0;
            for (int d = 0; d < DN; ++d) {
                const int row = h * (DN + DV) + d;
                acc += (double)kvl_bf16_to_f32(w->kv_b_proj[(size_t)row * R + r]) *
                       (double)qh[d];
            }
            q_lat[r] = (float)acc;
        }

        float max_score = -INFINITY;
        for (int j = 0; j <= position; ++j) {
            double dot = 0.0;
            for (int r = 0; r < R; ++r)
                dot += (double)q_lat[r] * (double)dequant_latent(state, j, r);
            for (int d = 0; d < DR; ++d)
                dot += (double)qrope[d] *
                       (double)kvl_bf16_to_f32(state->rope_bf16[(size_t)j * DR + d]);
            scores[j] = (float)dot * scale;
            if (scores[j] > max_score) max_score = scores[j];
        }
        double denom = 0.0;
        for (int j = 0; j <= position; ++j) {
            scores[j] = expf(scores[j] - max_score);
            denom += scores[j];
        }
        for (int r = 0; r < R; ++r) {
            double acc = 0.0;
            for (int j = 0; j <= position; ++j)
                acc += ((double)scores[j] / denom) *
                       (double)dequant_latent(state, j, r);
            mix[r] = (float)acc;
        }
        float *ho = head_out + (size_t)h * DV;
        for (int d = 0; d < DV; ++d) {
            const int row = h * (DN + DV) + DN + d;
            double acc = 0.0;
            for (int r = 0; r < R; ++r)
                acc += (double)kvl_bf16_to_f32(w->kv_b_proj[(size_t)row * R + r]) *
                       (double)mix[r];
            ho[d] = (float)acc;
        }
    }
    kvl_matvec_bf16(out, head_out, w->o_proj, NH * DV, H);
    state->len = position + 1;

    free(qtmp); free(katmp); free(qrope); free(q_lat); free(mix); free(scores);
    free(head_out); free(latent_tmp); free(rope_tmp);
    return 0;
}
