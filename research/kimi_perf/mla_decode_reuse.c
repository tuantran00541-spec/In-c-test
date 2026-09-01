#include "mla_decode_reuse.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/* Single-engine research workspace. kvl_generate executes decoder layers
 * serially, so one process-local arena can be reused across all 27 layers and
 * output tokens. This candidate changes only allocation lifetime; arithmetic
 * and loop order below are copied from kvl_mla_decode_compressed_bf16. */
static float *g_ws = NULL;
static size_t g_ws_floats = 0;
static int g_atexit_registered = 0;

void kvl_mla_decode_reuse_release(void) {
    free(g_ws);
    g_ws = NULL;
    g_ws_floats = 0;
}

static int ensure_workspace(size_t need) {
    if (need <= g_ws_floats) return 0;
    float *p = (float *)realloc(g_ws, need * sizeof(float));
    if (!p) return -1;
    g_ws = p;
    g_ws_floats = need;
    if (!g_atexit_registered) {
        if (atexit(kvl_mla_decode_reuse_release) != 0) return -1;
        g_atexit_registered = 1;
    }
    return 0;
}

static void rope_interleaved_reuse(float *dst, const float *raw, int dim,
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

int kvl_mla_decode_compressed_reuse_bf16(float *out,
                                          const float *x,
                                          int position,
                                          const KvlMlaConfig *cfg,
                                          const KvlMlaBF16 *w,
                                          KvlMlaCompressedState *state) {
    if (!out || !x || !cfg || !w || !state || !w->q_proj || !w->kv_a_proj ||
        !w->kv_a_norm || !w->kv_b_proj || !w->o_proj || position != state->len ||
        position < 0 || position >= state->capacity ||
        state->kv_lora_rank != cfg->kv_lora_rank ||
        state->qk_rope_dim != cfg->qk_rope_dim)
        return -1;

    const int H = cfg->hidden_size;
    const int NH = cfg->num_heads;
    const int DN = cfg->qk_nope_dim;
    const int DR = cfg->qk_rope_dim;
    const int DV = cfg->v_head_dim;
    const int R = cfg->kv_lora_rank;
    const int QD = DN + DR;
    const int QO = NH * QD;
    const int KVO = R + DR;
    if (H <= 0 || NH <= 0 || DN <= 0 || DR <= 0 || (DR & 1) || DV <= 0 ||
        R <= 0 || cfg->rope_theta <= 0.0f)
        return -1;

    const size_t need = (size_t)QO + KVO + DR + R + R +
                        (size_t)state->capacity + (size_t)NH * DV;
    if (ensure_workspace(need) != 0) return -1;

    float *p = g_ws;
    float *qtmp = p; p += QO;
    float *katmp = p; p += KVO;
    float *qrope = p; p += DR;
    float *q_lat = p; p += R;
    float *mix = p; p += R;
    float *scores = p; p += state->capacity;
    float *head_out = p;

    kvl_matvec_bf16(qtmp, x, w->q_proj, H, QO);
    kvl_matvec_bf16(katmp, x, w->kv_a_proj, H, KVO);
    float *cur_lat = state->latent + (size_t)position * R;
    float *cur_rope = state->rope + (size_t)position * DR;
    kvl_rmsnorm_bf16(cur_lat, katmp, w->kv_a_norm, R, cfg->rms_eps);
    rope_interleaved_reuse(cur_rope, katmp + R, DR, position, cfg->rope_theta);

    const float scale = 1.0f / sqrtf((float)QD);
    for (int h = 0; h < NH; ++h) {
        const float *qh = qtmp + (size_t)h * QD;
        rope_interleaved_reuse(qrope, qh + DN, DR, position, cfg->rope_theta);

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
            const float *lj = state->latent + (size_t)j * R;
            const float *rj = state->rope + (size_t)j * DR;
            double dot = 0.0;
            for (int r = 0; r < R; ++r) dot += (double)q_lat[r] * (double)lj[r];
            for (int d = 0; d < DR; ++d) dot += (double)qrope[d] * (double)rj[d];
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
                       (double)state->latent[(size_t)j * R + r];
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
    return 0;
}
