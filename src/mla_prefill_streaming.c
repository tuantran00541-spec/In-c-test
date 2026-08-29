#include "kvl/ops.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Long-context exact prefill path.
 *
 * The legacy implementation materializes Q, K and V for every token/head. This
 * implementation keeps only the shared MLA latent+RoPE prompt state plus one
 * head's K(no-PE)+V at a time. Query vectors are projected per token/head.
 * Attention math and causal masking remain dense and exact; this reduces the
 * temporary working set but intentionally does not claim sub-quadratic compute.
 */
static void rope_interleaved_stream(float *dst, const float *raw, int dim,
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

    const int S = seq_len;
    const int H = cfg->hidden_size;
    const int NH = cfg->num_heads;
    const int DN = cfg->qk_nope_dim;
    const int DR = cfg->qk_rope_dim;
    const int DV = cfg->v_head_dim;
    const int R = cfg->kv_lora_rank;
    const int QD = DN + DR;
    const int KVO = R + DR;
    const int KHV = DN + DV;
    const int HO = NH * DV;
    if (HO != H) return -1;

    float *latent_states = (float *)malloc((size_t)S * R * sizeof(float));
    float *rope_states = (float *)malloc((size_t)S * DR * sizeof(float));
    float *k_nope = (float *)malloc((size_t)S * DN * sizeof(float));
    float *v_states = (float *)malloc((size_t)S * DV * sizeof(float));
    float *scores = (float *)malloc((size_t)S * sizeof(float));
    float *qtmp = (float *)malloc((size_t)QD * sizeof(float));
    float *katmp = (float *)malloc((size_t)KVO * sizeof(float));
    float *kvtmp = (float *)malloc((size_t)KHV * sizeof(float));
    float *qrope = (float *)malloc((size_t)DR * sizeof(float));
    float *head_tmp = (float *)malloc((size_t)H * sizeof(float));
    double *value_acc = (double *)malloc((size_t)DV * sizeof(double));
    if (!latent_states || !rope_states || !k_nope || !v_states || !scores ||
        !qtmp || !katmp || !kvtmp || !qrope || !head_tmp || !value_acc) {
        free(latent_states); free(rope_states); free(k_nope); free(v_states);
        free(scores); free(qtmp); free(katmp); free(kvtmp); free(qrope);
        free(head_tmp); free(value_acc);
        return -1;
    }

    /* Shared compressed prompt representation. It is independent of head, so
     * compute it once instead of repeating kv_a projection for each head. */
    for (int t = 0; t < S; ++t) {
        const float *xt = x + (size_t)t * H;
        float *latent = latent_states + (size_t)t * R;
        float *rope = rope_states + (size_t)t * DR;
        kvl_matvec_bf16(katmp, xt, w->kv_a_proj, H, KVO);
        kvl_rmsnorm_bf16(latent, katmp, w->kv_a_norm, R, cfg->rms_eps);
        rope_interleaved_stream(rope, katmp + R, DR, t, cfg->rope_theta);
    }

    const float scale = 1.0f / sqrtf((float)QD);
    for (int h = 0; h < NH; ++h) {
        /* kv_b rows for one head are contiguous: [DN no-PE key, DV value]. */
        const uint16_t *kv_head = w->kv_b_proj + (size_t)h * KHV * R;
        for (int j = 0; j < S; ++j) {
            const float *latent = latent_states + (size_t)j * R;
            kvl_matvec_bf16(kvtmp, latent, kv_head, R, KHV);
            memcpy(k_nope + (size_t)j * DN, kvtmp, (size_t)DN * sizeof(float));
            memcpy(v_states + (size_t)j * DV, kvtmp + DN, (size_t)DV * sizeof(float));
        }

        const uint16_t *q_head = w->q_proj + (size_t)h * QD * H;
        for (int t = 0; t < S; ++t) {
            kvl_matvec_bf16(qtmp, x + (size_t)t * H, q_head, H, QD);
            rope_interleaved_stream(qrope, qtmp + DN, DR, t, cfg->rope_theta);

            float max_score = -INFINITY;
            for (int j = 0; j <= t; ++j) {
                double dot = 0.0;
                const float *kj = k_nope + (size_t)j * DN;
                const float *rj = rope_states + (size_t)j * DR;
                for (int d = 0; d < DN; ++d)
                    dot += (double)qtmp[d] * (double)kj[d];
                for (int d = 0; d < DR; ++d)
                    dot += (double)qrope[d] * (double)rj[d];
                scores[j] = (float)dot * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            double denom = 0.0;
            for (int j = 0; j <= t; ++j) {
                scores[j] = expf(scores[j] - max_score);
                denom += scores[j];
            }

            memset(value_acc, 0, (size_t)DV * sizeof(double));
            for (int j = 0; j <= t; ++j) {
                const double p = (double)scores[j] / denom;
                const float *vj = v_states + (size_t)j * DV;
                for (int d = 0; d < DV; ++d)
                    value_acc[d] += p * (double)vj[d];
            }
            float *ho = out + (size_t)t * H + (size_t)h * DV;
            for (int d = 0; d < DV; ++d) ho[d] = (float)value_acc[d];
        }
    }

    /* `out` currently stores concatenated head outputs. Preserve the input to
     * o_proj with one token-sized copy so input/output never alias. */
    for (int t = 0; t < S; ++t) {
        float *ot = out + (size_t)t * H;
        memcpy(head_tmp, ot, (size_t)H * sizeof(float));
        kvl_matvec_bf16(ot, head_tmp, w->o_proj, H, H);
    }

    free(latent_states); free(rope_states); free(k_nope); free(v_states);
    free(scores); free(qtmp); free(katmp); free(kvtmp); free(qrope);
    free(head_tmp); free(value_acc);
    return 0;
}
