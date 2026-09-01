#include "mla_prefill_token_parallel.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef KVL_USE_AVX2
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* Exact serial BF16 GEMV used inside token-parallel regions. Its per-row dot
 * order is copied from ops.c. The point is to parallelise many independent
 * prompt tokens with one OpenMP region rather than opening a region for every
 * small GEMV call. */
static float dot_bf16_serial(const uint16_t *row, const float *x, int n) {
#ifdef KVL_USE_AVX2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i w16 = _mm_loadu_si128((const __m128i *)(row + i));
        __m256i w32 = _mm256_cvtepu16_epi32(w16);
        w32 = _mm256_slli_epi32(w32, 16);
        const __m256 wf = _mm256_castsi256_ps(w32);
        const __m256 xv = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(wf, xv));
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float out = _mm_cvtss_f32(sum);
    for (; i < n; ++i) out += kvl_bf16_to_f32(row[i]) * x[i];
    return out;
#else
    double acc = 0.0;
    for (int i = 0; i < n; ++i)
        acc += (double)kvl_bf16_to_f32(row[i]) * (double)x[i];
    return (float)acc;
#endif
}

static void matvec_bf16_serial(float *y, const float *x, const uint16_t *w,
                               int in, int out) {
    for (int o = 0; o < out; ++o)
        y[o] = dot_bf16_serial(w + (size_t)o * (size_t)in, x, in);
}

static void rope_interleaved_tp(float *dst, const float *raw, int dim,
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

static double dot_f32_f64_tp(const float *a, const float *b, int n) {
#ifdef KVL_USE_AVX2
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128 af0 = _mm_loadu_ps(a + i);
        const __m128 bf0 = _mm_loadu_ps(b + i);
        const __m128 af1 = _mm_loadu_ps(a + i + 4);
        const __m128 bf1 = _mm_loadu_ps(b + i + 4);
        const __m256d ad0 = _mm256_cvtps_pd(af0);
        const __m256d bd0 = _mm256_cvtps_pd(bf0);
        const __m256d ad1 = _mm256_cvtps_pd(af1);
        const __m256d bd1 = _mm256_cvtps_pd(bf1);
        acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(ad0, bd0));
        acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(ad1, bd1));
    }
    const __m256d sumv = _mm256_add_pd(acc0, acc1);
    double lane[4];
    _mm256_storeu_pd(lane, sumv);
    double acc = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    for (; i < n; ++i) acc += (double)a[i] * (double)b[i];
    return acc;
#else
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double)a[i] * (double)b[i];
    return acc;
#endif
}

static void value_add_f64_tp(double *acc, const float *v, int n, double p) {
#ifdef KVL_USE_AVX2
    const __m256d pv = _mm256_set1_pd(p);
    int d = 0;
    for (; d + 4 <= n; d += 4) {
        const __m128 vf = _mm_loadu_ps(v + d);
        const __m256d vd = _mm256_cvtps_pd(vf);
        __m256d av = _mm256_loadu_pd(acc + d);
        av = _mm256_add_pd(av, _mm256_mul_pd(pv, vd));
        _mm256_storeu_pd(acc + d, av);
    }
    for (; d < n; ++d) acc[d] += p * (double)v[d];
#else
    for (int d = 0; d < n; ++d) acc[d] += p * (double)v[d];
#endif
}

int kvl_mla_prefill_compressed_token_parallel_bf16(float *out,
                                                    const float *x,
                                                    int seq_len,
                                                    const KvlMlaConfig *cfg,
                                                    const KvlMlaBF16 *w,
                                                    KvlMlaCompressedState *state) {
    if (!out || !x || !cfg || !w || !state || !w->q_proj || !w->kv_a_proj ||
        !w->kv_a_norm || !w->kv_b_proj || !w->o_proj || seq_len <= 0 ||
        cfg->hidden_size <= 0 || cfg->num_heads <= 0 || cfg->qk_nope_dim <= 0 ||
        cfg->qk_rope_dim <= 0 || (cfg->qk_rope_dim & 1) || cfg->v_head_dim <= 0 ||
        cfg->kv_lora_rank <= 0 || cfg->rope_theta <= 0.0f || state->len != 0 ||
        seq_len > state->capacity || state->kv_lora_rank != cfg->kv_lora_rank ||
        state->qk_rope_dim != cfg->qk_rope_dim || !state->latent || !state->rope)
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
    int worker_count = 1;
#ifdef _OPENMP
    worker_count = omp_get_max_threads();
    if (worker_count < 1) worker_count = 1;
    if (worker_count > S) worker_count = S;
#endif

    float *latent_states = state->latent;
    float *rope_states = state->rope;
    float *k_nope = (float *)malloc((size_t)S * DN * sizeof(float));
    float *v_states = (float *)malloc((size_t)S * DV * sizeof(float));
    float *scores_all = (float *)malloc((size_t)worker_count * S * sizeof(float));
    float *qtmp_all = (float *)malloc((size_t)worker_count * QD * sizeof(float));
    float *katmp_all = (float *)malloc((size_t)worker_count * KVO * sizeof(float));
    float *kvtmp_all = (float *)malloc((size_t)worker_count * KHV * sizeof(float));
    float *qrope_all = (float *)malloc((size_t)worker_count * DR * sizeof(float));
    float *head_tmp_all = (float *)malloc((size_t)worker_count * HO * sizeof(float));
    float *head_states = NULL;
    double *value_acc_all = (double *)malloc((size_t)worker_count * DV * sizeof(double));

    if (HO != H)
        head_states = (float *)malloc((size_t)S * HO * sizeof(float));
    float *head_seq = (HO == H) ? out : head_states;

    if (!k_nope || !v_states || !scores_all || !qtmp_all || !katmp_all ||
        !kvtmp_all || !qrope_all || !head_tmp_all || !value_acc_all || !head_seq) {
        free(k_nope); free(v_states); free(scores_all); free(qtmp_all);
        free(katmp_all); free(kvtmp_all); free(qrope_all); free(head_tmp_all);
        free(head_states); free(value_acc_all);
        return -1;
    }

    int t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(S >= 8) num_threads(worker_count)
#endif
    for (t = 0; t < S; ++t) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float *katmp = katmp_all + (size_t)tid * KVO;
        const float *xt = x + (size_t)t * H;
        float *latent = latent_states + (size_t)t * R;
        float *rope = rope_states + (size_t)t * DR;
        matvec_bf16_serial(katmp, xt, w->kv_a_proj, H, KVO);
        kvl_rmsnorm_bf16(latent, katmp, w->kv_a_norm, R, cfg->rms_eps);
        rope_interleaved_tp(rope, katmp + R, DR, t, cfg->rope_theta);
    }

    const float scale = 1.0f / sqrtf((float)QD);
    for (int h = 0; h < NH; ++h) {
        const uint16_t *kv_head = w->kv_b_proj + (size_t)h * KHV * R;
        int j;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(S >= 8) num_threads(worker_count)
#endif
        for (j = 0; j < S; ++j) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *kvtmp = kvtmp_all + (size_t)tid * KHV;
            const float *latent = latent_states + (size_t)j * R;
            matvec_bf16_serial(kvtmp, latent, kv_head, R, KHV);
            memcpy(k_nope + (size_t)j * DN, kvtmp, (size_t)DN * sizeof(float));
            memcpy(v_states + (size_t)j * DV, kvtmp + DN, (size_t)DV * sizeof(float));
        }

        const uint16_t *q_head = w->q_proj + (size_t)h * QD * H;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8) if(S >= 8) num_threads(worker_count)
#endif
        for (t = 0; t < S; ++t) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *scores = scores_all + (size_t)tid * S;
            float *qtmp = qtmp_all + (size_t)tid * QD;
            float *qrope = qrope_all + (size_t)tid * DR;
            double *value_acc = value_acc_all + (size_t)tid * DV;

            matvec_bf16_serial(qtmp, x + (size_t)t * H, q_head, H, QD);
            rope_interleaved_tp(qrope, qtmp + DN, DR, t, cfg->rope_theta);

            float max_score = -INFINITY;
            for (int jj = 0; jj <= t; ++jj) {
                const float *kj = k_nope + (size_t)jj * DN;
                const float *rj = rope_states + (size_t)jj * DR;
                const double dot = dot_f32_f64_tp(qtmp, kj, DN) +
                                   dot_f32_f64_tp(qrope, rj, DR);
                scores[jj] = (float)dot * scale;
                if (scores[jj] > max_score) max_score = scores[jj];
            }

            double denom = 0.0;
            for (int jj = 0; jj <= t; ++jj) {
                scores[jj] = expf(scores[jj] - max_score);
                denom += scores[jj];
            }

            memset(value_acc, 0, (size_t)DV * sizeof(double));
            for (int jj = 0; jj <= t; ++jj) {
                const double p = (double)scores[jj] / denom;
                const float *vj = v_states + (size_t)jj * DV;
                value_add_f64_tp(value_acc, vj, DV, p);
            }
            float *ho = head_seq + (size_t)t * HO + (size_t)h * DV;
            for (int d = 0; d < DV; ++d) ho[d] = (float)value_acc[d];
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(S >= 8) num_threads(worker_count)
#endif
    for (t = 0; t < S; ++t) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float *head_tmp = head_tmp_all + (size_t)tid * HO;
        const float *hs = head_seq + (size_t)t * HO;
        memcpy(head_tmp, hs, (size_t)HO * sizeof(float));
        matvec_bf16_serial(out + (size_t)t * H, head_tmp, w->o_proj, HO, H);
    }

    state->len = S;
    free(k_nope); free(v_states); free(scores_all); free(qtmp_all);
    free(katmp_all); free(kvtmp_all); free(qrope_all); free(head_tmp_all);
    free(head_states); free(value_acc_all);
    return 0;
}
