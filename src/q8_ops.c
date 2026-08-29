#include "kvl/ops.h"

#include <stddef.h>
#include <stdint.h>

#ifdef KVL_USE_AVX2
#include <immintrin.h>
#endif

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
    if (kvl_router_noaux_tc(router_cfg, x, router_weight, correction_bias,
                            top_ids, top_weights) != 0)
        return -1;

    (void)kvl_expert_cache_getmany(cache, layer, top_ids, router_cfg->top_k);
    for (int i = 0; i < H; ++i) out[i] = 0.0f;

    float *gate = scratch;
    float *up = gate + maxI;
    float *act = up + maxI;
    float *tmp = act + maxI;

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
        for (int i = 0; i < H; ++i) out[i] += w * tmp[i];
    }

    if (shared) {
        if (kvl_mlp_bf16(tmp, x, shared, H, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) out[i] += tmp[i];
    }
    return 0;
}
