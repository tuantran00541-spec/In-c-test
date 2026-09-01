#include "kvl/ops.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef KVL_USE_AVX2
#include <immintrin.h>
#endif

/* Research scheduler for packed row-wise Q8 experts.
 *
 * The production implementation parallelises rows inside each of gate/up/down.
 * Kimi routes top-6 experts, so that creates many short OpenMP regions. This
 * candidate instead resolves/touches the six cache entries serially, computes
 * the six experts in one outer parallel region, then reduces expert outputs in
 * the original router order. The reduction order therefore stays identical.
 *
 * Scratch contract for this research function is stronger than the generic MoE
 * API: scratch must hold top_k * (2*I + H) floats. Kimi's generator already
 * allocates 3*DENSE_I+H = 35,840 floats; top-6 needs only 29,184 floats.
 */

static float q8_dot_ref(const int8_t *row, const float *x, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double)row[i] * (double)x[i];
    return (float)acc;
}

#ifdef KVL_USE_AVX2
static float q8_dot_avx2(const int8_t *row, const float *x, int n) {
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

static void q8_matvec_serial(float *y, const float *x, const void *blob,
                             int in, int out) {
    const float *scales = (const float *)blob;
    const int8_t *q = (const int8_t *)(scales + out);
    for (int o = 0; o < out; ++o) {
        const int8_t *row = q + (size_t)o * (size_t)in;
#ifdef KVL_USE_AVX2
        const float dot = q8_dot_avx2(row, x, in);
#else
        const float dot = q8_dot_ref(row, x, in);
#endif
        y[o] = scales[o] * dot;
    }
}

static int research_fallback_requested(void) {
    const char *mask = getenv("KVL_MOE_MASK");
    const char *trace = getenv("KVL_MOE_TRACE");
    const char *profile = getenv("KVL_MOE_PROFILE_TRACE");
    return (mask && mask[0]) || (trace && trace[0]) || (profile && profile[0]);
}

int kvl_moe_token_q8_expert_parallel_auto(KvlExpertCache *cache, int layer,
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
        !scratch || expert_intermediate_size <= 0 || router_cfg->top_k <= 0 ||
        router_cfg->top_k > 64)
        return -1;

    /* Mask/trace paths carry extra semantics. Keep them on the established
     * implementation rather than making the scheduler experiment broader. */
    if (research_fallback_requested())
        return kvl_moe_token_auto(cache, layer, router_cfg, x, router_weight,
                                  correction_bias, expert_intermediate_size, shared,
                                  out, top_ids, top_weights, scratch);

    const int H = router_cfg->hidden_size;
    const int I = expert_intermediate_size;
    const int K = router_cfg->top_k;
    const int maxI = (shared && shared->intermediate_size > I)
        ? shared->intermediate_size : I;
    if (H <= 0 || I <= 0) return -1;

    if (kvl_router_noaux_tc(router_cfg, x, router_weight, correction_bias,
                            top_ids, top_weights) != 0)
        return -1;

    (void)kvl_expert_cache_getmany(cache, layer, top_ids, K);

    /* Resolve serially before the compute region. Besides avoiding cache data
     * races, this preserves the production request/hit/LRU touch order. */
    KvlCachedExpert routed[64];
    const size_t need_gu = (size_t)I * sizeof(float) + (size_t)I * (size_t)H;
    const size_t need_dn = (size_t)H * sizeof(float) + (size_t)H * (size_t)I;
    for (int j = 0; j < K; ++j) {
        if (kvl_expert_cache_get(cache, layer, top_ids[j], &routed[j]) != 0)
            return -1;
        if (routed[j].record->gate_bytes != need_gu ||
            routed[j].record->up_bytes != need_gu ||
            routed[j].record->down_bytes != need_dn)
            return -1;
    }

    const size_t lane_stride = (size_t)2 * (size_t)I + (size_t)H;
    int failed = 0;
    int j;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for (j = 0; j < K; ++j) {
        float *lane = scratch + (size_t)j * lane_stride;
        float *gate = lane;
        float *up = gate + I;
        float *tmp = up + I;
        const KvlCachedExpert *q = &routed[j];

        q8_matvec_serial(gate, x, q->gate, H, I);
        q8_matvec_serial(up, x, q->up, H, I);
        /* Safe alias: kvl_silu_mul reads gate[i] into a local before writing y[i]. */
        kvl_silu_mul(gate, gate, up, I);
        q8_matvec_serial(tmp, gate, q->down, I, H);
        if (!q->record) failed |= 1;
    }
    if (failed) return -1;

    /* Same accumulation order as production: j=0..top_k-1, then shared. */
    for (int i = 0; i < H; ++i) out[i] = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float *tmp = scratch + (size_t)k * lane_stride + (size_t)2 * I;
        const float w = top_weights[k];
        for (int i = 0; i < H; ++i) out[i] += w * tmp[i];
    }

    if (shared) {
        float *shared_tmp = scratch + (size_t)3 * (size_t)maxI;
        if (kvl_mlp_bf16(shared_tmp, x, shared, H, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) out[i] += shared_tmp[i];
    }
    return 0;
}
