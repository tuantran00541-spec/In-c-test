#include "kvl/ops.h"

#include <stddef.h>
#include <stdint.h>

#ifdef KVL_USE_AVX2
#include <immintrin.h>
#endif

#define KVL_Q5_GROUP 128

static int q5_sign5(unsigned v) {
    v &= 31u;
    return (v & 16u) ? (int)v - 32 : (int)v;
}

static int q5_read5(const unsigned char *q, size_t bitpos) {
    const size_t byte = bitpos >> 3;
    const unsigned shift = (unsigned)(bitpos & 7u);
    unsigned v = (unsigned)q[byte] >> shift;
    if (shift > 3u) v |= (unsigned)q[byte + 1] << (8u - shift);
    return q5_sign5(v);
}

static float dot_q5_ref(const unsigned char *q, size_t bitpos,
                        const float *x, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i)
        acc += (double)q5_read5(q, bitpos + (size_t)i * 5u) * (double)x[i];
    return (float)acc;
}

#ifdef KVL_USE_AVX2
static float dot_q5_byte_aligned_avx2(const unsigned char *q,
                                      const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    const unsigned char *p = q;
    for (; i + 8 <= n; i += 8, p += 5) {
        const uint64_t w =
            (uint64_t)p[0] |
            ((uint64_t)p[1] << 8) |
            ((uint64_t)p[2] << 16) |
            ((uint64_t)p[3] << 24) |
            ((uint64_t)p[4] << 32);
        const __m256i qi = _mm256_setr_epi32(
            q5_sign5((unsigned)(w >> 0)),
            q5_sign5((unsigned)(w >> 5)),
            q5_sign5((unsigned)(w >> 10)),
            q5_sign5((unsigned)(w >> 15)),
            q5_sign5((unsigned)(w >> 20)),
            q5_sign5((unsigned)(w >> 25)),
            q5_sign5((unsigned)(w >> 30)),
            q5_sign5((unsigned)(w >> 35)));
        const __m256 qf = _mm256_cvtepi32_ps(qi);
        const __m256 xv = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(qf, xv));
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float out = _mm_cvtss_f32(sum);
    if (i < n) {
        const size_t tail_bit = (size_t)i * 5u;
        out += dot_q5_ref(q, tail_bit, x + i, n - i);
    }
    return out;
}
#endif

static size_t q5_matrix_bytes(int in, int out) {
    const size_t groups = ((size_t)in + KVL_Q5_GROUP - 1u) / KVL_Q5_GROUP;
    const size_t scale_bytes = (size_t)out * groups * sizeof(float);
    const size_t weight_bits = (size_t)out * (size_t)in * 5u;
    return scale_bytes + (weight_bits + 7u) / 8u;
}

void kvl_matvec_q5_g128(float *y, const float *x, const void *blob,
                        int in, int out) {
    if (!y || !x || !blob || in <= 0 || out <= 0) return;
    const int groups = (in + KVL_Q5_GROUP - 1) / KVL_Q5_GROUP;
    const float *scales = (const float *)blob;
    const unsigned char *q = (const unsigned char *)(scales + (size_t)out * groups);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(out > 64)
#endif
    for (int o = 0; o < out; ++o) {
        float total = 0.0f;
        for (int g = 0; g < groups; ++g) {
            const int start = g * KVL_Q5_GROUP;
            const int n = (start + KVL_Q5_GROUP <= in) ? KVL_Q5_GROUP : in - start;
            const size_t bitpos = ((size_t)o * (size_t)in + (size_t)start) * 5u;
            float dot;
#ifdef KVL_USE_AVX2
            if ((bitpos & 7u) == 0u)
                dot = dot_q5_byte_aligned_avx2(q + (bitpos >> 3), x + start, n);
            else
                dot = dot_q5_ref(q, bitpos, x + start, n);
#else
            dot = dot_q5_ref(q, bitpos, x + start, n);
#endif
            total += scales[(size_t)o * groups + g] * dot;
        }
        y[o] = total;
    }
}

int kvl_moe_token_q5_auto(KvlExpertCache *cache, int layer,
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
    if (cache->store->hdr.dtype != KVL_DTYPE_Q5_G128 || !router_cfg || !x ||
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

    const size_t need_gu = q5_matrix_bytes(H, I);
    const size_t need_dn = q5_matrix_bytes(I, H);
    for (int j = 0; j < router_cfg->top_k; ++j) {
        KvlCachedExpert qx;
        if (kvl_expert_cache_get(cache, layer, top_ids[j], &qx) != 0) return -1;
        if (qx.record->gate_bytes != need_gu || qx.record->up_bytes != need_gu ||
            qx.record->down_bytes != need_dn)
            return -1;
        kvl_matvec_q5_g128(gate, x, qx.gate, H, I);
        kvl_matvec_q5_g128(up, x, qx.up, H, I);
        kvl_silu_mul(act, gate, up, I);
        kvl_matvec_q5_g128(tmp, act, qx.down, I, H);
        const float w = top_weights[j];
        for (int i = 0; i < H; ++i) out[i] += w * tmp[i];
    }

    if (shared) {
        if (kvl_mlp_bf16(tmp, x, shared, H, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) out[i] += tmp[i];
    }
    return 0;
}
