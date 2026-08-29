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
