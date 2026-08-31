#include "kvl/ops.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef KVL_USE_AVX2
#include <immintrin.h>
#endif

#define KVL_GGML_Q8_0_BLOCK 32
#define KVL_GGML_Q8_0_BYTES 34

static float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

static float q8_0_block_scale(const uint8_t *block) {
    const uint16_t h = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
    return f16_bits_to_f32(h);
}

static float q8_0_row_ref(const uint8_t *row, const float *x, int blocks_per_row) {
    double acc = 0.0;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t *block = row + (size_t)b * KVL_GGML_Q8_0_BYTES;
        const float d = q8_0_block_scale(block);
        const int8_t *q = (const int8_t *)(block + 2);
        const float *xb = x + (size_t)b * KVL_GGML_Q8_0_BLOCK;
        double dot = 0.0;
        for (int i = 0; i < KVL_GGML_Q8_0_BLOCK; ++i)
            dot += (double)q[i] * (double)xb[i];
        acc += (double)d * dot;
    }
    return (float)acc;
}

#ifdef KVL_USE_AVX2
static float hsum256_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/* GGML Q8_0 stores one FP16 scale followed by 32 signed int8 values per block.
 * Convert all 32 bytes with AVX2, multiply against four FP32 x vectors, then apply
 * the block scale once to the vector block sum. Two row accumulators break the
 * dependency chain across the many blocks in Kimi expert rows. As with the BF16
 * AVX2 backend, accumulation is intentionally FP32 and must stay regression-tested
 * against the scalar double-accumulation reference and real prompt gates. */
static float q8_0_row_avx2(const uint8_t *row, const float *x, int blocks_per_row) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t *block = row + (size_t)b * KVL_GGML_Q8_0_BYTES;
        const float d = q8_0_block_scale(block);
        const int8_t *q = (const int8_t *)(block + 2);
        const float *xb = x + (size_t)b * KVL_GGML_Q8_0_BLOCK;

        const __m256i packed = _mm256_loadu_si256((const __m256i *)q);
        const __m128i qlo = _mm256_castsi256_si128(packed);
        const __m128i qhi = _mm256_extracti128_si256(packed, 1);

        const __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qlo));
        const __m256 q1 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(qlo, 8)));
        const __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qhi));
        const __m256 q3 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(qhi, 8)));

        const __m256 p0 = _mm256_mul_ps(q0, _mm256_loadu_ps(xb));
        const __m256 p1 = _mm256_mul_ps(q1, _mm256_loadu_ps(xb + 8));
        const __m256 p2 = _mm256_mul_ps(q2, _mm256_loadu_ps(xb + 16));
        const __m256 p3 = _mm256_mul_ps(q3, _mm256_loadu_ps(xb + 24));
        const __m256 block_sum = _mm256_add_ps(_mm256_add_ps(p0, p1),
                                               _mm256_add_ps(p2, p3));
        const __m256 scaled = _mm256_mul_ps(block_sum, _mm256_set1_ps(d));
        if (b & 1)
            acc1 = _mm256_add_ps(acc1, scaled);
        else
            acc0 = _mm256_add_ps(acc0, scaled);
    }

    return hsum256_ps(_mm256_add_ps(acc0, acc1));
}
#endif

void kvl_matvec_ggml_q8_0(float *y, const float *x, const void *blob,
                          int in, int out) {
    if (!y || !x || !blob || in <= 0 || out <= 0 ||
        (in % KVL_GGML_Q8_0_BLOCK) != 0)
        return;

    const uint8_t *bytes = (const uint8_t *)blob;
    const int blocks_per_row = in / KVL_GGML_Q8_0_BLOCK;
    const size_t row_bytes = (size_t)blocks_per_row * KVL_GGML_Q8_0_BYTES;
    int o;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(out > 64)
#endif
    for (o = 0; o < out; ++o) {
        const uint8_t *row = bytes + (size_t)o * row_bytes;
#ifdef KVL_USE_AVX2
        y[o] = q8_0_row_avx2(row, x, blocks_per_row);
#else
        y[o] = q8_0_row_ref(row, x, blocks_per_row);
#endif
    }
}
