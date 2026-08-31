#include "kvl/ops.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

void kvl_matvec_ggml_q8_0(float *y, const float *x, const void *blob,
                          int in, int out) {
    if (!y || !x || !blob || in <= 0 || out <= 0 ||
        (in % KVL_GGML_Q8_0_BLOCK) != 0)
        return;

    const uint8_t *bytes = (const uint8_t *)blob;
    const int blocks_per_row = in / KVL_GGML_Q8_0_BLOCK;
    const size_t row_bytes = (size_t)blocks_per_row * KVL_GGML_Q8_0_BYTES;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(out > 64)
#endif
    for (int o = 0; o < out; ++o) {
        const uint8_t *row = bytes + (size_t)o * row_bytes;
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
        y[o] = (float)acc;
    }
}
