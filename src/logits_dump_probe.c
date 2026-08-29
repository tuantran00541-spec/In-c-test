#include <math.h>
#include <stdint.h>
#include <stdio.h>

void kvl_matvec_bf16_dump(float *y, const float *x, const uint16_t *w,
                          int in, int out);

int main(void) {
    const float x[2] = {1.0f, 2.0f};
    const uint16_t w[6] = {
        0x3f80u, 0x0000u,
        0x0000u, 0x3f80u,
        0x3f80u, 0x3f80u,
    };
    float y[3] = {0.0f, 0.0f, 0.0f};
    kvl_matvec_bf16_dump(y, x, w, 2, 3);
    if (fabsf(y[0] - 1.0f) > 1e-6f ||
        fabsf(y[1] - 2.0f) > 1e-6f ||
        fabsf(y[2] - 3.0f) > 1e-6f) {
        fprintf(stderr, "logits dump probe matvec mismatch %.9g %.9g %.9g\n", y[0], y[1], y[2]);
        return 1;
    }
    puts("KVL_LOGITS_DUMP_PROBE_PASS");
    return 0;
}
