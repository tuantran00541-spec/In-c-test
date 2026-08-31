#include "kvl/ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define H 2048
#define I 1408
#define Q8_0_BLOCK 32
#define Q8_0_BYTES 34

static int read_exact(const char *path, void *buf, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    const size_t got = fread(buf, 1, bytes, f);
    const int extra = fgetc(f);
    fclose(f);
    return (got == bytes && extra == EOF) ? 0 : -1;
}

static size_t q8_0_matrix_bytes(int in, int out) {
    return (size_t)out * (size_t)(in / Q8_0_BLOCK) * Q8_0_BYTES;
}

static int compare_vec(const char *name, const float *got, const float *ref, int n,
                       double *max_abs_out, double *rms_out) {
    double max_abs = 0.0;
    double sq = 0.0;
    double ref_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(got[i]) || !isfinite(ref[i])) {
            fprintf(stderr, "%s: non-finite at %d got=%g ref=%g\n", name, i, got[i], ref[i]);
            return -1;
        }
        const double d = (double)got[i] - (double)ref[i];
        const double a = fabs(d);
        if (a > max_abs) max_abs = a;
        sq += d * d;
        ref_sq += (double)ref[i] * (double)ref[i];
    }
    const double rms = sqrt(sq / (double)n);
    const double ref_rms = sqrt(ref_sq / (double)n);
    fprintf(stderr, "KIMI_GGUF_Q8_0_MATRIX name=%s n=%d max_abs=%.9g rms=%.9g ref_rms=%.9g\n",
            name, n, max_abs, rms, ref_rms);
    if (max_abs_out) *max_abs_out = max_abs;
    if (rms_out) *rms_out = rms;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr,
                "usage: %s gate.q8 up.q8 down.q8 x_h.f32 x_i.f32 gate_ref.f32 up_ref.f32 down_ref.f32\n",
                argv[0]);
        return 2;
    }

    const size_t gate_bytes = q8_0_matrix_bytes(H, I);
    const size_t up_bytes = q8_0_matrix_bytes(H, I);
    const size_t down_bytes = q8_0_matrix_bytes(I, H);
    void *gate_blob = malloc(gate_bytes);
    void *up_blob = malloc(up_bytes);
    void *down_blob = malloc(down_bytes);
    float *xh = (float *)malloc((size_t)H * sizeof(float));
    float *xi = (float *)malloc((size_t)I * sizeof(float));
    float *gate_ref = (float *)malloc((size_t)I * sizeof(float));
    float *up_ref = (float *)malloc((size_t)I * sizeof(float));
    float *down_ref = (float *)malloc((size_t)H * sizeof(float));
    float *gate = (float *)malloc((size_t)I * sizeof(float));
    float *up = (float *)malloc((size_t)I * sizeof(float));
    float *down = (float *)malloc((size_t)H * sizeof(float));

    if (!gate_blob || !up_blob || !down_blob || !xh || !xi || !gate_ref || !up_ref ||
        !down_ref || !gate || !up || !down) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    if (read_exact(argv[1], gate_blob, gate_bytes) ||
        read_exact(argv[2], up_blob, up_bytes) ||
        read_exact(argv[3], down_blob, down_bytes) ||
        read_exact(argv[4], xh, (size_t)H * sizeof(float)) ||
        read_exact(argv[5], xi, (size_t)I * sizeof(float)) ||
        read_exact(argv[6], gate_ref, (size_t)I * sizeof(float)) ||
        read_exact(argv[7], up_ref, (size_t)I * sizeof(float)) ||
        read_exact(argv[8], down_ref, (size_t)H * sizeof(float))) {
        fprintf(stderr, "fixture read failed\n");
        return 2;
    }

    kvl_matvec_ggml_q8_0(gate, xh, gate_blob, H, I);
    kvl_matvec_ggml_q8_0(up, xh, up_blob, H, I);
    kvl_matvec_ggml_q8_0(down, xi, down_blob, I, H);

    double max_gate = 0.0, max_up = 0.0, max_down = 0.0;
    double rms_gate = 0.0, rms_up = 0.0, rms_down = 0.0;
    if (compare_vec("gate", gate, gate_ref, I, &max_gate, &rms_gate) ||
        compare_vec("up", up, up_ref, I, &max_up, &rms_up) ||
        compare_vec("down", down, down_ref, H, &max_down, &rms_down))
        return 1;

    const double max_abs = fmax(max_gate, fmax(max_up, max_down));
    const double max_rms = fmax(rms_gate, fmax(rms_up, rms_down));
    fprintf(stderr,
            "KIMI_GGUF_Q8_0_NATIVE_SMOKE max_abs=%.9g max_rms=%.9g gate_bytes=%zu up_bytes=%zu down_bytes=%zu\n",
            max_abs, max_rms, gate_bytes, up_bytes, down_bytes);

    /* The independent Python oracle uses the pinned llama.cpp Q8_0 dequantizer
     * and float64 dot products. This tolerance is intentionally tight enough to
     * catch block layout/FP16/axis mistakes while allowing harmless accumulation
     * order differences. */
    if (max_abs > 2e-4 || max_rms > 2e-5) {
        fprintf(stderr, "KIMI_GGUF_Q8_0_NATIVE_SMOKE_FAIL\n");
        return 1;
    }
    fprintf(stderr, "KIMI_GGUF_Q8_0_NATIVE_SMOKE_PASS\n");

    free(gate_blob); free(up_blob); free(down_blob);
    free(xh); free(xi); free(gate_ref); free(up_ref); free(down_ref);
    free(gate); free(up); free(down);
    return 0;
}
