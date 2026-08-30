#include "kvl/ops.h"

#include <stdio.h>
#include <stdlib.h>

#define Q5_GROUP 128

static int read_exact(const char *path, void *buf, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    const size_t got = fread(buf, 1, bytes, f);
    const int extra = fgetc(f);
    fclose(f);
    return (got == bytes && extra == EOF) ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s q5.blob x.f32 in out y.f32\n", argv[0]);
        return 2;
    }
    const int in = atoi(argv[3]);
    const int out = atoi(argv[4]);
    if (in <= 0 || out <= 0) return 2;

    const size_t groups = ((size_t)in + Q5_GROUP - 1u) / Q5_GROUP;
    const size_t scale_bytes = (size_t)out * groups * sizeof(float);
    const size_t weight_bits = (size_t)out * (size_t)in * 5u;
    const size_t blob_bytes = scale_bytes + (weight_bits + 7u) / 8u;
    const size_t x_bytes = (size_t)in * sizeof(float);
    void *blob = malloc(blob_bytes);
    float *x = (float *)malloc(x_bytes);
    float *y = (float *)malloc((size_t)out * sizeof(float));
    if (!blob || !x || !y) return 2;
    if (read_exact(argv[1], blob, blob_bytes) || read_exact(argv[2], x, x_bytes)) return 2;

    kvl_matvec_q5_g128(y, x, blob, in, out);
    FILE *f = fopen(argv[5], "wb");
    if (!f) return 2;
    const size_t wrote = fwrite(y, sizeof(float), (size_t)out, f);
    fclose(f);
    if (wrote != (size_t)out) return 2;
    fprintf(stderr, "kvl_q5_probe: in=%d out=%d groups=%zu blob=%.3f MiB\n",
            in, out, groups, blob_bytes / 1048576.0);
    free(blob); free(x); free(y);
    return 0;
}
