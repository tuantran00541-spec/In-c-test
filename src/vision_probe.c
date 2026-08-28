#include "kvl/vision.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int read_f32(const char *path, float *p, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t got = fread(p, sizeof(float), n, f);
    int extra = fgetc(f);
    fclose(f);
    return got == n && extra == EOF ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s vision.bin vision.idx patches.f32 grid_h grid_w output.f32\n", argv[0]);
        return 2;
    }
    const int gh = atoi(argv[4]), gw = atoi(argv[5]);
    if (gh <= 0 || gw <= 0 || (gh & 1) || (gw & 1)) return 2;
    const int seq = gh * gw;
    const int out_n = (gh / 2) * (gw / 2);
    float *patches = (float *)malloc((size_t)seq * 588 * sizeof(float));
    float *out = (float *)malloc((size_t)out_n * 2048 * sizeof(float));
    if (!patches || !out || read_f32(argv[3], patches, (size_t)seq * 588) != 0) return 2;

    KvlTrunkStore vs;
    if (kvl_trunk_store_open(&vs, argv[1], argv[2], 1) != 0) {
        fprintf(stderr, "failed to open vision store\n");
        return 2;
    }
    int produced = 0;
    if (kvl_vision_forward(&vs, patches, gh, gw, out, &produced) != 0 || produced != out_n) {
        fprintf(stderr, "vision forward failed\n");
        return 1;
    }
    FILE *f = fopen(argv[6], "wb");
    if (!f || fwrite(out, sizeof(float), (size_t)produced * 2048, f) != (size_t)produced * 2048) return 2;
    if (f) fclose(f);

    double l2 = 0.0;
    float mn = out[0], mx = out[0];
    for (size_t i = 0; i < (size_t)produced * 2048; ++i) {
        const double v = out[i]; l2 += v * v;
        if (out[i] < mn) mn = out[i];
        if (out[i] > mx) mx = out[i];
    }
    fprintf(stderr, "kvl_vision: grid=%dx%d media_tokens=%d direct_io=%s min=%.7g max=%.7g rms=%.7g\n",
            gh, gw, produced, vs.direct_io ? "yes" : "no", mn, mx,
            sqrt(l2 / ((double)produced * 2048.0)));
    kvl_trunk_store_close(&vs);
    free(patches); free(out);
    return 0;
}
