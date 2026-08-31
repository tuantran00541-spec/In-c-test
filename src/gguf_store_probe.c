#include "kvl/ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define H 2048
#define I 1408

static int read_exact(const char *path, void *buf, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    const size_t got = fread(buf, 1, bytes, f);
    const int extra = fgetc(f);
    fclose(f);
    return (got == bytes && extra == EOF) ? 0 : -1;
}

static int compare_vec(const char *name, const float *got, const float *ref, int n,
                       double *max_abs_out, double *rms_out) {
    double max_abs = 0.0, sq = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(got[i]) || !isfinite(ref[i])) return -1;
        const double d = (double)got[i] - (double)ref[i];
        const double a = fabs(d);
        if (a > max_abs) max_abs = a;
        sq += d * d;
    }
    const double rms = sqrt(sq / (double)n);
    fprintf(stderr, "KIMI_GGUF_STORE_MATRIX name=%s max_abs=%.9g rms=%.9g\n",
            name, max_abs, rms);
    *max_abs_out = max_abs;
    *rms_out = rms;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr,
                "usage: %s sparse.gguf experts.idx x_h.f32 x_i.f32 gate_ref.f32 up_ref.f32 down_ref.f32\n",
                argv[0]);
        return 2;
    }

    float *xh = (float *)malloc((size_t)H * sizeof(float));
    float *xi = (float *)malloc((size_t)I * sizeof(float));
    float *gate_ref = (float *)malloc((size_t)I * sizeof(float));
    float *up_ref = (float *)malloc((size_t)I * sizeof(float));
    float *down_ref = (float *)malloc((size_t)H * sizeof(float));
    float *gate = (float *)malloc((size_t)I * sizeof(float));
    float *up = (float *)malloc((size_t)I * sizeof(float));
    float *down = (float *)malloc((size_t)H * sizeof(float));
    if (!xh || !xi || !gate_ref || !up_ref || !down_ref || !gate || !up || !down)
        return 2;
    if (read_exact(argv[3], xh, (size_t)H * sizeof(float)) ||
        read_exact(argv[4], xi, (size_t)I * sizeof(float)) ||
        read_exact(argv[5], gate_ref, (size_t)I * sizeof(float)) ||
        read_exact(argv[6], up_ref, (size_t)I * sizeof(float)) ||
        read_exact(argv[7], down_ref, (size_t)H * sizeof(float)))
        return 2;

    KvlExpertStore store;
    KvlExpertCache cache;
    if (kvl_expert_store_open(&store, argv[1], argv[2], 1) != 0) {
        fprintf(stderr, "KIMI_GGUF_STORE_OPEN_FAIL\n");
        return 1;
    }
    if (store.hdr.dtype != KVL_DTYPE_GGUF_Q8_0 || !store.gguf_q8_sources) {
        fprintf(stderr, "KIMI_GGUF_STORE_BAD_DTYPE dtype=%u\n", store.hdr.dtype);
        return 1;
    }
    if (kvl_expert_cache_init(&cache, &store, 64u * 1024u * 1024u) != 0) {
        fprintf(stderr, "KIMI_GGUF_STORE_CACHE_INIT_FAIL\n");
        return 1;
    }

    KvlCachedExpert q;
    if (kvl_expert_cache_get(&cache, 1, 0, &q) != 0) {
        fprintf(stderr, "KIMI_GGUF_STORE_GET_FAIL\n");
        return 1;
    }
    kvl_matvec_ggml_q8_0(gate, xh, q.gate, H, I);
    kvl_matvec_ggml_q8_0(up, xh, q.up, H, I);
    kvl_matvec_ggml_q8_0(down, xi, q.down, I, H);

    double ag = 0.0, au = 0.0, ad = 0.0, rg = 0.0, ru = 0.0, rd = 0.0;
    if (compare_vec("gate", gate, gate_ref, I, &ag, &rg) ||
        compare_vec("up", up, up_ref, I, &au, &ru) ||
        compare_vec("down", down, down_ref, H, &ad, &rd))
        return 1;
    const double max_abs = fmax(ag, fmax(au, ad));
    const double max_rms = fmax(rg, fmax(ru, rd));

    fprintf(stderr,
            "KIMI_GGUF_STORE_SMOKE direct_io=%d dtype=%u slot_bytes=%zu arena_bytes=%zu "
            "requests=%llu misses=%llu read_ops=%llu bytes_read=%llu max_abs=%.9g max_rms=%.9g\n",
            store.direct_io, store.hdr.dtype, cache.slot_bytes, cache.arena_bytes,
            (unsigned long long)cache.requests, (unsigned long long)cache.misses,
            (unsigned long long)cache.read_ops, (unsigned long long)cache.bytes_read,
            max_abs, max_rms);
    kvl_expert_cache_report(&cache);

    if (!store.direct_io || cache.requests != 1 || cache.misses != 1 ||
        cache.read_ops != 3 || cache.bytes_read != q.record->read_bytes ||
        max_abs > 2e-4 || max_rms > 2e-5) {
        fprintf(stderr, "KIMI_GGUF_STORE_SMOKE_FAIL\n");
        return 1;
    }
    fprintf(stderr, "KIMI_GGUF_STORE_SMOKE_PASS\n");

    kvl_expert_cache_close(&cache);
    kvl_expert_store_close(&store);
    free(xh); free(xi); free(gate_ref); free(up_ref); free(down_ref);
    free(gate); free(up); free(down);
    return 0;
}
