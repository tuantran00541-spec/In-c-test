#include "kvl/expert_cache.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t fnv1a64(const unsigned char *p, size_t n) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s experts.bin experts.idx CACHE_BYTES LAYER REPEATS EXPERT...\n", argv[0]);
        return 2;
    }
    const size_t cache_bytes = (size_t)strtoull(argv[3], NULL, 10);
    const int layer = atoi(argv[4]);
    const int repeats = atoi(argv[5]);
    const int n = argc - 6;
    int *ids = (int *)malloc((size_t)n * sizeof *ids);
    if (!ids) return 1;
    for (int i = 0; i < n; ++i) ids[i] = atoi(argv[i + 6]);

    KvlExpertStore s;
    if (kvl_expert_store_open(&s, argv[1], argv[2], 0) != 0) {
        fprintf(stderr, "failed to open store\n"); free(ids); return 1;
    }
    KvlExpertCache c;
    if (kvl_expert_cache_init(&c, &s, cache_bytes) != 0) {
        fprintf(stderr, "failed to init cache\n"); kvl_expert_store_close(&s); free(ids); return 1;
    }

    uint64_t checksum = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        if (kvl_expert_cache_getmany(&c, layer, ids, n) < 0) return 1;
        for (int i = 0; i < n; ++i) {
            KvlCachedExpert e;
            if (kvl_expert_cache_get(&c, layer, ids[i], &e) != 0) {
                fprintf(stderr, "get failed L%d/E%d\n", layer, ids[i]); return 1;
            }
            checksum ^= fnv1a64(e.base, (size_t)e.record->payload_bytes) + (uint64_t)(i + 1 + rep * n);
        }
    }
    printf("checksum=%016llx slots=%d arena_bytes=%llu budget_bytes=%llu\n",
           (unsigned long long)checksum, c.n_slots,
           (unsigned long long)c.arena_bytes, (unsigned long long)c.budget_bytes);
    kvl_expert_cache_report(&c);
    kvl_expert_cache_close(&c);
    kvl_expert_store_close(&s);
    free(ids);
    return 0;
}
