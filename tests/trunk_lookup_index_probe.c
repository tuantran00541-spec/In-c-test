#include "kvl/trunk_store.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <process.h>
#define kvl_getpid _getpid
#else
#include <unistd.h>
#define kvl_getpid getpid
#endif

static int write_fixture(const char *bin_path, const char *idx_path) {
    static const uint32_t kinds[12] = {
        10, 11, 12, 13, 14, 15, 16, 30, 31, 32, 33, 34
    };
    const uint32_t n_records = 27u * 12u + 3u;
    KvlTrunkRecord *records = (KvlTrunkRecord *)calloc(n_records, sizeof *records);
    if (!records) return -1;

    uint32_t at = 0;
    for (uint32_t layer = 0; layer < 27; ++layer) {
        for (uint32_t j = 0; j < 12; ++j) {
            KvlTrunkRecord *r = &records[at];
            r->layer = layer;
            r->kind = kinds[j];
            r->dtype = KVL_TRUNK_DTYPE_BF16;
            r->ndim = 1;
            r->dims[0] = 1;
            r->file_offset = (uint64_t)at * KVL_TRUNK_ALIGN;
            r->read_bytes = KVL_TRUNK_ALIGN;
            r->payload_bytes = 2;
            ++at;
        }
    }
    for (uint32_t kind = 1; kind <= 3; ++kind) {
        KvlTrunkRecord *r = &records[at];
        r->layer = KVL_TRUNK_GLOBAL_LAYER;
        r->kind = kind;
        r->dtype = KVL_TRUNK_DTYPE_BF16;
        r->ndim = 1;
        r->dims[0] = 1;
        r->file_offset = (uint64_t)at * KVL_TRUNK_ALIGN;
        r->read_bytes = KVL_TRUNK_ALIGN;
        r->payload_bytes = 2;
        ++at;
    }

    FILE *bf = fopen(bin_path, "wb");
    if (!bf) { free(records); return -1; }
    fclose(bf);

    KvlTrunkIndexHeader hdr;
    memset(&hdr, 0, sizeof hdr);
    memcpy(hdr.magic, KVL_TRUNK_MAGIC, 8);
    hdr.version = KVL_TRUNK_VERSION;
    hdr.align = KVL_TRUNK_ALIGN;
    hdr.n_records = n_records;
    hdr.records_offset = sizeof hdr;
    hdr.data_file_bytes = (uint64_t)n_records * KVL_TRUNK_ALIGN;

    FILE *f = fopen(idx_path, "wb");
    if (!f) { free(records); remove(bin_path); return -1; }
    const int ok = fwrite(&hdr, 1, sizeof hdr, f) == sizeof hdr &&
                   fwrite(records, sizeof *records, n_records, f) == n_records;
    fclose(f);
    free(records);
    if (!ok) { remove(bin_path); remove(idx_path); return -1; }
    return 0;
}

static uint64_t next_u64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * UINT64_C(2685821657736338717);
}

int main(void) {
    char bin_path[96], idx_path[96];
    const int pid = (int)kvl_getpid();
    snprintf(bin_path, sizeof bin_path, "kvl_trunk_lookup_%d.bin", pid);
    snprintf(idx_path, sizeof idx_path, "kvl_trunk_lookup_%d.idx", pid);
    remove(bin_path); remove(idx_path);

    if (write_fixture(bin_path, idx_path) != 0) {
        fprintf(stderr, "fixture write failed\n");
        return 1;
    }

    KvlTrunkStore indexed;
    if (kvl_trunk_store_open(&indexed, bin_path, idx_path, 0) != 0) {
        fprintf(stderr, "store open failed\n");
        remove(bin_path); remove(idx_path);
        return 2;
    }
    if (!indexed.lookup_slots || indexed.lookup_cap < (size_t)indexed.hdr.n_records * 2u ||
        (indexed.lookup_cap & (indexed.lookup_cap - 1u)) != 0) {
        fprintf(stderr, "lookup table invariant failed cap=%zu n=%u\n",
                indexed.lookup_cap, indexed.hdr.n_records);
        kvl_trunk_store_close(&indexed);
        remove(bin_path); remove(idx_path);
        return 3;
    }

    KvlTrunkStore linear = indexed;
    linear.lookup_slots = NULL;
    linear.lookup_cap = 0;

    for (uint32_t i = 0; i < indexed.hdr.n_records; ++i) {
        const KvlTrunkRecord *want = &indexed.records[i];
        const KvlTrunkRecord *a = kvl_trunk_find(&indexed, want->layer, want->kind);
        const KvlTrunkRecord *b = kvl_trunk_find(&linear, want->layer, want->kind);
        if (a != want || b != want || a != b) {
            fprintf(stderr, "lookup mismatch record=%u layer=%u kind=%u\n",
                    i, want->layer, want->kind);
            kvl_trunk_store_close(&indexed);
            remove(bin_path); remove(idx_path);
            return 4;
        }
    }

    static const uint32_t missing[][2] = {
        {0, 999}, {26, 999}, {99, 10}, {KVL_TRUNK_GLOBAL_LAYER, 34}
    };
    for (size_t i = 0; i < sizeof missing / sizeof missing[0]; ++i) {
        if (kvl_trunk_find(&indexed, missing[i][0], missing[i][1]) != NULL ||
            kvl_trunk_find(&linear, missing[i][0], missing[i][1]) != NULL) {
            fprintf(stderr, "missing lookup mismatch i=%zu\n", i);
            kvl_trunk_store_close(&indexed);
            remove(bin_path); remove(idx_path);
            return 5;
        }
    }

    const int iters = 200000;
    uint64_t rng = UINT64_C(0x2609015eed);
    volatile uint64_t sum_indexed = 0, sum_linear = 0;
    clock_t t0 = clock();
    for (int i = 0; i < iters; ++i) {
        const uint32_t ri = (uint32_t)(next_u64(&rng) % indexed.hdr.n_records);
        const KvlTrunkRecord *r = &indexed.records[ri];
        const KvlTrunkRecord *got = kvl_trunk_find(&indexed, r->layer, r->kind);
        if (!got) return 6;
        sum_indexed += (uint64_t)(got - indexed.records) + 1u;
    }
    clock_t t1 = clock();

    rng = UINT64_C(0x2609015eed);
    for (int i = 0; i < iters; ++i) {
        const uint32_t ri = (uint32_t)(next_u64(&rng) % indexed.hdr.n_records);
        const KvlTrunkRecord *r = &indexed.records[ri];
        const KvlTrunkRecord *got = kvl_trunk_find(&linear, r->layer, r->kind);
        if (!got) return 7;
        sum_linear += (uint64_t)(got - indexed.records) + 1u;
    }
    clock_t t2 = clock();

    const double indexed_s = (double)(t1 - t0) / CLOCKS_PER_SEC;
    const double linear_s = (double)(t2 - t1) / CLOCKS_PER_SEC;
    printf("records=%u lookup_cap=%zu checksums_equal=%s indexed_s=%.6f linear_s=%.6f ratio=%.3f\n",
           indexed.hdr.n_records, indexed.lookup_cap,
           sum_indexed == sum_linear ? "yes" : "no",
           indexed_s, linear_s, indexed_s > 0.0 ? linear_s / indexed_s : 0.0);
    if (sum_indexed != sum_linear) {
        kvl_trunk_store_close(&indexed);
        remove(bin_path); remove(idx_path);
        return 8;
    }

    kvl_trunk_store_close(&indexed);
    remove(bin_path); remove(idx_path);
    puts("TRUNK_INDEX_LOOKUP_EXACT_PASS");
    return 0;
}
