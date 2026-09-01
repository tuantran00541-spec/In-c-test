#ifndef KVL_TRUNK_STORE_H
#define KVL_TRUNK_STORE_H

#include <stddef.h>
#include <stdint.h>
#include "kvl/trunk_format.h"

typedef struct {
    int fd;
    int direct_io;
    KvlTrunkIndexHeader hdr;
    KvlTrunkRecord *records;

    /* Opt-in bounded cache for non-global trunk tensors. Global embedding,
     * final norm, and LM head are already retained by the generator and are
     * deliberately excluded to avoid duplicate residency. */
    void **cache_base; /* one aligned allocation per index record, NULL if uncached */
    size_t cache_budget_bytes;
    size_t cache_bytes;
    int cache_configured;
    uint64_t load_calls;
    uint64_t cache_hits;
    uint64_t cache_inserts;
    uint64_t bytes_read;
} KvlTrunkStore;

typedef struct {
    const KvlTrunkRecord *record;
    void *base;
    void *data;
    int owned; /* non-zero only for transient buffers that tensor_free must release */
} KvlTrunkTensor;

int kvl_trunk_store_open(KvlTrunkStore *s, const char *bin_path, const char *idx_path,
                         int prefer_direct_io);
void kvl_trunk_store_close(KvlTrunkStore *s);
const KvlTrunkRecord *kvl_trunk_find(const KvlTrunkStore *s, uint32_t layer, uint32_t kind);
int kvl_trunk_load(KvlTrunkStore *s, uint32_t layer, uint32_t kind, KvlTrunkTensor *out);
void kvl_trunk_tensor_free(KvlTrunkTensor *t);
void kvl_trunk_cache_report(const KvlTrunkStore *s);

#endif
