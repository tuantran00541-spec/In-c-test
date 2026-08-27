#ifndef KVL_EXPERT_CACHE_H
#define KVL_EXPERT_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include "kvl/expert_store.h"

#define KVL_CACHE_EMPTY    (-1)
#define KVL_CACHE_INFLIGHT (-2)

typedef struct {
    const KvlExpertRecord *record;
    const unsigned char *base;
    const unsigned char *gate;
    const unsigned char *up;
    const unsigned char *down;
} KvlCachedExpert;

typedef struct {
    KvlExpertStore *store;
    unsigned char *arena;
    size_t budget_bytes;
    size_t arena_bytes;
    size_t slot_bytes;
    int n_slots;

    int32_t *slot_of;   /* layer*n_experts+expert -> slot, -1 if absent */
    int32_t *key_of;    /* EMPTY, INFLIGHT, or flattened key */
    uint64_t *used_at;
    const KvlExpertRecord **record_of_slot;
    uint64_t clock;

    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_read;
    uint64_t read_ops;
    uint64_t prefetch_batches;
    uint64_t prefetch_reads;
    uint64_t read_failures;
    double read_seconds;
} KvlExpertCache;

int kvl_expert_cache_init(KvlExpertCache *c, KvlExpertStore *store, size_t budget_bytes);
void kvl_expert_cache_close(KvlExpertCache *c);

/* Prefetch as many missing experts in the batch as cache capacity permits.
 * Returns number of successfully loaded experts, or -1 on invalid arguments. */
int kvl_expert_cache_getmany(KvlExpertCache *c, int layer, const int *experts, int n);

/* Resolve one expert, loading it synchronously on miss. */
int kvl_expert_cache_get(KvlExpertCache *c, int layer, int expert, KvlCachedExpert *out);

/* Return 1 only if the expert is already valid/resident. Never performs I/O. */
int kvl_expert_cache_resident(KvlExpertCache *c, int layer, int expert, KvlCachedExpert *out);

void kvl_expert_cache_reset_stats(KvlExpertCache *c);
void kvl_expert_cache_report(const KvlExpertCache *c);

#endif
