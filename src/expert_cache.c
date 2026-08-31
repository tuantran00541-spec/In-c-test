#define _POSIX_C_SOURCE 200809L
#include "kvl/expert_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <malloc.h>
#endif

static double now_s(void) {
#ifdef _WIN32
    return (double)clock() / (double)CLOCKS_PER_SEC;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
#endif
}

static size_t align_up_size(size_t x, size_t a) {
    return (x + a - 1u) / a * a;
}

static int aligned_alloc_cache(void **out, size_t align, size_t bytes) {
#ifdef _WIN32
    *out = _aligned_malloc(bytes, align);
    return *out ? 0 : -1;
#else
    return posix_memalign(out, align, bytes);
#endif
}

static void aligned_free_cache(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static int key_for(const KvlExpertCache *c, int layer, int expert, int32_t *out) {
    if (!c || !c->store || layer < 0 || expert < 0 ||
        layer >= (int)c->store->hdr.n_layers || expert >= (int)c->store->hdr.n_experts)
        return -1;
    const uint64_t k = (uint64_t)(uint32_t)layer * c->store->hdr.n_experts + (uint32_t)expert;
    if (k > INT32_MAX) return -1;
    *out = (int32_t)k;
    return 0;
}

static int pick_victim(KvlExpertCache *c) {
    int best = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < c->n_slots; ++i) {
        if (c->key_of[i] == KVL_CACHE_INFLIGHT) continue;
        if (c->key_of[i] == KVL_CACHE_EMPTY) return i;
        if (c->used_at[i] < oldest) {
            oldest = c->used_at[i];
            best = i;
        }
    }
    return best;
}

static void publish_out(KvlExpertCache *c, int slot, KvlCachedExpert *out) {
    const KvlExpertRecord *r = c->record_of_slot[slot];
    const unsigned char *base = c->arena + (size_t)slot * c->slot_bytes;
    out->record = r;
    out->base = base;
    out->gate = base + r->gate_off;
    out->up = base + r->up_off;
    out->down = base + r->down_off;
}

static int load_into_slot(KvlExpertCache *c, int slot, const KvlExpertRecord *r) {
    const double t0 = now_s();
    const int64_t got = kvl_expert_load(c->store, r,
        c->arena + (size_t)slot * c->slot_bytes);
    const double dt = now_s() - t0;
    c->read_seconds += dt;
    c->read_ops += kvl_expert_load_read_ops(c->store, r);
    if (got != (int64_t)r->read_bytes) {
        c->read_failures++;
        return -1;
    }
    c->bytes_read += r->read_bytes;
    return 0;
}

int kvl_expert_cache_init(KvlExpertCache *c, KvlExpertStore *store, size_t budget_bytes) {
    if (!c || !store || store->fd < 0 || budget_bytes == 0) return -1;
    memset(c, 0, sizeof *c);
    c->store = store;
    c->budget_bytes = budget_bytes;

    size_t max_read = 0;
    for (uint32_t i = 0; i < store->hdr.n_records; ++i) {
        const size_t n = (size_t)store->records[i].read_bytes;
        if (n > max_read) max_read = n;
    }
    if (!max_read) return -1;
    c->slot_bytes = align_up_size(max_read, KVL_EXPERT_ALIGN);
    c->n_slots = (int)(budget_bytes / c->slot_bytes);
    if (c->n_slots < 1) {
        fprintf(stderr, "kvl_cache: budget %.2f MiB is smaller than one %.2f MiB slot\n",
                budget_bytes / 1048576.0, c->slot_bytes / 1048576.0);
        return -1;
    }
    c->arena_bytes = (size_t)c->n_slots * c->slot_bytes;
    if (c->arena_bytes > budget_bytes) return -1; /* hard invariant */

    if (aligned_alloc_cache((void **)&c->arena, KVL_EXPERT_ALIGN, c->arena_bytes) != 0)
        goto fail;

    const size_t map_n = (size_t)store->hdr.n_layers * store->hdr.n_experts;
    c->slot_of = (int32_t *)malloc(map_n * sizeof *c->slot_of);
    c->key_of = (int32_t *)malloc((size_t)c->n_slots * sizeof *c->key_of);
    c->used_at = (uint64_t *)calloc((size_t)c->n_slots, sizeof *c->used_at);
    c->record_of_slot = (const KvlExpertRecord **)calloc((size_t)c->n_slots, sizeof *c->record_of_slot);
    if (!c->slot_of || !c->key_of || !c->used_at || !c->record_of_slot) goto fail;

    for (size_t i = 0; i < map_n; ++i) c->slot_of[i] = -1;
    for (int i = 0; i < c->n_slots; ++i) c->key_of[i] = KVL_CACHE_EMPTY;
    return 0;

fail:
    kvl_expert_cache_close(c);
    return -1;
}

void kvl_expert_cache_close(KvlExpertCache *c) {
    if (!c) return;
    aligned_free_cache(c->arena);
    free(c->slot_of);
    free(c->key_of);
    free(c->used_at);
    free(c->record_of_slot);
    memset(c, 0, sizeof *c);
}

int kvl_expert_cache_resident(KvlExpertCache *c, int layer, int expert, KvlCachedExpert *out) {
    int32_t key;
    if (key_for(c, layer, expert, &key) != 0) return 0;
    const int slot = c->slot_of[key];
    if (slot < 0 || slot >= c->n_slots || c->key_of[slot] != key) return 0;
    if (out) publish_out(c, slot, out);
    return 1;
}

int kvl_expert_cache_get(KvlExpertCache *c, int layer, int expert, KvlCachedExpert *out) {
    if (!c || !out) return -1;
    int32_t key;
    if (key_for(c, layer, expert, &key) != 0) return -1;
    c->requests++;

    int slot = c->slot_of[key];
    if (slot >= 0 && slot < c->n_slots && c->key_of[slot] == key) {
        c->hits++;
        c->used_at[slot] = ++c->clock;
        publish_out(c, slot, out);
        return 0;
    }
    c->misses++;

    const KvlExpertRecord *r = kvl_expert_find(c->store, layer, expert);
    if (!r || r->read_bytes > c->slot_bytes) return -1;
    slot = pick_victim(c);
    if (slot < 0) return -1;
    if (c->key_of[slot] >= 0) {
        c->slot_of[c->key_of[slot]] = -1;
        c->evictions++;
    }
    c->key_of[slot] = KVL_CACHE_INFLIGHT;
    c->record_of_slot[slot] = NULL;
    c->used_at[slot] = ++c->clock;

    if (load_into_slot(c, slot, r) != 0) {
        c->key_of[slot] = KVL_CACHE_EMPTY;
        return -1;
    }

    c->record_of_slot[slot] = r;
    c->key_of[slot] = key;
    c->slot_of[key] = slot;
    publish_out(c, slot, out);
    return 0;
}

typedef struct {
    int slot;
    int expert;
    int32_t key;
    const KvlExpertRecord *r;
    int64_t got;
} PrefetchWork;

static int cmp_work(const void *a, const void *b) {
    const PrefetchWork *x = (const PrefetchWork *)a;
    const PrefetchWork *y = (const PrefetchWork *)b;
    if (x->r->file_offset < y->r->file_offset) return -1;
    if (x->r->file_offset > y->r->file_offset) return 1;
    return 0;
}

int kvl_expert_cache_getmany(KvlExpertCache *c, int layer, const int *experts, int n) {
    if (!c || !experts || n < 0) return -1;
    if (n == 0) return 0;
    c->prefetch_batches++;

    PrefetchWork *w = (PrefetchWork *)calloc((size_t)n, sizeof *w);
    if (!w) return -1;
    int nw = 0;

    /* Phase 1: reserve distinct slots serially. */
    for (int i = 0; i < n; ++i) {
        const int e = experts[i];
        int32_t key;
        if (key_for(c, layer, e, &key) != 0) continue;
        if (c->slot_of[key] >= 0) continue;

        int dup = 0;
        for (int j = 0; j < nw; ++j) {
            if (w[j].key == key) { dup = 1; break; }
        }
        if (dup) continue;

        const KvlExpertRecord *r = kvl_expert_find(c->store, layer, e);
        if (!r || r->read_bytes > c->slot_bytes) continue;
        const int slot = pick_victim(c);
        if (slot < 0) break;
        if (c->key_of[slot] >= 0) {
            c->slot_of[c->key_of[slot]] = -1;
            c->evictions++;
        }
        c->key_of[slot] = KVL_CACHE_INFLIGHT;
        c->record_of_slot[slot] = NULL;
        c->used_at[slot] = ++c->clock;
        w[nw].slot = slot;
        w[nw].expert = e;
        w[nw].key = key;
        w[nw].r = r;
        w[nw].got = -1;
        nw++;
    }

    if (nw == 0) { free(w); return 0; }
    qsort(w, (size_t)nw, sizeof *w, cmp_work);

    /* Phase 2: only independent positioned reads happen in parallel. Measure the WALL
     * around the whole batch, not the sum of per-read durations, so reported aggregate
     * throughput is meaningful when reads overlap. */
    const double batch_t0 = now_s();
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (i = 0; i < nw; ++i) {
        w[i].got = kvl_expert_load(c->store, w[i].r,
            c->arena + (size_t)w[i].slot * c->slot_bytes);
    }
    c->read_seconds += now_s() - batch_t0;

    /* Phase 3: publish only complete records. */
    int ok = 0;
    for (int j = 0; j < nw; ++j) {
        c->read_ops += kvl_expert_load_read_ops(c->store, w[j].r);
        if (w[j].got != (int64_t)w[j].r->read_bytes) {
            c->read_failures++;
            c->key_of[w[j].slot] = KVL_CACHE_EMPTY;
            c->record_of_slot[w[j].slot] = NULL;
            continue;
        }
        c->record_of_slot[w[j].slot] = w[j].r;
        c->key_of[w[j].slot] = w[j].key;
        c->slot_of[w[j].key] = w[j].slot;
        c->used_at[w[j].slot] = ++c->clock;
        c->bytes_read += w[j].r->read_bytes;
        c->prefetch_reads++;
        ok++;
    }
    free(w);
    return ok;
}

void kvl_expert_cache_reset_stats(KvlExpertCache *c) {
    if (!c) return;
    c->requests = c->hits = c->misses = c->evictions = 0;
    c->bytes_read = c->read_ops = c->prefetch_batches = c->prefetch_reads = 0;
    c->read_failures = 0;
    c->read_seconds = 0.0;
}

void kvl_expert_cache_report(const KvlExpertCache *c) {
    if (!c) return;
    const double hit_rate = c->requests ? 100.0 * (double)c->hits / (double)c->requests : 0.0;
    const double mib = (double)c->bytes_read / 1048576.0;
    const double mib_s = c->read_seconds > 0.0 ? mib / c->read_seconds : 0.0;
    fprintf(stderr,
        "kvl_cache: slots=%d slot=%.2f MiB arena=%.2f/%.2f MiB "
        "req=%llu hit=%llu miss=%llu hit_rate=%.1f%% evict=%llu "
        "prefetch=%llu/%llu reads=%llu bytes=%.2f MiB rate=%.1f MiB/s failures=%llu\n",
        c->n_slots, c->slot_bytes / 1048576.0, c->arena_bytes / 1048576.0,
        c->budget_bytes / 1048576.0,
        (unsigned long long)c->requests, (unsigned long long)c->hits,
        (unsigned long long)c->misses, hit_rate, (unsigned long long)c->evictions,
        (unsigned long long)c->prefetch_reads, (unsigned long long)c->prefetch_batches,
        (unsigned long long)c->read_ops, mib, mib_s,
        (unsigned long long)c->read_failures);
}
