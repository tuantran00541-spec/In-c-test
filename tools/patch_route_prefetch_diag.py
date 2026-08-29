#!/usr/bin/env python3
"""Patch src/generate.c for a Linux-only speculative expert prefetch benchmark.

Safety model:
- authoritative router/math are unchanged;
- predicted experts for layer L are prefetched only while attention(L) runs;
- the prefetch thread is joined before MoE(L), so cache metadata is never touched
  concurrently by the prefetch worker and MoE;
- wrong predictions can only add I/O/cache pressure, never change selected experts.

This is intentionally a CI diagnostic patch, not production code.
"""
from pathlib import Path

p = Path("src/generate.c")
s = p.read_text()

inc = '#include <string.h>\n'
if s.count(inc) != 1:
    raise SystemExit('include marker mismatch')
s = s.replace(inc, inc + '#include <pthread.h>\n', 1)

marker = "static int forward_token(KvlTrunkStore *ts, KvlExpertCache *cache,\n"
if s.count(marker) != 1:
    raise SystemExit('forward marker mismatch')

helper = r'''typedef struct {
    KvlExpertCache *cache;
    int layer;
    int ids[TOPK];
    int loaded;
} KvlRoutePrefetchTask;

static void *route_prefetch_worker(void *arg) {
    KvlRoutePrefetchTask *t = (KvlRoutePrefetchTask *)arg;
    t->loaded = kvl_expert_cache_getmany(t->cache, t->layer, t->ids, TOPK);
    return NULL;
}

static int predict_next_router_n2(KvlTrunkStore *ts, int next_layer,
                                  const float *n2, float *router, float *bias,
                                  int *pred_ids, float *weights) {
    KvlTrunkTensor rt = {0}, rb = {0};
    int rc = -1;
    if (next_layer <= 0 || next_layer >= LN) return -1;
    if (load_kind(ts, (uint32_t)next_layer, KVL_TENSOR_ROUTER_WEIGHT, &rt) ||
        load_kind(ts, (uint32_t)next_layer, KVL_TENSOR_ROUTER_BIAS, &rb)) goto done;
    expand(router, (const uint16_t *)rt.data, (size_t)E * H);
    expand(bias, (const uint16_t *)rb.data, E);
    KvlRouterConfig cfg = {H, E, TOPK, 1, 1, 1, 2.446f};
    rc = kvl_router_noaux_tc(&cfg, n2, router, bias, pred_ids, weights);
done:
    kvl_trunk_tensor_free(&rt);
    kvl_trunk_tensor_free(&rb);
    return rc;
}

'''
s = s.replace(marker, helper + marker, 1)

old = r'''    for (int layer = 0; layer < LN; ++layer) {
        if (attention_token(ts, layer, x, position, &states[layer], r1, n2) != 0) return -1;
        if (mlp_token(ts, cache, layer, n2, y, router, bias, ids, weights, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) x[i] = r1[i] + y[i];
    }
'''
new = r'''    int pred_ids[TOPK] = {0};
    int have_pred = 0;
    uint64_t spec_batches = 0, spec_loaded = 0, spec_failed = 0;
    for (int layer = 0; layer < LN; ++layer) {
        pthread_t prefetch_thread;
        KvlRoutePrefetchTask task;
        int launched = 0;
        if (layer > 0 && have_pred) {
            memset(&task, 0, sizeof task);
            task.cache = cache;
            task.layer = layer;
            memcpy(task.ids, pred_ids, sizeof pred_ids);
            if (pthread_create(&prefetch_thread, NULL, route_prefetch_worker, &task) == 0) {
                launched = 1;
                spec_batches++;
            } else {
                spec_failed++;
            }
        }

        const int attn_rc = attention_token(ts, layer, x, position, &states[layer], r1, n2);
        if (launched) {
            if (pthread_join(prefetch_thread, NULL) != 0) {
                spec_failed++;
            } else if (task.loaded < 0) {
                spec_failed++;
            } else {
                spec_loaded += (uint64_t)task.loaded;
            }
        }
        if (attn_rc != 0) return -1;

        if (mlp_token(ts, cache, layer, n2, y, router, bias, ids, weights, scratch) != 0) return -1;
        for (int i = 0; i < H; ++i) x[i] = r1[i] + y[i];

        have_pred = 0;
        if (layer + 1 < LN && layer + 1 > 0) {
            if (predict_next_router_n2(ts, layer + 1, n2, router, bias,
                                       pred_ids, weights) == 0)
                have_pred = 1;
        }
    }
    fprintf(stderr,
            "kvl_route_prefetch: position=%d batches=%llu loaded=%llu failed=%llu\n",
            position,
            (unsigned long long)spec_batches,
            (unsigned long long)spec_loaded,
            (unsigned long long)spec_failed);
'''
if s.count(old) != 1:
    raise SystemExit('forward loop marker mismatch')
s = s.replace(old, new, 1)
p.write_text(s)
print('patched src/generate.c for safe attention-overlapped route prefetch diagnostic')
