#!/usr/bin/env python3
"""Patch src/generate.c in CI to measure cheap next-layer router prediction accuracy.

Diagnostic only: it never changes the authoritative route or expert computation. After each
layer finishes, it applies the next layer's real router to three current-layer candidate hidden
states and later compares those predicted top-k IDs with the next layer's true route.
"""
from pathlib import Path

p = Path("src/generate.c")
s = p.read_text()

marker = "static int forward_token(KvlTrunkStore *ts, KvlExpertCache *cache,\n"
if s.count(marker) != 1:
    raise SystemExit("forward_token marker mismatch")

helper = r'''static int route_overlap_topk(const int *a, const int *b) {
    int hit = 0;
    for (int i = 0; i < TOPK; ++i)
        for (int j = 0; j < TOPK; ++j)
            if (a[i] == b[j]) { ++hit; break; }
    return hit;
}

static int predict_next_router3(KvlTrunkStore *ts, int layer,
                                const float *n2, const float *r1, const float *x,
                                float *router, float *bias, float *weights,
                                int *pred_n2, int *pred_r1, int *pred_x) {
    KvlTrunkTensor rt = {0}, rb = {0};
    int rc = -1;
    if (load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_WEIGHT, &rt) ||
        load_kind(ts, (uint32_t)layer, KVL_TENSOR_ROUTER_BIAS, &rb)) goto done;
    expand(router, (const uint16_t *)rt.data, (size_t)E * H);
    expand(bias, (const uint16_t *)rb.data, E);
    KvlRouterConfig cfg = {H, E, TOPK, 1, 1, 1, 2.446f};
    if (kvl_router_noaux_tc(&cfg, n2, router, bias, pred_n2, weights) != 0) goto done;
    if (kvl_router_noaux_tc(&cfg, r1, router, bias, pred_r1, weights) != 0) goto done;
    if (kvl_router_noaux_tc(&cfg, x, router, bias, pred_x, weights) != 0) goto done;
    rc = 0;
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
new = r'''    int pred_n2[TOPK] = {0}, pred_r1[TOPK] = {0}, pred_x[TOPK] = {0};
    int have_pred = 0;
    for (int layer = 0; layer < LN; ++layer) {
        if (attention_token(ts, layer, x, position, &states[layer], r1, n2) != 0) return -1;
        if (mlp_token(ts, cache, layer, n2, y, router, bias, ids, weights, scratch) != 0) return -1;
        if (layer > 0 && have_pred) {
            fprintf(stderr,
                    "kvl_route_pred: position=%d layer=%d n2_hit=%d r1_hit=%d x_hit=%d\n",
                    position, layer,
                    route_overlap_topk(pred_n2, ids),
                    route_overlap_topk(pred_r1, ids),
                    route_overlap_topk(pred_x, ids));
        }
        for (int i = 0; i < H; ++i) x[i] = r1[i] + y[i];
        if (layer + 1 < LN) {
            if (predict_next_router3(ts, layer + 1, n2, r1, x,
                                     router, bias, weights,
                                     pred_n2, pred_r1, pred_x) != 0)
                return -1;
            have_pred = 1;
        }
    }
'''
if s.count(old) != 1:
    raise SystemExit("forward loop marker mismatch")
s = s.replace(old, new, 1)
p.write_text(s)
print("patched src/generate.c for route prediction diagnostics")
