#include "kvl/expert_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int pinned(const KvlExpertCache *c, int layer, int expert) {
    const size_t key = (size_t)layer * c->store->hdr.n_experts + (size_t)expert;
    return c->pinned_of && c->pinned_of[key] != 0;
}

int main(void) {
    KvlExpertStore store;
    KvlExpertCache cache;
    memset(&store, 0, sizeof store);
    memset(&cache, 0, sizeof cache);
    store.hdr.n_layers = 4;
    store.hdr.n_experts = 8;
    cache.store = &store;

    const int l1a[2] = {2, 5};
    const int l2[2] = {1, 7};
    const int l1b[2] = {3, 4};

    if (kvl_expert_cache_pin_layer(&cache, 1, l1a, 2) != 0 ||
        !pinned(&cache, 1, 2) || !pinned(&cache, 1, 5))
        return 1;
    if (kvl_expert_cache_pin_layer(&cache, 2, l2, 2) != 0 ||
        !pinned(&cache, 2, 1) || !pinned(&cache, 2, 7) ||
        !pinned(&cache, 1, 2) || !pinned(&cache, 1, 5))
        return 1;
    if (kvl_expert_cache_pin_layer(&cache, 1, l1b, 2) != 0 ||
        pinned(&cache, 1, 2) || pinned(&cache, 1, 5) ||
        !pinned(&cache, 1, 3) || !pinned(&cache, 1, 4) ||
        !pinned(&cache, 2, 1) || !pinned(&cache, 2, 7))
        return 1;

    const int bad[1] = {99};
    if (kvl_expert_cache_pin_layer(&cache, 1, bad, 1) == 0 ||
        !pinned(&cache, 1, 3) || !pinned(&cache, 1, 4))
        return 1;
    if (kvl_expert_cache_pin_layer(&cache, 1, NULL, 0) != 0 ||
        pinned(&cache, 1, 3) || pinned(&cache, 1, 4) ||
        !pinned(&cache, 2, 1) || !pinned(&cache, 2, 7))
        return 1;

    free(cache.pinned_of);
    puts("KIMI_CACHE_LAYER_PIN_UNIT_PASS");
    return 0;
}
