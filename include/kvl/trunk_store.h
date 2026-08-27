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
} KvlTrunkStore;

typedef struct {
    const KvlTrunkRecord *record;
    void *base;
    void *data;
} KvlTrunkTensor;

int kvl_trunk_store_open(KvlTrunkStore *s, const char *bin_path, const char *idx_path,
                         int prefer_direct_io);
void kvl_trunk_store_close(KvlTrunkStore *s);
const KvlTrunkRecord *kvl_trunk_find(const KvlTrunkStore *s, uint32_t layer, uint32_t kind);
int kvl_trunk_load(const KvlTrunkStore *s, uint32_t layer, uint32_t kind, KvlTrunkTensor *out);
void kvl_trunk_tensor_free(KvlTrunkTensor *t);

#endif
