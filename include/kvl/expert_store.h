#ifndef KVL_EXPERT_STORE_H
#define KVL_EXPERT_STORE_H

#include <stddef.h>
#include <stdint.h>
#include "kvl/format.h"

typedef struct {
    int fd;
    int direct_io;
    KvlExpertIndexHeader hdr;
    KvlExpertRecord *records;
    int32_t *record_of; /* layer*n_experts+expert -> record index, -1 if absent */
} KvlExpertStore;

int kvl_expert_store_open(KvlExpertStore *s, const char *bin_path, const char *idx_path,
                          int prefer_direct_io);
void kvl_expert_store_close(KvlExpertStore *s);
const KvlExpertRecord *kvl_expert_find(const KvlExpertStore *s, int layer, int expert);
int kvl_expert_alloc_buffer(const KvlExpertRecord *r, void **out);
void kvl_expert_free_buffer(void *p);
int64_t kvl_expert_load(const KvlExpertStore *s, const KvlExpertRecord *r, void *aligned_buf);

#endif
