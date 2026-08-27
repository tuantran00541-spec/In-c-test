#include "kvl/expert_store.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s experts.bin experts.idx LAYER EXPERT\n", argv[0]);
        return 2;
    }
    KvlExpertStore s;
    if (kvl_expert_store_open(&s, argv[1], argv[2], 1) != 0) {
        fprintf(stderr, "failed to open expert store\n"); return 1;
    }
    int layer = atoi(argv[3]), expert = atoi(argv[4]);
    const KvlExpertRecord *r = kvl_expert_find(&s, layer, expert);
    if (!r) { fprintf(stderr, "expert L%d/E%d not present\n", layer, expert); return 1; }
    void *buf = NULL;
    if (kvl_expert_alloc_buffer(r, &buf) != 0) { fprintf(stderr, "buffer alloc failed\n"); return 1; }
    int64_t got = kvl_expert_load(&s, r, buf);
    printf("L%d/E%d offset=%llu payload=%llu read=%llu got=%lld direct_io=%s\n",
           layer, expert, (unsigned long long)r->file_offset,
           (unsigned long long)r->payload_bytes, (unsigned long long)r->read_bytes,
           (long long)got, s.direct_io ? "yes" : "no");
    kvl_expert_free_buffer(buf);
    kvl_expert_store_close(&s);
    return got == (int64_t)r->read_bytes ? 0 : 1;
}
