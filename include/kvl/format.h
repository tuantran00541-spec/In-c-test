#ifndef KVL_FORMAT_H
#define KVL_FORMAT_H

#include <stdint.h>

#define KVL_EXPERT_MAGIC "KVLXPRT1"
#define KVL_EXPERT_VERSION 1u
#define KVL_EXPERT_ALIGN 4096u
#define KVL_DTYPE_BF16 1u
#define KVL_DTYPE_MXFP4 2u

#pragma pack(push, 1)
typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t align;
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t n_records;
    uint32_t dtype;
    uint64_t records_offset;
    uint64_t data_file_bytes;
} KvlExpertIndexHeader;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t file_offset;
    uint64_t read_bytes;   /* padded direct-I/O span */
    uint64_t payload_bytes;
    uint64_t gate_off;
    uint64_t gate_bytes;
    uint64_t up_off;
    uint64_t up_bytes;
    uint64_t down_off;
    uint64_t down_bytes;
} KvlExpertRecord;
#pragma pack(pop)

#endif
