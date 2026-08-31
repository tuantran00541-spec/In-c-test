#ifndef KVL_FORMAT_H
#define KVL_FORMAT_H

#include <stdint.h>

#define KVL_EXPERT_MAGIC "KVLXPRT1"
#define KVL_EXPERT_VERSION 1u
#define KVL_EXPERT_ALIGN 4096u
#define KVL_DTYPE_BF16 1u
#define KVL_DTYPE_MXFP4 2u
#define KVL_DTYPE_Q8_ROW 3u
/* Experimental routed-expert format: signed symmetric 5-bit weights,
 * group size 128 along the input dimension, with FP32 scale per group. */
#define KVL_DTYPE_Q5_G128 4u
/* External GGUF-backed expert store. The normal KVL records describe the
 * in-cache layout while one KvlGgufQ8Source per record describes three aligned
 * positioned reads from a GGUF Q8_0 file. */
#define KVL_DTYPE_GGUF_Q8_0 5u

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
    uint64_t read_bytes;   /* padded direct-I/O span, or sum of GGUF part spans */
    uint64_t payload_bytes;
    uint64_t gate_off;
    uint64_t gate_bytes;
    uint64_t up_off;
    uint64_t up_bytes;
    uint64_t down_off;
    uint64_t down_bytes;
} KvlExpertRecord;

/* Appended immediately after KvlExpertRecord[n_records] when dtype is
 * KVL_DTYPE_GGUF_Q8_0. Each destination offset is 4096-aligned inside a cache
 * slot; the corresponding KvlExpertRecord gate/up/down offset points at the
 * actual Q8_0 payload within that aligned envelope. */
typedef struct {
    uint64_t gate_file_offset;
    uint64_t gate_read_bytes;
    uint64_t gate_dst_offset;
    uint64_t up_file_offset;
    uint64_t up_read_bytes;
    uint64_t up_dst_offset;
    uint64_t down_file_offset;
    uint64_t down_read_bytes;
    uint64_t down_dst_offset;
} KvlGgufQ8Source;
#pragma pack(pop)

#endif