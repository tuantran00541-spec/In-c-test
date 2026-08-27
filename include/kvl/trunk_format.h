#ifndef KVL_TRUNK_FORMAT_H
#define KVL_TRUNK_FORMAT_H

#include <stdint.h>

#define KVL_TRUNK_MAGIC "KVLTRNK1"
#define KVL_TRUNK_VERSION 1u
#define KVL_TRUNK_ALIGN 4096u
#define KVL_TRUNK_GLOBAL_LAYER 0xffffffffu

#define KVL_TRUNK_DTYPE_BF16 1u
#define KVL_TRUNK_DTYPE_F32  2u

typedef enum {
    KVL_TENSOR_EMBED_TOKENS = 1,
    KVL_TENSOR_FINAL_NORM = 2,
    KVL_TENSOR_LM_HEAD = 3,

    KVL_TENSOR_INPUT_NORM = 10,
    KVL_TENSOR_POST_ATTN_NORM = 11,
    KVL_TENSOR_Q_PROJ = 12,
    KVL_TENSOR_KV_A_PROJ = 13,
    KVL_TENSOR_KV_A_NORM = 14,
    KVL_TENSOR_KV_B_PROJ = 15,
    KVL_TENSOR_O_PROJ = 16,

    KVL_TENSOR_DENSE_GATE = 20,
    KVL_TENSOR_DENSE_UP = 21,
    KVL_TENSOR_DENSE_DOWN = 22,

    KVL_TENSOR_ROUTER_WEIGHT = 30,
    KVL_TENSOR_ROUTER_BIAS = 31,
    KVL_TENSOR_SHARED_GATE = 32,
    KVL_TENSOR_SHARED_UP = 33,
    KVL_TENSOR_SHARED_DOWN = 34
} KvlTrunkTensorKind;

#pragma pack(push, 1)
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t align;
    uint32_t n_records;
    uint32_t reserved;
    uint64_t records_offset;
    uint64_t data_file_bytes;
} KvlTrunkIndexHeader;

typedef struct {
    uint32_t layer;       /* KVL_TRUNK_GLOBAL_LAYER for model-global tensors */
    uint32_t kind;
    uint32_t dtype;
    uint32_t ndim;
    uint32_t dims[4];
    uint64_t file_offset;
    uint64_t read_bytes;  /* padded direct-I/O span */
    uint64_t payload_bytes;
} KvlTrunkRecord;
#pragma pack(pop)

#endif
