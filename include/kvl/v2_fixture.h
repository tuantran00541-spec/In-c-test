#ifndef KVL_V2_FIXTURE_H
#define KVL_V2_FIXTURE_H
#include <stdint.h>
#define KVL_V2_FIXTURE_MAGIC "KVLV2OR1"
#pragma pack(push,1)
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t hidden;
    uint32_t expert_intermediate;
    uint32_t shared_intermediate;
    uint32_t n_experts;
    uint32_t top_k;
    uint32_t n_group;
    uint32_t topk_group;
    uint32_t layer;
    uint32_t norm_topk_prob;
    float routed_scaling_factor;
    uint64_t off_x;
    uint64_t off_router;
    uint64_t off_bias;
    uint64_t off_shared_gate;
    uint64_t off_shared_up;
    uint64_t off_shared_down;
    uint64_t off_expected_ids;
    uint64_t off_expected_weights;
    uint64_t off_expected_out;
    uint64_t file_bytes;
} KvlV2FixtureHeader;
#pragma pack(pop)
#endif
