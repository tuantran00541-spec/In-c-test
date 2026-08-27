#ifndef KVL_V4_FIXTURE_H
#define KVL_V4_FIXTURE_H
#include <stdint.h>
#define KVL_V4_FIXTURE_MAGIC "KVLV4OR1"
#pragma pack(push,1)
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t seq_len;
    uint32_t hidden;
    uint32_t num_heads;
    uint32_t qk_nope_dim;
    uint32_t qk_rope_dim;
    uint32_t v_head_dim;
    uint32_t kv_lora_rank;
    uint32_t dense_intermediate;
    uint32_t expert_intermediate;
    uint32_t shared_intermediate;
    uint32_t n_experts;
    uint32_t top_k;
    uint32_t n_group;
    uint32_t topk_group;
    uint32_t first_layer;
    uint32_t n_layers;
    uint32_t norm_topk_prob;
    float rms_eps;
    float rope_theta;
    float routed_scaling_factor;
    uint64_t off_x;
    uint64_t off_expected_after_dense;
    uint64_t off_expected_ids;
    uint64_t off_expected_weights;
    uint64_t off_expected_final;
    uint64_t file_bytes;
} KvlV4FixtureHeader;
#pragma pack(pop)
#endif
