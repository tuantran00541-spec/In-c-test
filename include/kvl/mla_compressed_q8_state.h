#ifndef KVL_MLA_COMPRESSED_Q8_STATE_H
#define KVL_MLA_COMPRESSED_Q8_STATE_H

#include <stddef.h>
#include <stdint.h>
#include "kvl/ops.h"

/* Experimental Kimi MLA history compression.
 *
 * The normalized kv_lora latent is shared by the absorbed key path and value mix,
 * so the usual KIVI split (key per-channel, value per-token) cannot be applied
 * literally.  The first lab format therefore uses conservative symmetric INT8
 * per token for the 512-d latent, while keeping the 64-d rotated positional key
 * in BF16.  One FP32 scale is stored per token.
 *
 * This is intentionally a lab type, separate from the exact FP32 production
 * KvlMlaCompressedState, until official-weight trajectory/quality gates exist. */
typedef struct {
    int capacity;
    int len;
    int kv_lora_rank;
    int qk_rope_dim;
    int8_t *latent_q8;       /* [capacity, kv_lora_rank] */
    float *latent_scale;     /* [capacity] */
    uint16_t *rope_bf16;     /* [capacity, qk_rope_dim] */
} KvlMlaCompressedQ8State;

int kvl_mla_compressed_q8_state_init(KvlMlaCompressedQ8State *state,
                                     const KvlMlaConfig *cfg,
                                     int capacity);
void kvl_mla_compressed_q8_state_reset(KvlMlaCompressedQ8State *state);
int kvl_mla_compressed_q8_state_truncate(KvlMlaCompressedQ8State *state,
                                         int new_len);
void kvl_mla_compressed_q8_state_free(KvlMlaCompressedQ8State *state);
size_t kvl_mla_compressed_q8_state_bytes(const KvlMlaCompressedQ8State *state);

int kvl_mla_compressed_q8_state_prefill_bf16(const float *x,
                                             int seq_len,
                                             const KvlMlaConfig *cfg,
                                             const KvlMlaBF16 *w,
                                             KvlMlaCompressedQ8State *state);

int kvl_mla_decode_compressed_q8_bf16(float *out,
                                      const float *x,
                                      int position,
                                      const KvlMlaConfig *cfg,
                                      const KvlMlaBF16 *w,
                                      KvlMlaCompressedQ8State *state);

#endif
