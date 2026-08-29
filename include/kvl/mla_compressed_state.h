#ifndef KVL_MLA_COMPRESSED_STATE_H
#define KVL_MLA_COMPRESSED_STATE_H

#include <stddef.h>
#include "kvl/ops.h"

typedef struct {
    int capacity;
    int len;
    int kv_lora_rank;
    int qk_rope_dim;
    float *latent;
    float *rope;
} KvlMlaCompressedState;

int kvl_mla_compressed_state_init(KvlMlaCompressedState *state,
                                  const KvlMlaConfig *cfg,
                                  int capacity);
void kvl_mla_compressed_state_reset(KvlMlaCompressedState *state);

int kvl_mla_compressed_state_truncate(KvlMlaCompressedState *state, int new_len);
void kvl_mla_compressed_state_free(KvlMlaCompressedState *state);
size_t kvl_mla_compressed_state_bytes(const KvlMlaCompressedState *state);

int kvl_mla_compressed_state_prefill_bf16(const float *x,
                                          int seq_len,
                                          const KvlMlaConfig *cfg,
                                          const KvlMlaBF16 *w,
                                          KvlMlaCompressedState *state);

int kvl_mla_decode_compressed_bf16(float *out,
                                   const float *x,
                                   int position,
                                   const KvlMlaConfig *cfg,
                                   const KvlMlaBF16 *w,
                                   KvlMlaCompressedState *state);

int kvl_mla_decode_compressed_block_bf16(float *out,
                                         const float *x,
                                         int count,
                                         int start_position,
                                         const KvlMlaConfig *cfg,
                                         const KvlMlaBF16 *w,
                                         KvlMlaCompressedState *state);

#endif
