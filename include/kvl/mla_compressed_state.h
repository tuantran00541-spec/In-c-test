#ifndef KVL_MLA_COMPRESSED_STATE_H
#define KVL_MLA_COMPRESSED_STATE_H

#include <stddef.h>
#include "kvl/ops.h"

/* Production-oriented MLA state: for each layer/token retain only the normalized
 * kv_lora latent and the rotated positional-key component. Historical full per-head
 * K/V vectors are never materialized in the persistent cache. */
typedef struct {
    int capacity;
    int len;
    int kv_lora_rank;
    int qk_rope_dim;
    float *latent; /* [capacity, kv_lora_rank] */
    float *rope;   /* [capacity, qk_rope_dim] */
} KvlMlaCompressedState;

int kvl_mla_compressed_state_init(KvlMlaCompressedState *state,
                                  const KvlMlaConfig *cfg,
                                  int capacity);
void kvl_mla_compressed_state_reset(KvlMlaCompressedState *state);
/* Roll back only the logical tail. Stale entries are overwritten by subsequent appends.
 * This is the primitive needed by exact speculative verification after a rejected suffix. */
int kvl_mla_compressed_state_truncate(KvlMlaCompressedState *state, int new_len);
void kvl_mla_compressed_state_free(KvlMlaCompressedState *state);
size_t kvl_mla_compressed_state_bytes(const KvlMlaCompressedState *state);

/* Fill persistent compressed history for a causal prompt batch without running the
 * attention output path. `x` is the already-normalized attention input [seq_len,H].
 * This is used by V8 layer-major batch prefill after kvl_mla_prefill_bf16() computes
 * the causal attention outputs for the same prompt. State must be empty. */
int kvl_mla_compressed_state_prefill_bf16(const float *x,
                                          int seq_len,
                                          const KvlMlaConfig *cfg,
                                          const KvlMlaBF16 *w,
                                          KvlMlaCompressedState *state);

/* One-token absorbed MLA decode.
 *
 * For the no-PE key path:
 *   q_nope dot (W_k * latent_j) == (W_k^T * q_nope) dot latent_j
 *
 * For values:
 *   sum_j p_j (W_v * latent_j) == W_v * (sum_j p_j latent_j)
 *
 * This keeps the persistent history compressed while remaining mathematically
 * equivalent to the expanded K/V reference path. */
int kvl_mla_decode_compressed_bf16(float *out,
                                   const float *x,
                                   int position,
                                   const KvlMlaConfig *cfg,
                                   const KvlMlaBF16 *w,
                                   KvlMlaCompressedState *state);

/* Exact continuation block primitive for speculative verification. The inputs are
 * consecutive tokens beginning at start_position==state->len. It deliberately reuses
 * the one-token arithmetic in-order so its first implementation is an oracle-quality
 * semantic building block; the full decoder can still hoist trunk loads layer-major. */
int kvl_mla_decode_compressed_block_bf16(float *out,
                                         const float *x,
                                         int count,
                                         int start_position,
                                         const KvlMlaConfig *cfg,
                                         const KvlMlaBF16 *w,
                                         KvlMlaCompressedState *state);

#endif
