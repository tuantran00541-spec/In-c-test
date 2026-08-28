#ifndef KVL_MLA_STATE_H
#define KVL_MLA_STATE_H

#include <stddef.h>
#include "kvl/ops.h"

/* V6a correctness/reference state.
 *
 * This deliberately stores expanded FP32 K/V so incremental decode can be proven against
 * the already-validated causal prefill kernel before introducing compressed MLA state.
 * It is NOT the final low-RAM cache representation.
 */
typedef struct {
    int capacity;
    int len;
    int num_heads;
    int qk_nope_dim;
    int qk_rope_dim;
    int v_head_dim;
    float *keys;   /* [capacity, num_heads, qk_nope_dim + qk_rope_dim] */
    float *values; /* [capacity, num_heads, v_head_dim] */
} KvlMlaState;

int kvl_mla_state_init(KvlMlaState *state, const KvlMlaConfig *cfg, int capacity);
void kvl_mla_state_reset(KvlMlaState *state);
void kvl_mla_state_free(KvlMlaState *state);
size_t kvl_mla_state_bytes(const KvlMlaState *state);

/* Decode exactly one sequential token. `position` must equal state->len.
 * The input is expected to already be input-RMS-normalized, matching
 * kvl_mla_prefill_bf16(). The current token's K/V is appended before causal attention. */
int kvl_mla_decode_bf16(float *out,
                        const float *x,
                        int position,
                        const KvlMlaConfig *cfg,
                        const KvlMlaBF16 *w,
                        KvlMlaState *state);

#endif
