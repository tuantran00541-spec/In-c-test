#ifndef KVL_RESEARCH_MLA_PREFILL_FUSED_H
#define KVL_RESEARCH_MLA_PREFILL_FUSED_H

#include "kvl/mla_compressed_state.h"

int kvl_mla_prefill_compressed_fused_bf16(float *out,
                                           const float *x,
                                           int seq_len,
                                           const KvlMlaConfig *cfg,
                                           const KvlMlaBF16 *w,
                                           KvlMlaCompressedState *state);

#endif
