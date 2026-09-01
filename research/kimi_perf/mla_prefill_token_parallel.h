#ifndef KVL_RESEARCH_MLA_PREFILL_TOKEN_PARALLEL_H
#define KVL_RESEARCH_MLA_PREFILL_TOKEN_PARALLEL_H

#include "kvl/mla_compressed_state.h"

int kvl_mla_prefill_compressed_token_parallel_bf16(float *out,
                                                    const float *x,
                                                    int seq_len,
                                                    const KvlMlaConfig *cfg,
                                                    const KvlMlaBF16 *w,
                                                    KvlMlaCompressedState *state);

#endif
