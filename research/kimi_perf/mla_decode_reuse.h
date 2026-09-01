#ifndef KVL_RESEARCH_MLA_DECODE_REUSE_H
#define KVL_RESEARCH_MLA_DECODE_REUSE_H

#include "kvl/mla_compressed_state.h"

int kvl_mla_decode_compressed_reuse_bf16(float *out,
                                          const float *x,
                                          int position,
                                          const KvlMlaConfig *cfg,
                                          const KvlMlaBF16 *w,
                                          KvlMlaCompressedState *state);
void kvl_mla_decode_reuse_release(void);

#endif
