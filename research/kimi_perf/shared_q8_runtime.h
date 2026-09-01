#ifndef KVL_RESEARCH_SHARED_Q8_RUNTIME_H
#define KVL_RESEARCH_SHARED_Q8_RUNTIME_H

#include "kvl/ops.h"

int kvl_moe_token_q8_shared_sidecar_auto(KvlExpertCache *cache, int layer,
                                          const KvlRouterConfig *router_cfg,
                                          const float *x,
                                          const float *router_weight,
                                          const float *correction_bias,
                                          int expert_intermediate_size,
                                          const KvlMlpBF16 *shared,
                                          float *out,
                                          int *top_ids,
                                          float *top_weights,
                                          float *scratch);

void kvl_shared_q8_sidecar_report(void);
void kvl_shared_q8_sidecar_close(void);

#endif
