int kvl_router_noaux_tc_stack(const struct KvlRouterConfig *cfg,
                              const float *x,
                              const float *router_weight,
                              const float *correction_bias,
                              int *top_ids, float *top_weights);

#define kvl_router_noaux_tc kvl_router_noaux_tc_stack
#define kvl_matvec_q8_rowwise kvl_matvec_q8_rowwise_stack_router
#define kvl_moe_token_auto kvl_moe_token_q8_stack_router_auto
#include "../../src/q8_ops.c"
