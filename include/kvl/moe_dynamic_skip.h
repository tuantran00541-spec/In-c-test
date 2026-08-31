#ifndef KVL_MOE_DYNAMIC_SKIP_H
#define KVL_MOE_DYNAMIC_SKIP_H

#include "kvl/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    KVL_MOE_FAMILY_CONTROL = 0,
    KVL_MOE_FAMILY_CONTENT = 1,
    KVL_MOE_FAMILY_MEDIA = 2,
    KVL_MOE_FAMILY_COUNT = 3,
};

/* Research-only Kimi chat-template classifier used to protect all structural
 * tokens while distinguishing natural-language content from <|media_pad|>.
 * out_family must have n entries. */
int kvl_moe_dynskip_classify_prompt(const int *prompt, int n,
                                    unsigned char *out_family);

/* Policy entries threshold the normalized mass of the already-selected top-k
 * route. They never reroute and never renormalize surviving weights.
 * Control tokens are hard-protected regardless of policy contents. */
void kvl_moe_dynskip_reset_policy(void);
int kvl_moe_dynskip_set_policy(int family, int layer,
                               float threshold, int min_keep);
int kvl_moe_dynskip_load_policy(const char *path);

/* Pure decision helper for unit tests and offline calibration. */
int kvl_moe_dynskip_apply_policy(int family, int layer, int top_k,
                                 const float *top_weights,
                                 unsigned char *keep,
                                 int *out_skipped);

/* Q8-only runtime dispatch. When KVL_MOE_DYNSKIP_POLICY is set, this consumes
 * KVL_MOE_DYNSKIP_PROMPT_IDS to infer the token family from the exact layer-major
 * prefill call order. Decode is automatically protected. Static KVL_MOE_MASK is
 * rejected so pruning and dynamic skipping cannot be accidentally mixed. */
int kvl_moe_token_q8_dynskip_auto(KvlExpertCache *cache, int layer,
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

#ifdef __cplusplus
}
#endif

#endif
