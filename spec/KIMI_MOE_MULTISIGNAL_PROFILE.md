# Kimi-VL multi-signal MoE profiler

Status: research instrumentation and offline profiling only. This does not
promote a pruning mask and does not change the production routing result when
tracing is disabled.

## Why this replaces scalar-only screening

The protected56 experiment showed non-monotonic mask interactions and failed
the strict text gate at 12/14 exact cases. A nested scalar REAP order therefore
is not a safe pruning oracle for the general multimodal target.

The new profiler keeps the old REAP analyzer and masks as controls, but records
the signals needed to compare several hypotheses before another expensive
full-weight gate.

## Trace formats

`KVL_MOE_TRACE=/path/to/trace.tsv` keeps the established six-column format:

```text
event layer expert router_weight output_l2 saliency
```

`KVL_MOE_PROFILE_TRACE=/path/to/profile.tsv` opts into the seven-column v2
format:

```text
event layer expert router_weight output_l2 saliency output_max_abs
```

`output_max_abs` is measured on the routed expert output immediately after its
`down_proj`, before multiplication by the router weight and before summation
with other routed/shared experts. If both environment variables are set, the
profile trace takes precedence. With neither variable set, the extra scan is
not executed.

The offline profiler accepts both formats. Legacy traces produce `null`
max-absolute statistics and cannot support outlier classification.

## Implemented expert signals

For routed expert `j`, routed-token count `N_j`, router weight `g_j(t)`, and
expert output `f_j(t)`:

```text
route_frequency = N_j / token_layer_events
REAP = (1 / N_j) sum_t |g_j(t)| ||f_j(t)||_2
MAN  = (1 / N_j) sum_t ||f_j(t)||_2
MSAN = (1 / N_j) sum_t ||f_j(t)||_2^2
```

This is the relevant subset of the unified score family
`S_j(b, alpha, beta)`. REAP is `S(1,1,1)`, MAN is `S(1,0,1)`, and MSAN is
`S(1,0,2)`. Router-weight mean/max and output max-absolute P95/P99/P99.5 are
also reported per expert and domain.

Reference: [How to Score Experts for One-Shot MoE Expert Pruning](https://arxiv.org/abs/2606.15716).

## Output-outlier boundary

The Super Experts paper profiles the maximum magnitude of every expert's
`down_proj` output and requires all of:

1. expert maximum greater than the global P99.5 of expert maxima;
2. expert maximum greater than one tenth of the global maximum;
3. the expert belongs to a layer responsible for forming massive hidden-state
   activations.

The v2 trace directly supports the first two conditions. It does not yet trace
hidden-state massive-activation formation, so the report uses the label
`super_expert_like_candidate` and explicitly sets
`paper_se_layer_condition_available=false`. It must not be presented as proof
that L11E41 or any other Kimi expert is a Super Expert.

Reference: [Unveiling Super Experts in Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2507.23279).

## Domain coverage and collaboration evidence

Every input trace is a separately labelled sample/domain. The report preserves
independent domain profiles instead of collapsing them into one scalar. It also
emits:

- within-layer expert pair counts for the same token-layer event;
- a sparse sample activation vector containing the sum of absolute router
  weights for every `(layer, expert)` slot;
- per-domain MAN rankings;
- a deterministic round-robin coverage order across domain rankings.

The activation vectors are the input evidence needed for later cross-layer
correlation, graph, or dictionary-learning experiments. This first tool does
not pretend that pair counts alone recover collaborative expert groups.

References:

- [Specialization through Collaboration](https://aclanthology.org/2026.eacl-long.104/)
- [Generic Expert Coverage for Pruning Sparse MoE Language Models](https://arxiv.org/abs/2607.01710)

## Distribution overlap / ESAP boundary

`tools/compare_kimi_logits.py` now reports full-vocabulary probability overlap
and total variation for aligned records:

```text
overlap(p, q) = sum_v min(p_v, q_v) = 1 - TV(p, q)
```

This value is token-level ESAP only when the baseline and candidate logits were
computed from the same teacher-forced prefix. Existing autoregressive A/B dumps
can use the number as a distribution diagnostic, but must not call it ESAP once
their prefixes have drifted. A future search workflow needs an explicit
teacher-forced answer-token runner and per-sample averaging before using this as
EvoESAP fitness.

Reference: [EvoESAP](https://arxiv.org/abs/2603.06003).

## Recommended first real calibration

Collect v2 traces with one sample per file and explicit domain labels. Start
with disjoint calibration buckets for English, Vietnamese, factual QA,
reasoning, instruction following, long-form text, VL description, OCR,
counting, and spatial/pattern cases. Profile first; do not generate or promote a
new mask in the same job.

After inspecting coverage and outlier evidence:

1. freeze plausible protection rules;
2. construct multiple within-layer orders (at least MAN, MSAN, and REAP);
3. search non-uniform per-layer budgets near 56 total disabled slots;
4. use shared-prefix teacher-forced overlap to screen candidates;
5. run strict and semantic text/VL gates only on finalists.

Q5 mask support and physical Q5 compaction remain separate work and must not be
mixed into this profiling experiment.
