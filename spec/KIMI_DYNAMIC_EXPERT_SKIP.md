# Kimi-VL dynamic expert skipping pilot

Status: research-only, full-Q8 correctness/performance experiment. This is not a production path and is not a pruning result.

## Scope

The pilot keeps the complete Kimi-VL routed-expert store and the released router unchanged. It targets routed-expert work during prompt prefill, especially `<|media_pad|>` tokens. It deliberately does **not** combine:

- static `KVL_MOE_MASK` pruning,
- Q5 expert compression,
- visual-token reduction,
- decode-stage expert skipping.

Those effects must be measured independently before any composition experiment.

## Runtime semantics

1. Run the released `noaux_tc` router and obtain the original top-k IDs and weights.
2. Normalize those already-selected route weights only for the skip decision.
3. Apply a token-family + layer threshold from `KVL_MOE_DYNSKIP_POLICY`.
4. Enforce `min_keep` by restoring the largest normalized route masses if necessary.
5. Set skipped routed weights to zero.
6. Do **not** choose replacement experts.
7. Do **not** renormalize the surviving routed weights.
8. Request only surviving expert IDs from the expert cache, then execute only those routed experts.
9. Always execute the shared expert unchanged.

This placement is intentional: a skipped route can avoid both routed-expert matvecs and the corresponding cache/store request. Actual physical SSD bytes saved still depend on cache state and must be measured rather than inferred from skip count.

## Token families

The exact prompt token IDs are classified with the released Kimi chat structure:

- `control`: entire system span, role/header/boundary tokens, media wrapper tokens, assistant transition;
- `content`: natural-language user content outside the media wrapper;
- `media`: `<|media_pad|>` visual-token placeholders.

Control tokens are hard-protected. Decode is hard-protected in this pilot.

Because the current prefill loop is layer-major and token-serial inside each layer, the research dispatcher can recover token position from the MoE call order after loading the exact prompt IDs. If that call order changes, the dispatcher fails closed rather than silently applying the wrong token family.

## Environment

- `KVL_MOE_DYNSKIP_POLICY=/path/to/policy`
- `KVL_MOE_DYNSKIP_PROMPT_IDS=/path/to/exact/prompt.ids`
- optional `KVL_MOE_DYNSKIP_STATS=/path/to/stats.tsv`

`KVL_MOE_MASK` is rejected whenever dynamic skipping is active.

Policy format:

```text
# family layer normalized_topk_mass_threshold min_keep
media 22 0.12 5
```

Only `content` and `media` entries are accepted. Layer numbering matches the decoder layer index used by the native runtime; Kimi routed MoE layers are 1..26.

## Evidence ladder

A green build is not a quality result. Promotion requires, in order:

1. portable unit tests for token classification, thresholding, `min_keep`, and policy parsing;
2. trace simulator agreement with the pre-intervention layer of a real run;
3. same-prefix first-token logit comparison using probability overlap / total variation / JS plus exact greedy token ID;
4. multiple held-out VL cases covering description, counting, OCR, spatial/pattern, English and Vietnamese;
5. autoregressive generation checks;
6. measured cache/store counters and TTFT / prefill timing on repeated A/B runs;
7. only then consider more aggressive thresholds or composition with visual-token reduction.

## Initial policy

`tests/data/kimi-dynskip-media-l22-26-t012-k5.policy` is intentionally conservative: media only, layers 22..26, normalized route-mass threshold 0.12, and at least five of the released top six routed experts retained. It is a correctness pilot, not a recommended deployment policy.

## Claim boundary

A route-count reduction is not a speedup. A first-token argmax match is not a quality proof. A single CI timing is not a benchmark. Any speed statement requires measured runtime/cache evidence; any quality statement must name the tested scope.
