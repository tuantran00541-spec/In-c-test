# V9 speed/quality lab

Branch: `research/v9-speed-quality-lab`

This branch is intentionally experimental. `main` remains the correctness/release baseline.
The target hardware is low-RAM CPU-first inference with SSD-resident sparse MoE experts.

## Baseline bottleneck

The original V9 BF16 release stores approximately:

- trunk: 2.916 GiB BF16
- routed experts: 26.812 GiB BF16
- vision: 0.834 GiB BF16
- expert cache: 512 MiB = 31 x ~16.5 MiB expert records

On the accepted GitHub 2-thread run, decode was ~10.87 s/token and expert traffic was tens of
GiB per short generation. Therefore routed-expert bandwidth/compute was the first lab target.

## Q8 routed-expert stage — DONE

Q8 symmetric per-row quantization is applied only to routed experts. Router weights, shared experts,
trunk and vision remain BF16. The production Q8 path is opt-in (`--expert-format q8`); BF16 remains
the compatibility default.

The Q8 implementation passed synthetic kernel tests, real official-weight single-expert tests,
full top-6 routed+shared MoE tests, Windows/MSVC/direct-I/O sanity, and full multimodal token-trajectory
gates. It was merged to `main` in PR #1 at commit
`ef4ad15ed6486adcaff72c14e2982d22d9fbc40b`.

### Q8 measured result

On the exact English V9 image/prompt with the pinned Kimi checkpoint:

- routed expert store: 26.812 GiB BF16 -> 13.438 GiB Q8
- 512 MiB cache capacity: 31 -> 61 expert slots
- first token: ~134.6 s BF16 -> ~41.5 s Q8 (~3.24x faster)
- average next token: ~10.87 s BF16 -> ~7.59 s Q8 (~1.43x faster)
- text total: ~297.6 s BF16 -> ~155.4 s Q8 (~1.91x faster)
- expert SSD traffic: ~88.65 GiB -> ~31.43 GiB (~64.5% lower)
- generated output/token IDs: exact 16/16 match to the accepted BF16 trajectory

### Q8 cache sweep

Run `33231711448` tested one packed Q8 model with four cache budgets. All four runs preserved the
same accepted 16/16 generated tokens.

| Cache | Approx slots | Avg next token | Text total | Expert reads |
| --- | ---: | ---: | ---: | ---: |
| 512 MiB | 61 | 7.635 s | 156.45 s | 3801 |
| 1024 MiB | 123 | 7.118 s | 147.64 s | lower than 512 |
| 1536 MiB | 185 | 6.308 s | 136.12 s | lower than 1024 |
| 2048 MiB | 247 | 5.824 s | 128.83 s | 2343 |

For the current 4 GiB text-phase budget, 2048 MiB was the best tested cache size. The runtime planner
reported about 3590 MiB / 4096 MiB for the 90-token V9 prompt, leaving limited but valid headroom.
This is a configuration result, not a new default: users can select it with `--cache-mib 2048`.

### Next-layer route prediction

Run `33231931305` measured a cheap predictor that applies the next layer's real router to hidden states
available from the current layer. The authoritative next-layer router remained unchanged.

- current-layer `n2` -> next-layer top-6 recall: 83.29% (best)
- post-layer `x` recall: 79.87%
- `r1` recall: 75.21%
- exact 6/6 `n2` prediction: 30.77%
- generation remained exact 16/16 tokens

This was accurate enough to justify a real speculative-I/O benchmark, but prediction accuracy alone
was not considered a speed result.

### Attention-overlapped speculative prefetch — REJECTED

Run `33234380050` compared baseline Q8+2048 MiB against a safe diagnostic prefetch implementation on
the same runner, packed model, exact image/prompt and 16-token greedy trajectory. Predicted experts for
layer L were read concurrently only while attention(L) executed; the worker joined before the real MoE
route used the cache. Therefore prediction errors could waste I/O/cache capacity but could not change
model math.

| Metric | Q8 + 2048 baseline | + speculative prefetch |
| --- | ---: | ---: |
| First token | 41.624 s | 38.403 s |
| Avg next token | 6.212 s | 6.766 s |
| Text total | 134.824 s | 139.912 s |
| Expert reads | 2816 | 3198 |
| Expert bytes | 23287.00 MiB | 26445.96 MiB |
| Evictions | 2569 | 2951 |
| Exact accepted tokens | yes | yes |

The prefetcher improved first-token latency by about 8.4%, but made sustained decode about 8.9% slower,
made total text time about 3.8% slower, and increased expert reads/bytes by 13.57%. It issued 390
speculative batches, loaded 1472 predicted experts, had zero I/O failures, and preserved exact token
parity.

**Decision: do not merge this blind top-6 prefetch strategy.** The 2 GiB LRU already has a high hit
rate, so incorrect speculative admissions add cache pollution and SSD traffic. If expert prefetch is
revisited, test confidence-gated subsets, non-polluting staging buffers, or layer-aware admission rather
than blindly admitting all six predicted experts.

This closes the current Q8 optimization stage. Q8 itself is production-worthy and is already on
`main`; the losing prefetch diagnostic remains lab-only.

## Research candidates after Q8

### 1. Routed-expert lower-bit kernels — later

Relevant work:

- AWQ — Activation-aware Weight Quantization: https://arxiv.org/abs/2306.00978
- T-MAC — CPU low-bit LUT inference: https://arxiv.org/abs/2407.00088
- AQLM — additive quantization with CPU kernels: https://arxiv.org/abs/2401.06118
- QQQ — quality W4A8 quantization: https://arxiv.org/abs/2406.09904
- SpinQuant — learned rotations for low-bit accuracy: https://arxiv.org/abs/2405.16406
- QuaRot — rotation-based outlier removal: https://arxiv.org/abs/2404.00456

Q4 remains interesting, but only with activation-aware/outlier protection and the same official-weight
quality gates used for Q8. A LUT-style CPU kernel inspired by T-MAC is also worth revisiting if expert
compute remains dominant after long-context work.

### 2. Expert prediction/prefetch — only with selective admission

Relevant work:

- Pre-Attention Expert Prediction and Prefetching: https://arxiv.org/abs/2511.10676
- SP-MoE speculative expert prefetching: https://arxiv.org/abs/2510.10302
- MoE-SpeQ speculative quantized decoding/prefetch: https://arxiv.org/abs/2511.14102
- SliceMoE predictive cache warmup / bit-sliced caching: https://arxiv.org/abs/2512.12990
- SPICE speculative prefetching: https://arxiv.org/abs/2608.21240

The Kimi `n2` predictor is promising, but the first blind top-6 admission policy lost overall. Any new
version must explicitly measure useful-prefetch precision, cache pollution and extra bytes read.

### 3. Speculative decoding — medium priority

Relevant work:

- EAGLE-3: https://arxiv.org/abs/2503.01840
- Medusa: https://arxiv.org/abs/2401.10774
- Lookahead Decoding: https://arxiv.org/abs/2402.02057
- LayerSkip: https://arxiv.org/abs/2404.16710

These methods can reduce target-model decode steps, but most gains assume accelerator parallelism,
a trained draft/head, or cheap verification. On this CPU/SSD MoE runtime, this remains behind
long-context attention/KV work.

### 4. Visual-token pruning — lower priority for current fixtures

Relevant work:

- FitPrune: https://arxiv.org/abs/2409.10197
- TopV: https://arxiv.org/abs/2503.18278

The current 196x196 acceptance images produce only 49 media tokens. Revisit this for large images or
multi-image prompts.

### 5. KV-cache / long-context optimization — next research target

V9 already stores compressed MLA latent+RoPE state rather than full MHA K/V, so existing KV-cache
papers cannot be copied mechanically. Current custom state is FP32: 512 latent + 64 RoPE values per
layer/token across 27 layers, about 60.75 KiB/token. This is ~972 MiB at 16k context before prefill
working memory.

First Kimi-specific candidates:

1. store MLA latent in BF16 and keep RoPE BF16 as a conservative baseline;
2. test row/token-wise Q8 latent while retaining RoPE at BF16;
3. only after low-bit state passes official-weight quality gates, evaluate importance-aware mixed
   precision or token retention/pruning;
4. independently replace the current O(n^2), fully-materialized long-prompt attention prefill with a
   tiled/streaming implementation so 16k context is practical in time and temporary memory.

Relevant starting papers include KIVI, KVQuant, SnapKV, H2O/Heavy-Hitter Oracle, StreamingLLM,
MiniCache and newer attention-aware adaptive compression work. These should be adapted to MLA rather
than treated as drop-in implementations.

## Quality policy

Speed changes are accepted only after comparison against the pinned official checkpoint revision used
by V9. Low-bit or cache-lossy work must report numerical error and then run real token-trajectory and
semantic tests. A faster branch is not a release candidate merely because it builds or produces text.
