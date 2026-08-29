# V9 speed/quality lab

Branch: `research/v9-speed-quality-lab`

This branch is intentionally experimental. `main` remains the correctness/release baseline.
The target hardware is low-RAM CPU-first inference with SSD-resident sparse MoE experts.

## Baseline bottleneck

The V9 release stores approximately:

- trunk: 2.916 GiB BF16
- routed experts: 26.812 GiB BF16
- vision: 0.834 GiB BF16
- expert cache: 512 MiB = 31 x ~16.5 MiB expert records

On the accepted GitHub 2-thread run, decode was ~10.87 s/token and expert traffic was tens of
GiB per short generation. Therefore routed-expert bandwidth/compute is the first lab target.
Compressed MLA KV state is already small, so KV-cache quantization is lower priority for this model.

## Research candidates

### 1. Routed-expert low-bit quantization — highest priority

Relevant work:

- AWQ — Activation-aware Weight Quantization: https://arxiv.org/abs/2306.00978
- T-MAC — CPU low-bit LUT inference: https://arxiv.org/abs/2407.00088
- AQLM — additive quantization with CPU kernels: https://arxiv.org/abs/2401.06118
- QQQ — quality W4A8 quantization: https://arxiv.org/abs/2406.09904
- SpinQuant — learned rotations for low-bit accuracy: https://arxiv.org/abs/2405.16406
- QuaRot — rotation-based outlier removal: https://arxiv.org/abs/2404.00456

Lab sequence:

1. Q8 symmetric per-row routed experts only; keep router/shared/trunk/vision BF16.
2. Measure exact kernel error and real expert MLP error.
3. Wire Q8 store into the routed MoE path and compare official-token trajectories.
4. Only after Q8 is safe, evaluate Q4 with activation-aware scaling / outlier protection.
5. Consider LUT-style kernels inspired by T-MAC if AVX2 dequantized dot products are still compute-bound.

Expected Q8 record size is roughly half BF16, so a 512 MiB cache should hold about twice as many
experts and SSD traffic should fall close to 2x before any further kernel speedup.

### 2. Expert prediction, prefetch and cache warmup — high priority after quantization

Relevant work:

- Pre-Attention Expert Prediction and Prefetching: https://arxiv.org/abs/2511.10676
- SP-MoE speculative expert prefetching: https://arxiv.org/abs/2510.10302
- MoE-SpeQ speculative quantized decoding/prefetch: https://arxiv.org/abs/2511.14102
- SliceMoE predictive cache warmup / bit-sliced caching: https://arxiv.org/abs/2512.12990
- SPICE speculative prefetching: https://arxiv.org/abs/2608.21240

Kimi-specific idea: record routed expert frequencies and transitions during multimodal/text prefill,
then warm the 512 MiB cache for decode. A later version can train a tiny predictor for next-layer or
future-token expert IDs. Exact target routing must remain authoritative; a wrong prediction may waste
I/O but must never change the selected experts.

### 3. Speculative decoding — medium priority

Relevant work:

- EAGLE-3: https://arxiv.org/abs/2503.01840
- Medusa: https://arxiv.org/abs/2401.10774
- Lookahead Decoding: https://arxiv.org/abs/2402.02057
- LayerSkip: https://arxiv.org/abs/2404.16710

These methods can reduce target-model decode steps, but most gains assume accelerator parallelism,
a trained draft/head, or a target model whose verification pass is cheap enough. On this CPU/SSD MoE
runtime, a draft model also needs to predict expert demand early enough to hide storage I/O. Therefore
this comes after expert quantization/prefetch.

### 4. Visual-token pruning — lower priority for the current 196x196 acceptance image

Relevant work:

- FitPrune: https://arxiv.org/abs/2409.10197
- TopV: https://arxiv.org/abs/2503.18278

The current acceptance images produce only 49 media tokens, so pruning them is unlikely to beat expert
optimization. Revisit this for large images / multi-image prompts.

### 5. KV-cache quantization — useful mainly for very long context

Relevant work:

- KIVI: https://arxiv.org/abs/2402.02750
- KVQuant: https://arxiv.org/abs/2401.18079

V9 already uses compressed MLA latent+RoPE state, so KV quantization is not the main decode bottleneck.
It can become useful if context grows enough that compressed FP32 MLA state becomes significant.

## Quality policy

Speed changes are accepted only after comparison against the pinned official checkpoint revision used
by V9. Low-bit work must report numerical error and then run real token-trajectory/semantic tests. A
faster branch is not a release candidate merely because it builds or produces text.
