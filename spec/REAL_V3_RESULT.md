# Real-weight V3 result

Date: 2026-08-27

Model: `moonshotai/Kimi-VL-A3B-Instruct`

Scope: a complete **4-token causal prefill through decoder layer 1** using official model
weights. Routed experts are read through the low-RAM direct-I/O cache; attention and other
layer weights are resident for this probe. This is not yet a multi-layer or full-model run.

## Checkpoint subset

All 205 tensors for decoder layer 1 resolve to one official shard:

- `model-00003-of-00007.safetensors`
- 6 self-attention tensors
- 3 layer-norm tensors reported by the prefix scan
- 192 routed-expert tensors = 64 experts x 3 matrices
- 3 shared-expert tensors
- 2 router tensors

The routed experts were repacked to `experts.bin` = 1.031 GiB plus a 5,168-byte index.

## Four-token routing

The deterministic V3 oracle selected 18 unique routed experts across 24 expert calls:

```text
token 0: [5, 26, 34, 51, 55, 53]
token 1: [27, 4, 2, 34, 57, 3]
token 2: [20, 14, 35, 4, 11, 53]
token 3: [0, 11, 2, 3, 45, 32]
```

Unique expert ids:

```text
[0, 2, 3, 4, 5, 11, 14, 20, 26, 27, 32, 34, 35, 45, 51, 53, 55, 57]
```

## Numerical comparison

The complete C decoder layer matched the manual PyTorch oracle:

```text
norm1_max      = 1.1920929e-07
attn_max       = 5.2154064e-08
resid1_max     = 5.9604645e-08
norm2_max      = 8.9406967e-08
router_ids     = OK
max_weight_abs = 1.1920929e-07
final_max      = 1.1920929e-07
final_rms      = 1.4125544e-08
direct_io      = yes
```

This validates the V3 FP32-activation numerical contract for:

```text
input
  -> RMSNorm
  -> Kimi-VL MLA
       q projection
       compressed KV projection
       KV RMSNorm
       KV expansion
       exact Kimi-VL RoPE permutation/rotation
       causal scaled dot-product attention
       output projection
  -> residual
  -> RMSNorm
  -> real top-6 router
  -> direct-I/O streamed routed experts
  -> shared expert
  -> residual
```

## Forced low-memory expert cache

The real probe deliberately used a **256 MiB cache** against the 1.031 GiB routed-expert
pack, so the layer's expert pool could not remain resident:

```text
slots       = 15
slot        = 16.50 MiB
arena       = 247.50 / 256.00 MiB
requests    = 24
prefetch    = 18 reads across 4 batches
physical IO = 297.00 MiB
evictions   = 3
failures    = 0
direct_io   = yes
```

`hit=24, miss=0` at compute time means each token's `getmany(top-6)` completed before the
per-expert `get()` calls. Physical I/O still happened: 18 expert records were read, totaling
297 MiB, and three cache evictions occurred.

## What this proves — and what it does not

V3 proves that a complete real Kimi-VL decoder layer can be evaluated in C while its routed
expert pool is larger than the configured expert cache and is serviced from the backing
store. It also validates the current MLA/RoPE interpretation against a Python oracle over a
non-trivial 4-token causal sequence.

It does **not** yet prove:

- all 27 decoder layers chained together;
- incremental decode / persistent compressed KV cache;
- embedding, tokenizer, final norm, LM head, or real logits;
- native BF16 activation rounding;
- MXFP4/Q4 expert kernels;
- MoonViT / vision input.

Those are later milestones.

GitHub Actions run: `33090681334` (`Real Kimi-VL V3 decoder layer`, run #1).
