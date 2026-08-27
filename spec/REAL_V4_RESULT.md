# Real-weight V4 result

Date: 2026-08-27

Model: `moonshotai/Kimi-VL-A3B-Instruct`

Scope: four-token causal prefill through official decoder layers 0 -> 1. Layer 0 is the
model's dense first layer; layer 1 is sparse MoE. Resident tensors come from the V4
`trunk.bin/trunk.idx` backing store and routed experts come from `experts.bin/experts.idx`.
This is not yet a full 27-layer logits test.

## Bounded-shard conversion

Layer 0 resolved to:

- `model-00001-of-00007.safetensors`
- 11 checkpoint tensors matched by the resolver

Its runtime tensors were packed first, then the raw shard was deleted before layer 1 was
downloaded.

Layer 1 resolved to:

- `model-00003-of-00007.safetensors`
- 205 checkpoint tensors

After both layers were packed:

```text
trunk records = 22
trunk data    = 0.213 GiB
trunk index   = 1,272 bytes
experts data  = 1.031 GiB
experts index = 5,168 bytes
```

This validates the conversion strategy where a source checkpoint shard can be consumed,
converted into the runtime format, and deleted before the next large source shard is
fetched.

## Four-token routing

The real oracle selected 17 unique layer-1 routed experts:

```text
[7, 11, 16, 24, 25, 27, 29, 30, 32, 35, 39, 40, 45, 52, 54, 55, 62]
```

Per token:

```text
t0 [16, 55, 7, 29, 62, 35]
t1 [7, 55, 32, 30, 27, 11]
t2 [24, 54, 35, 62, 52, 40]
t3 [30, 55, 35, 39, 45, 25]
```

## C runtime comparison

The complete real two-layer stack passed:

```text
dense_layer_max = 5.9604645e-08
router_ids       = OK
max_weight_abs   = 8.9406967e-08
final_max        = 1.1920929e-07
final_rms        = 1.765652e-08
trunk_direct_io  = yes
expert_direct_io = yes
```

Expert cache was deliberately capped at 256 MiB although the layer-1 routed pool is
1.031 GiB:

```text
slots      = 15
slot       = 16.50 MiB
arena      = 247.50 / 256.00 MiB
requests   = 24
prefetch   = 17 experts / 4 batches
reads      = 17
bytes read = 280.50 MiB
evictions  = 2
failures   = 0
```

The reported compute-side `hits=24` are expected because `getmany()` materializes the
selected experts before each token's per-expert `get()` calls.

## What V4 proves

The following path is now validated against official weights:

```text
checkpoint shard for layer 0
  -> bounded-shard converter
  -> aligned trunk backing store
  -> dense decoder layer 0
  -> free layer-0 working tensors
  -> aligned trunk backing store for layer 1
  -> MLA + router/shared tensors
  -> hard-budget direct-I/O routed expert cache
  -> sparse decoder layer 1
  -> C result vs Torch oracle
```

Both backing stores used direct I/O in the passing run.

## Next: V5 logits

Generalize the two-layer probe into an arbitrary decoder loop, pack all 27 decoder layers
plus global token embeddings, final RMSNorm and LM head, and compare full-prefill text
logits. Only after full-prefill logits agree should incremental MLA/KV caching be added.

GitHub Actions run: `33092195221` (`Real Kimi-VL V4 stack`, run #1).
