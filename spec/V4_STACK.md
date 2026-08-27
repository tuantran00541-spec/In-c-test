# V4: streamed trunk + multi-layer decoder stack

V4 moves non-routed decoder tensors out of correctness fixtures and into the runtime's
own backing store. The model is now split into two independent aligned files:

```
trunk.bin / trunk.idx       attention, norms, dense MLP, routers, shared experts
experts.bin / experts.idx   routed expert gate/up/down matrices
```

Both stores use 4096-byte aligned records and positioned direct I/O where available.
The V4 correctness probe streams one layer's trunk tensors, computes the layer, frees
those tensors, then advances to the next layer. This is deliberately a simple precursor
to a layer-contiguous two-slot ring buffer.

## Milestone scope

The first V4 stack is layers 0 -> 1 because it crosses the architecture boundary:

- layer 0: dense MLP (`intermediate_size=11264` in the released model);
- layer 1: sparse MoE (`64` routed experts, top-6, two shared experts).

A four-token causal prefill is used so RoPE, causal softmax and cross-token attention are
exercised. Layer 1 routed experts come through the V1 hard-budget cache.

## Low-disk real-weight workflow

The real CI job intentionally never keeps both large source shards at once:

1. download metadata + the shard(s) needed by layer 0;
2. pack layer 0 into `trunk.bin`;
3. delete raw layer-0 shard(s);
4. download the shard(s) needed by layer 1;
5. append layer-1 resident tensors to `trunk.bin`;
6. pack routed experts to `experts.bin`;
7. build the Torch oracle from packed trunk tensors plus only selected layer-1 experts;
8. compare the C stack under a 256 MiB expert-cache budget.

This makes the packer itself usable as a bounded-working-set conversion path.

## Synthetic result

The deterministic small-shape regression currently passes with both stores using direct
I/O. Typical output:

```
dense_layer_max=1.4901161e-08
router_ids=OK
max_weight_abs=5.9604645e-08
final_max=2.9802322e-08
final_rms=1.3762389e-08
trunk_direct_io=yes
expert_direct_io=yes
```

ASan + UBSan also pass this path.

## Next after the real two-layer pass

Generalize `kvl_stack_probe` into a decoder loop over arbitrary layer records, add global
embedding/final-norm/LM-head records, then validate real text logits. Incremental MLA KV
state should follow the full-prefill logits oracle so cache bugs and model-math bugs remain
separable.
