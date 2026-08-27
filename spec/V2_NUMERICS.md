# V2 numerical path

V2 is the first revision that computes a Kimi-VL MoE block rather than only moving bytes.

## Implemented contract

For one token, one MoE decoder layer:

1. router logits = FP32 `W_router @ x`;
2. sigmoid scores;
3. `noaux_tc` selection using correction bias;
4. mixing weights gathered from the **unbiased** scores, normalized, then multiplied by
   `routed_scaling_factor`;
5. `getmany(top-k)` through the V1 expert cache;
6. each BF16 expert is consumed in-place as `down(SiLU(gate(x)) * up(x))`;
7. routed outputs are weighted and summed in FP32;
8. one resident shared MLP is added (its intermediate width represents the two shared
   experts concatenated, as in the released Kimi-VL implementation).

The scalar BF16 GEMV expands each BF16 element only at the multiply. No expert matrix is
widened into a second FP32 allocation.

## Current precision boundary

V2 deliberately uses **FP32 activations with BF16 weight storage**. It does not yet mimic
native Transformers BF16 rounding after each linear/activation. This is intentional:
first prove tensor layout, routing, streaming, and the mathematical graph; decoder-level
native-dtype equivalence is a later milestone.

## Automated oracle

`tests/test_moe_oracle.py` creates 64 routed experts and top-6 routing (the real Kimi-VL
cardinality) with reduced matrix dimensions. PyTorch computes the oracle; the expert
weights are saved as BF16 safetensors, passed through the real packer, read through the
real direct-I/O cache, and evaluated by the C path.

Acceptance:

- same top-k expert set;
- max gate-weight absolute error < 2e-5;
- max MoE output absolute error < 3e-4;
- no cache read failures;
- ASan/UBSan clean.

Current observed synthetic-oracle error: approximately `1.5e-8` max absolute output.

## Real-checkpoint oracle without loading 32.8 GB

Once the original checkpoint exists locally:

```sh
python tools/pack_experts.py MODEL_DIR PACKED_DIR --layer 1
python tools/dump_real_moe_reference.py MODEL_DIR real-layer1.fixture --layer 1
build/kvl_moe_probe \
  PACKED_DIR/experts.bin PACKED_DIR/experts.idx \
  real-layer1.fixture 1200000000
```

The reference helper opens tensors directly from safetensors. It loads router+bias,
chooses top-6, then reads only those six routed experts plus the shared MLP for the chosen
layer. It never instantiates the full Transformers model.

A real Kimi-VL layer contains 64 BF16 routed experts, so `--layer 1` still creates about
1 GiB of packed expert data, but that is much safer than packing/loading all 26 MoE layers
just to validate the numerical path.
