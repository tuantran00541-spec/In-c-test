# V3 — one complete Kimi-VL decoder layer

V3 extends the real-weight V2 MoE path into a complete 4-token decoder-layer forward:

```text
input
 -> RMSNorm
 -> MLA (Q + compressed KV + KV RMSNorm + RoPE + causal softmax + O projection)
 -> residual
 -> RMSNorm
 -> streamed routed MoE + resident shared expert
 -> residual
```

## Numerical contract

V3 still uses BF16 checkpoint storage with FP32 activations/arithmetic. The C kernels expand
BF16 values as they are consumed; they do not materialize full FP32 copies of routed expert
matrices. Native BF16 activation rounding and quantized experts are later milestones.

The real configuration exercised by the probe is:

- hidden = 2048
- heads = 16
- Q/K NoPE = 128 per head
- Q/K RoPE = 64 per head
- V = 128 per head
- compressed KV rank = 512
- RoPE theta = 800000
- routed experts = 64, top-6
- expert intermediate = 1408
- shared experts = 2 (combined intermediate 2816)
- sequence length = 4, causal prefill

The 4-token sequence is deliberate: a one-token attention test would make softmax degenerate
to 1.0 and would barely validate RoPE or causal attention.

## RoPE detail

Kimi-VL's reference code reshapes the raw 64-D RoPE projection to `[32,2]`, transposes the
last two dimensions, then flattens before `rotate_half`. V3 computes the equivalent mapping
from raw `[e0,o0,e1,o1,...]` to `[all evens, all odds]` and applies the complex rotation.

## Correctness checkpoints

`kvl_layer_probe` compares all of these against the Python oracle:

1. first RMSNorm output;
2. MLA output;
3. first residual output;
4. post-attention RMSNorm output;
5. top-6 expert ids and routing weights for every token;
6. final decoder-layer output.

Synthetic V3 is run with a deliberately over-capacity expert access pattern and passes under
ASan/UBSan. The real-weight workflow uses a 256 MiB expert cache against a 1.031 GiB routed
expert pack, forcing eviction instead of allowing the whole layer's expert set to remain
resident.
