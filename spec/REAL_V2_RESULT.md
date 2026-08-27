# Real-weight V2 result

Date: 2026-08-27

Model: `moonshotai/Kimi-VL-A3B-Instruct`

Scope: decoder layer 1 MLP / MoE path only. This is not yet a full decoder-layer or full-model inference test.

## Checkpoint subset

`model.safetensors.index.json` resolves all layer-1 MLP tensors to a single checkpoint shard:

- `model-00003-of-00007.safetensors`
- 197 MLP tensors total
- 192 routed-expert tensors = 64 experts x 3 matrices
- 3 shared-expert tensors
- 2 router tensors

The routed experts were repacked to an aligned `experts.bin` of 1.031 GiB plus a 5,168-byte index.

## Oracle routing

For the deterministic BF16-representable hidden vector used by `dump_real_moe_reference.py`:

- top-6 expert ids: `[25, 59, 16, 31, 51, 6]`
- routing weights: `[0.43385458, 0.43481228, 0.37181559, 0.40347558, 0.39909950, 0.40294254]`

Only the selected experts were materialized by the Python oracle.

## C runtime comparison

The real-weight run passed:

```text
router_ids=OK
max_weight_abs=5.9604645e-08
moe_max_abs=2.2351742e-08
moe_rms=6.0536789e-09
direct_io=yes
```

Expert-cache report for this one-token probe:

```text
slots=69
slot=16.50 MiB
arena=1138.50 MiB
requests=6
hits=6
misses=0
prefetch=6/1
reads=6
bytes=99.00 MiB
failures=0
```

The six requests count as cache hits during compute because `getmany()` prefetched all six selected experts before the per-expert `get()` calls. The storage layer physically performed six reads totaling 99 MiB.

## What this proves

The following real checkpoint path is now validated end-to-end for one Kimi-VL MoE layer:

```text
official safetensors shard
  -> tensor index/name resolution
  -> routed-expert repack
  -> aligned/direct-I/O expert cache
  -> real router + top-6 selection
  -> BF16 gate/up/down GEMV
  -> SiLU-GLU
  -> routed weighted sum + shared expert
  -> C output vs PyTorch oracle
```

This validates V2 against actual model weights. It does not yet validate RMSNorm, MLA, residuals, multiple decoder layers, tokenization, logits, or vision. Those belong to V3+.

GitHub Actions run: `33089218187` (`Real Kimi-VL V2 probe`, run #2).
