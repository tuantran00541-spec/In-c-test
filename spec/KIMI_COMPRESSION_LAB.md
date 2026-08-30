# Kimi-VL compression laboratory

Status: research only.  This lane starts from the pinned official revision
`398eede0903cd983a2bfa0cc634e9ac1d843f375` and must not silently use a
moving Hugging Face revision.

## Goal

Find a measured Pareto frontier for the Kimi-VL routed MoE between:

- quality/regression behavior,
- physical packed bytes,
- CPU decode cost,
- SSD/NVMe I/O,
- and RAM/cache pressure.

The target is *not* "the smallest bit width".  A useful outcome may be a
mixed-precision model where different experts receive different bit widths and
some already-validated low-value expert slots are physically pruned.

## Evidence discipline

1. Quantization simulation is not a native runtime implementation.
2. Projected bytes are not physical packed bytes.
3. Reconstruction error is not end-to-end model quality.
4. A pruning candidate and a quantization candidate passing separately does
   not imply the combined candidate passes.
5. No throughput/speed claim is allowed until the corresponding native packed
   format is benchmarked on real hardware.
6. Low-bit final candidates must be quantized from pinned BF16 source weights,
   not requantized from the existing Q8 expert store.

## Phase Q0: offline sensitivity map

For each routed expert `(layer, expert)` gather a bounded calibration reservoir
of actual expert-input activations.  Preserve provenance so text and vision
samples can be inspected separately.

For each expert, simulate at minimum:

- Q8,
- Q6,
- Q5,
- Q4,

with a simple symmetric RTN group-wise baseline.  Measure:

- gate/up/down weight MSE,
- expert-output MSE/RMSE,
- expert-output cosine,
- relative L2 error,
- max absolute output error,
- projected low-bit payload and scale overhead.

The initial simulator is `tools/kimi_compression_lab.py`.  It dequantizes back
to FP32 for measurement and deliberately does not write a native low-bit
store.

## Phase Q1: calibration-aware quantizers

Compare the RTN baseline against:

- AWQ-style activation-aware scaling,
- GPTQ-style reconstruction/error compensation,
- expert-balanced calibration reservoirs,
- and, if needed, protected outlier/salient channels.

Do not attribute an improvement to GPTQ/AWQ unless the experiment changes only
that component relative to a stated baseline.

## Phase Q2: mixed-bit assignment

Create candidate assignments using measured sensitivity, not routing frequency
alone.  The assignment may include Q8/Q6/Q5/Q4 and the already quality-gated
pruning mask.  Every combined pruning+quantization candidate is a new model
candidate and must rerun the quality gates.

## Phase Q3: native format

Only after Q0-Q2 identify a promising frontier should native C formats/kernels
be added.  Requirements include:

- deterministic indexed expert records,
- random-access-friendly block layout,
- direct-I/O compatibility where possible,
- explicit format/version metadata,
- fail-fast mismatch handling,
- unit tests against the simulator,
- exact physical byte accounting.

## Quality gates

Use cheap gates first and expensive gates last:

1. deterministic sentinel,
2. text next-token distribution/argmax,
3. VL next-token distribution/argmax,
4. text long generation,
5. VL long generation,
6. semantic oracle/reference checks that are not merely candidate-vs-baseline
   self-consistency.

The historical C2 pruning result identified `mm56` as the strongest tested
quality-gated pruning candidate, with `mm58` crossing the tested divergence
boundary.  This compression lane branches from that research state but does
not assume `mm56 + low-bit` is safe until combined gates pass.
