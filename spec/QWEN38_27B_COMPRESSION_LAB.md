# Qwen3.8-27B dense compression lab

Research branch: `research/qwen38-27b-compression-lab`

Pinned upstream model:
- repo: `Qwen/Qwen3.8-27B`
- revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- license: Apache-2.0

Pinned architecture facts used by the lab:
- dense native multimodal model
- 64 text layers
- hidden size 5120
- SwiGLU intermediate size 17408
- 48 Gated DeltaNet linear-attention layers + 16 full-attention layers
- vision depth 27, vision hidden 1152
- MTP draft head present
- official BF16 repository is about 55.6 GB on the Hub

## Why this lane exists

This model is a useful control against Kimi-VL MoE compression. Every text token traverses every dense MLP, so quantization/pruning failures cannot be hidden by expert rerouting. The first target is therefore quantization sensitivity, not aggressive structured pruning.

The dense SwiGLU MLPs alone contain exactly 17,112,760,320 weights (~31.875 GiB in BF16). This makes MLP quantization the largest single low-risk storage target.

## Phase D0: metadata and synthetic validation

No model shards are downloaded. Validate the pinned architecture/tensor naming and unit-test projection math.

## Phase D1: BF16 layer sensitivity

After the Kimi Q0 heavy job is no longer consuming CI resources:
1. download only the shard(s) needed for selected layers;
2. collect or generate calibration activations for those layers;
3. simulate RTN Q8/Q6/Q5/Q4 and record output reconstruction error;
4. add AWQ-style activation scaling and GPTQ-style error compensation only after the RTN baseline is measured;
5. compare linear-attention vs full-attention-neighbor MLP sensitivity;
6. delete source shards after measurement.

Initial pilot layers should span early/middle/late depth and both layer classes, e.g. 0, 3, 16, 31, 47, 63. This is a proposal until a real workflow is run.

## Phase D2: mixed-bit dense plan

Use measured per-layer/per-matrix sensitivity rather than a uniform bit width. Candidate shape:
- highly sensitive matrix/layer: Q8/Q6
- ordinary MLP matrices: Q5/Q4
- unusually robust matrices: Q3 only if real regression gates allow it
- attention/recurrent-state/vision/MTP components remain untouched until MLP-only candidates are understood

Every storage number before a packed artifact is explicitly a projection. Every quality claim must come from a deterministic full-model regression run.

## Important contrast with MoE pruning

Dense pruning is substantially riskier than MoE expert removal: there is no router redundancy to reroute around a removed MLP. Do not transfer the `mm56` pruning logic to this model. Start with quantization and, if pruning is explored later, prefer structured channel/block sparsity with reconstruction calibration rather than deleting whole dense MLPs.
