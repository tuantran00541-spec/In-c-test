# Kimi-VL Q8_0 GGUF compatibility evidence

This note records a physical-layout compatibility probe only. It does **not** claim end-to-end logits, model quality, speed, or native Windows no-buffering performance.

## Source

- GGUF repository: `mradermacher/Kimi-VL-A3B-Instruct-GGUF`
- File: `Kimi-VL-A3B-Instruct.Q8_0.gguf`
- Resolved Hugging Face revision: `d645665be0a3028dca3ef3d08dddb51ab23ecf31`
- Downloaded bytes: `16,974,941,728`
- Probe branch HEAD/run input: `fdb14e89206d0ca86d0111fe0e199a8695124a2b`
- GitHub Actions run: `33391055394`
- Job: `99484521036`
- Evidence artifact: `9757535280`
- Artifact digest: `sha256:67029327b7f2405c8a3c1a1e46417a6a85a69f1b00181a96d7f9e755558c3b7c`

## Metadata checks

The downloaded GGUF reports:

- architecture: `deepseek2`
- block count: `27`
- leading dense blocks: `1`
- routed experts: `64`
- experts used per token: `6`
- routed expert FF width: `1408`
- tensor count: `430`

All 26 expected MoE decoder layers (`1..26`) were found.

## Routed expert layout

All routed tensors are `Q8_0` and use split tensors per layer:

- `blk.L.ffn_down_exps.weight`
- `blk.L.ffn_gate_exps.weight`
- `blk.L.ffn_up_exps.weight`

For every MoE layer, the expert index is the outermost physical axis. Each expert slice is contiguous in the GGUF file.

Representative layer 1:

- down logical shape: `[1408, 2048, 64]`
- gate logical shape: `[2048, 1408, 64]`
- up logical shape: `[2048, 1408, 64]`
- each matrix slice per expert: `3,063,808` bytes = `2.921875 MiB`
- all three matrices per expert: `9,191,424` bytes = `8.765625 MiB`

The routed-expert tensors occupy `14.244140625 GiB` inside the GGUF.

For comparison, the project's current custom Q8_ROW expert record is `8,671,232` bytes = `8.26953125 MiB`; GGUF Q8_0 is about 6.0% larger per expert because it stores a scale per 32-value block rather than one FP32 scale per output row.

## Direct-I/O probe

For sampled expert slices from all MoE layers:

- requested alignment: `4096` bytes
- minimum aligned-envelope padding: `4096` bytes
- maximum aligned-envelope padding: `4096` bytes
- mean aligned-envelope padding: `4096` bytes

The expert slice sizes are themselves 4096-byte multiples. The extra page comes from the GGUF tensor base offset not being 4096-aligned. A direct-I/O reader can therefore read one aligned envelope and use the in-buffer sub-offset without repacking the whole model.

## Q8_0 data smoke

A real sample from `blk.1.ffn_down_exps.weight`, expert 0, first two rows was dequantized using the pinned llama.cpp GGUF implementation:

- raw sample bytes: `2992`
- dequantized shape: `[2, 1408]`
- finite: yes
- non-zero: yes
- min: `-0.0678858757`
- max: `0.0922908783`
- mean absolute value: `0.0180987772`

## Verdict

`direct_streamable_from_gguf = true`

`drop_in_current_kvl_q8_row = false`

The Q8_0 GGUF is physically well suited to the project's SSD-resident sparse-MoE design. It should **not** be converted to the existing Q8_ROW representation by default. The preferred integration path is:

1. parse/index the GGUF tensor table;
2. expose `(layer, expert, gate/up/down)` Q8_0 file ranges;
3. read aligned envelopes through the existing direct-I/O backend;
4. add a native Q8_0 GEMV/GEMM path for expert weights;
5. keep the existing hard-budget ExpertCache above that storage layer;
6. only after one-expert numerical validation, run full decoder A/B logits against the existing Q8_ROW/BF16 oracle.
