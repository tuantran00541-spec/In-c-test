#!/usr/bin/env python3
"""Real-weight layer-level Qwen3.8-27B basic prune+Q6 smoke test.

This is deliberately not a full-model quality claim. It loads one pinned BF16
SwiGLU MLP, prunes a small number of intermediate channels structurally, then
simulates symmetric group-wise Q6 and compares MLP outputs on deterministic
RMS-normalized synthetic activations.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PREFIX = "model.language_model.layers"
HIDDEN = 5120
INTERMEDIATE = 17408


def qdq_rows(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    if w.ndim != 2 or group <= 0 or not (2 <= bits <= 8):
        raise ValueError("bad qdq arguments")
    rows, cols = w.shape
    pad = (-cols) % group
    if pad:
        w2 = torch.nn.functional.pad(w, (0, pad))
    else:
        w2 = w
    blocks = w2.reshape(rows, -1, group)
    qmax = (1 << (bits - 1)) - 1
    amax = blocks.abs().amax(dim=2, keepdim=True)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    q = torch.round(blocks / scale).clamp(-qmax, qmax)
    out = (q * scale).reshape(rows, -1)
    return out[:, :cols]


def metrics(ref: torch.Tensor, got: torch.Tensor) -> dict:
    a = ref.reshape(-1).double(); b = got.reshape(-1).double(); e = b - a
    an = torch.linalg.vector_norm(a); en = torch.linalg.vector_norm(e)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = float(torch.dot(a, b) / denom) if float(denom) else (1.0 if torch.equal(a, b) else 0.0)
    return {
        "mse": float(torch.mean(e * e)),
        "rmse": float(torch.sqrt(torch.mean(e * e))),
        "max_abs": float(e.abs().max()),
        "relative_l2": float(en / an) if float(an) else float(en),
        "cosine": cosine,
    }


def find_tensor(index: dict, layer: int, part: str) -> tuple[str, str]:
    name = f"{PREFIX}.{layer}.mlp.{part}.weight"
    shard = index["weight_map"].get(name)
    if not shard:
        raise KeyError(name)
    return name, shard


def load_tensor(root: Path, name: str, shard: str) -> torch.Tensor:
    p = root / shard
    with safe_open(str(p), framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name}: expected BF16, got {t.dtype}")
    return t.contiguous()


def channel_scores(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    score = torch.zeros(gate.shape[0], dtype=torch.float64)
    for w in (gate, up):
        for s in range(0, w.shape[0], chunk):
            c = w[s:s+chunk].float()
            score[s:s+c.shape[0]] += (c.double() * c.double()).sum(dim=1)
    for s in range(0, down.shape[0], chunk):
        c = down[s:s+chunk].float().double()
        score += (c * c).sum(dim=0)
    return score


def linear_rows(x: torch.Tensor, w: torch.Tensor, row_idx: torch.Tensor | None = None,
                bits: int | None = None, group: int = 128, chunk: int = 256) -> torch.Tensor:
    nout = w.shape[0] if row_idx is None else row_idx.numel()
    out = torch.empty((x.shape[0], nout), dtype=torch.float32)
    for o in range(0, nout, chunk):
        if row_idx is None:
            wc = w[o:o+chunk].float()
        else:
            idx = row_idx[o:o+chunk]
            wc = w.index_select(0, idx).float()
        if bits is not None:
            wc = qdq_rows(wc, bits, group)
        out[:, o:o+wc.shape[0]] = x @ wc.T
    return out


def linear_down(x: torch.Tensor, down: torch.Tensor, keep: torch.Tensor | None = None,
                bits: int | None = None, group: int = 128, chunk: int = 256) -> torch.Tensor:
    out = torch.empty((x.shape[0], down.shape[0]), dtype=torch.float32)
    for o in range(0, down.shape[0], chunk):
        wc = down[o:o+chunk].float()
        if keep is not None:
            wc = wc.index_select(1, keep)
        if bits is not None:
            wc = qdq_rows(wc, bits, group)
        out[:, o:o+wc.shape[0]] = x @ wc.T
    return out


def mlp(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
        keep: torch.Tensor | None = None, bits: int | None = None, group: int = 128) -> torch.Tensor:
    g = linear_rows(x, gate, keep, bits, group)
    u = linear_rows(x, up, keep, bits, group)
    h = torch.nn.functional.silu(g) * u
    return linear_down(h, down, keep, bits, group)


def projected_qbytes(hidden: int, inter: int, bits: int, group: int) -> dict:
    params = 3 * hidden * inter
    scales = 2 * inter * math.ceil(hidden / group) + hidden * math.ceil(inter / group)
    payload = (params * bits + 7) // 8
    return {"params": params, "payload_bytes": payload, "fp16_scale_bytes": scales * 2,
            "total_bytes": payload + scales * 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prune-channels", type=int, default=896)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.prune_channels <= 0 or args.prune_channels >= INTERMEDIATE:
        raise SystemExit("invalid prune count")
    if (INTERMEDIATE - args.prune_channels) % args.group_size:
        raise SystemExit("kept intermediate must divide group size for this pilot")

    root = args.work_dir; root.mkdir(parents=True, exist_ok=True)
    for fn in ("config.json", "model.safetensors.index.json"):
        hf_hub_download(MODEL_ID, filename=fn, revision=REVISION, local_dir=str(root))
    index = json.loads((root / "model.safetensors.index.json").read_text())
    parts = {p: find_tensor(index, args.layer, p) for p in ("gate_proj", "up_proj", "down_proj")}
    shards = sorted({s for _, s in parts.values()})
    for shard in shards:
        hf_hub_download(MODEL_ID, filename=shard, revision=REVISION, local_dir=str(root))

    t0 = time.monotonic()
    gate = load_tensor(root, *parts["gate_proj"])
    up = load_tensor(root, *parts["up_proj"])
    down = load_tensor(root, *parts["down_proj"])
    if tuple(gate.shape) != (INTERMEDIATE, HIDDEN) or tuple(up.shape) != (INTERMEDIATE, HIDDEN) or tuple(down.shape) != (HIDDEN, INTERMEDIATE):
        raise SystemExit(f"unexpected shapes gate={tuple(gate.shape)} up={tuple(up.shape)} down={tuple(down.shape)}")

    score = channel_scores(gate, up, down)
    keep_n = INTERMEDIATE - args.prune_channels
    keep = torch.topk(score, k=keep_n, largest=True, sorted=False).indices.sort().values

    gen = torch.Generator(device="cpu"); gen.manual_seed(args.seed)
    x = torch.randn((args.samples, HIDDEN), generator=gen, dtype=torch.float32)
    x = x / torch.sqrt(torch.mean(x * x, dim=1, keepdim=True))

    ref = mlp(x, gate, up, down)
    prune = mlp(x, gate, up, down, keep=keep)
    qonly = mlp(x, gate, up, down, bits=args.bits, group=args.group_size)
    combo = mlp(x, gate, up, down, keep=keep, bits=args.bits, group=args.group_size)

    full_params = 3 * HIDDEN * INTERMEDIATE
    kept_params = 3 * HIDDEN * keep_n
    full_bf16 = full_params * 2
    proj = projected_qbytes(HIDDEN, keep_n, args.bits, args.group_size)
    result = {
        "schema": "qwen38-basic-knife-v1",
        "scope": "single real BF16 MLP layer; deterministic synthetic RMS-normalized activations; not full-model quality",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "layer": args.layer,
        "source_shards": shards,
        "hidden": HIDDEN,
        "intermediate_before": INTERMEDIATE,
        "intermediate_after": keep_n,
        "pruned_channels": args.prune_channels,
        "pruned_fraction": args.prune_channels / INTERMEDIATE,
        "full_mlp_params": full_params,
        "kept_mlp_params": kept_params,
        "removed_mlp_params": full_params - kept_params,
        "bits": args.bits,
        "group_size": args.group_size,
        "samples": args.samples,
        "metrics": {
            "prune_only_bf16": metrics(ref, prune),
            "q6_only": metrics(ref, qonly),
            "prune_plus_q6": metrics(ref, combo),
        },
        "storage_projection": {
            "projection_only": True,
            "full_layer_mlp_bf16_bytes": full_bf16,
            "structured_pruned_bf16_bytes": kept_params * 2,
            "pruned_q6_with_fp16_scales_bytes": proj["total_bytes"],
            "reduction_vs_full_bf16": 1.0 - proj["total_bytes"] / full_bf16,
        },
        "wall_seconds": time.monotonic() - t0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"QWEN38_BASIC_KNIFE_DONE layer={args.layer} prune={args.prune_channels}/{INTERMEDIATE} bits={args.bits} rel_l2={result['metrics']['prune_plus_q6']['relative_l2']:.6g} cosine={result['metrics']['prune_plus_q6']['cosine']:.9g}")


if __name__ == "__main__":
    main()
