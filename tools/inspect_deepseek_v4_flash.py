#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request

DEFAULT_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
HF_BASE = "https://huggingface.co"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "in-c-test-deepseek-bootstrap/1"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def require_equal(obj: dict, key: str, expected) -> None:
    got = obj.get(key)
    if got != expected:
        raise SystemExit(f"metadata mismatch: {key}={got!r}, expected {expected!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the official DeepSeek-V4-Flash-0731 metadata without downloading model shards.")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args()

    model_meta = fetch_json(f"{HF_BASE}/api/models/{args.repo}")
    revision = model_meta.get("sha")
    if not revision:
        raise SystemExit("Hugging Face model metadata did not provide a revision SHA")

    base = f"{HF_BASE}/{args.repo}/resolve/{revision}"
    cfg = fetch_json(f"{base}/config.json")
    inference_cfg = fetch_json(f"{base}/inference/config.json")
    index = fetch_json(f"{base}/model.safetensors.index.json")

    require_equal(cfg, "model_type", "deepseek_v4")
    require_equal(cfg, "hidden_size", 4096)
    require_equal(cfg, "num_hidden_layers", 43)
    require_equal(cfg, "n_routed_experts", 256)
    require_equal(cfg, "num_experts_per_tok", 6)
    require_equal(cfg, "num_hash_layers", 3)
    require_equal(cfg, "expert_dtype", "fp4")
    require_equal(cfg, "scoring_func", "sqrtsoftplus")
    require_equal(inference_cfg, "dtype", "fp8")
    require_equal(inference_cfg, "expert_dtype", "fp4")
    require_equal(inference_cfg, "n_layers", 43)
    require_equal(inference_cfg, "n_routed_experts", 256)
    require_equal(inference_cfg, "n_activated_experts", 6)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("checkpoint index contains no weight_map")
    shards = sorted(set(weight_map.values()))
    if len(shards) < 40:
        raise SystemExit(f"unexpectedly small shard set: {len(shards)}")

    total_size = index.get("metadata", {}).get("total_size")
    total_gib = (int(total_size) / 1024**3) if total_size is not None else None

    expert_names = [name for name in weight_map if ".experts." in name]
    router_names = [name for name in weight_map if "gate" in name or "router" in name]
    indexer_names = [name for name in weight_map if "index" in name.lower()]
    compressor_names = [name for name in weight_map if "compress" in name.lower()]

    print(f"repo={args.repo}")
    print(f"revision={revision}")
    print(f"architecture={cfg.get('architectures')}")
    print(f"layers={cfg['num_hidden_layers']} hidden={cfg['hidden_size']}")
    print(f"experts={cfg['n_routed_experts']} topk={cfg['num_experts_per_tok']} hash_layers={cfg['num_hash_layers']}")
    print(f"trunk_quant={cfg.get('quantization_config', {}).get('quant_method')} expert_dtype={cfg['expert_dtype']}")
    print(f"checkpoint_tensors={len(weight_map)} shards={len(shards)}")
    if total_gib is not None:
        print(f"checkpoint_index_total={total_gib:.3f} GiB")
    print(f"expert_tensor_names={len(expert_names)} router_like_names={len(router_names)}")
    print(f"indexer_like_names={len(indexer_names)} compressor_like_names={len(compressor_names)}")
    print("first_shards=" + ",".join(shards[:3]))
    print("DEEPSEEK_V4_METADATA_PASS")


if __name__ == "__main__":
    main()
