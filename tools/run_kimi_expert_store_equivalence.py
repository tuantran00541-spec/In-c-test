#!/usr/bin/env python3
"""Compare full and compacted Kimi Q8 expert stores under one logical mask.

The full-store run always receives --mask explicitly. By default the sparse-store
run receives the same explicit mask too. With --sparse-auto-mask, KVL_MOE_MASK is
removed from the sparse process environment so the native runtime must discover,
validate and bind the sparse index's sibling .mask sidecar itself.

Generated TOKEN stdout, routed-expert trace and raw logits dump must be
byte-identical. This isolates expert-store layout/binding equivalence from pruning
quality: both sides execute the same logical disabled-expert set.

Use --media with kvl_generate_vl; omit it with kvl_generate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def first_mismatch(a: pathlib.Path, b: pathlib.Path) -> int | None:
    pos = 0
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            aa = fa.read(1024 * 1024)
            bb = fb.read(1024 * 1024)
            if aa == bb:
                if not aa:
                    return None
                pos += len(aa)
                continue
            n = min(len(aa), len(bb))
            for i in range(n):
                if aa[i] != bb[i]:
                    return pos + i
            return pos + n


def run_variant(
    name: str,
    binary: pathlib.Path,
    trunk_bin: pathlib.Path,
    trunk_idx: pathlib.Path,
    experts_bin: pathlib.Path,
    experts_idx: pathlib.Path,
    prompt_ids: pathlib.Path,
    media: pathlib.Path | None,
    mask: pathlib.Path | None,
    work: pathlib.Path,
    cache_mib: int,
    max_new: int,
    temperature: float,
    seed: int,
) -> dict:
    root = work / name
    root.mkdir(parents=True, exist_ok=True)
    stdout = root / "tokens.txt"
    stderr = root / "stderr.txt"
    trace = root / "route.tsv"
    logits = root / "logits.bin"

    cmd = [
        str(binary), str(trunk_bin), str(trunk_idx),
        str(experts_bin), str(experts_idx), str(prompt_ids),
    ]
    if media is not None:
        cmd.append(str(media))
    cmd.extend([
        str(cache_mib * 1024 * 1024), str(max_new), str(temperature), str(seed)
    ])

    env = os.environ.copy()
    if mask is None:
        env.pop("KVL_MOE_MASK", None)
        mask_mode = "sparse-sidecar-auto"
    else:
        env["KVL_MOE_MASK"] = str(mask)
        mask_mode = "explicit"
    env["KVL_MOE_TRACE"] = str(trace)
    env["KVL_LOGITS_DUMP"] = str(logits)
    env["KVL_LOGITS_DUMP_LIMIT"] = "0"

    with stdout.open("wb") as out, stderr.open("wb") as err:
        proc = subprocess.run(cmd, env=env, stdout=out, stderr=err)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} runtime failed rc={proc.returncode}; stderr={stderr}"
        )
    for path in (stdout, trace, logits):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{name}: missing/empty evidence file {path}")
    return {
        "name": name,
        "mask_mode": mask_mode,
        "experts_bin": str(experts_bin),
        "experts_idx": str(experts_idx),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "trace": str(trace),
        "logits": str(logits),
        "stdout_sha256": sha256_file(stdout),
        "trace_sha256": sha256_file(trace),
        "logits_sha256": sha256_file(logits),
    }


def compare_outputs(full: dict, sparse: dict) -> dict:
    checks = {}
    for label, key in (("tokens", "stdout"), ("route", "trace"), ("logits", "logits")):
        a = pathlib.Path(full[key])
        b = pathlib.Path(sparse[key])
        mismatch = first_mismatch(a, b)
        checks[label] = {
            "byte_exact": mismatch is None,
            "first_mismatch_byte": mismatch,
            "full_bytes": a.stat().st_size,
            "sparse_bytes": b.stat().st_size,
            "full_sha256": sha256_file(a),
            "sparse_sha256": sha256_file(b),
        }
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=pathlib.Path, required=True)
    ap.add_argument("--trunk-bin", type=pathlib.Path, required=True)
    ap.add_argument("--trunk-idx", type=pathlib.Path, required=True)
    ap.add_argument("--full-experts-bin", type=pathlib.Path, required=True)
    ap.add_argument("--full-experts-idx", type=pathlib.Path, required=True)
    ap.add_argument("--sparse-experts-bin", type=pathlib.Path, required=True)
    ap.add_argument("--sparse-experts-idx", type=pathlib.Path, required=True)
    ap.add_argument("--prompt-ids", type=pathlib.Path, required=True)
    ap.add_argument("--media", type=pathlib.Path)
    ap.add_argument("--mask", type=pathlib.Path, required=True)
    ap.add_argument(
        "--sparse-auto-mask", action="store_true",
        help="omit KVL_MOE_MASK for sparse runtime and require its bound .mask sidecar",
    )
    ap.add_argument("--work-dir", type=pathlib.Path, required=True)
    ap.add_argument("--cache-mib", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.cache_mib <= 0 or args.max_new <= 0 or args.temperature < 0:
        raise SystemExit("invalid cache/max-new/temperature")
    required_files = [
        args.binary, args.trunk_bin, args.trunk_idx,
        args.full_experts_bin, args.full_experts_idx,
        args.sparse_experts_bin, args.sparse_experts_idx,
        args.prompt_ids, args.mask,
    ]
    if args.media is not None:
        required_files.append(args.media)
    missing = [str(p) for p in required_files if not p.is_file()]
    if missing:
        raise SystemExit("missing required files: " + ", ".join(missing))

    if args.sparse_auto_mask:
        sidecar = (
            args.sparse_experts_idx.with_suffix(".mask")
            if args.sparse_experts_idx.suffix == ".idx"
            else pathlib.Path(str(args.sparse_experts_idx) + ".mask")
        )
        if not sidecar.is_file():
            raise SystemExit(f"missing sparse bound mask sidecar: {sidecar}")

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    common = dict(
        binary=args.binary.resolve(),
        trunk_bin=args.trunk_bin.resolve(), trunk_idx=args.trunk_idx.resolve(),
        prompt_ids=args.prompt_ids.resolve(),
        media=args.media.resolve() if args.media else None,
        work=work, cache_mib=args.cache_mib, max_new=args.max_new,
        temperature=args.temperature, seed=args.seed,
    )
    full = run_variant(
        "full-store-mask", experts_bin=args.full_experts_bin.resolve(),
        experts_idx=args.full_experts_idx.resolve(), mask=args.mask.resolve(), **common,
    )
    sparse = run_variant(
        "sparse-store-auto-mask" if args.sparse_auto_mask else "sparse-store-mask",
        experts_bin=args.sparse_experts_bin.resolve(),
        experts_idx=args.sparse_experts_idx.resolve(),
        mask=None if args.sparse_auto_mask else args.mask.resolve(), **common,
    )
    checks = compare_outputs(full, sparse)
    report = {
        "schema_version": 2,
        "scope": (
            "full explicit-mask vs sparse bound-sidecar runtime equivalence; not a pruning-quality claim"
            if args.sparse_auto_mask else
            "same-mask runtime expert-store layout equivalence; not a pruning-quality claim"
        ),
        "mode": "vl" if args.media else "text",
        "cache_mib": args.cache_mib,
        "max_new": args.max_new,
        "temperature": args.temperature,
        "seed": args.seed,
        "sparse_auto_mask": args.sparse_auto_mask,
        "full": full,
        "sparse": sparse,
        "checks": checks,
        "byte_exact": all(v["byte_exact"] for v in checks.values()),
    }
    out = work / "expert-store-equivalence.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for label, check in checks.items():
        print(
            f"EXPERT_STORE_EQ {label} exact={check['byte_exact']} "
            f"bytes={check['full_bytes']}/{check['sparse_bytes']} "
            f"first_mismatch={check['first_mismatch_byte']}"
        )
    if not report["byte_exact"]:
        print(f"KIMI_EXPERT_STORE_RUNTIME_EQ_REJECT report={out}")
        return 1
    print(
        f"KIMI_EXPERT_STORE_RUNTIME_EQ_PASS mode={report['mode']} "
        f"max_new={args.max_new} sparse_mask={sparse['mask_mode']} report={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
