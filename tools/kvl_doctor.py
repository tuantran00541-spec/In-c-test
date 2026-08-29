#!/usr/bin/env python3
"""Fast structural preflight for a packed Kimi-VL runtime.

The doctor intentionally does not mmap/load model tensors. It checks the packed files, index
magic, pinned source revision (when recorded), and native image/text executables so users can
catch incomplete downloads/builds before starting a slow CPU inference run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

PINNED_REVISION = "398eede0903cd983a2bfa0cc634e9ac1d843f375"


def binary_path(build_dir: Path, name: str) -> Path:
    if os.name == "nt":
        release = build_dir / "Release" / f"{name}.exe"
        flat = build_dir / f"{name}.exe"
        return release if release.is_file() else flat
    return build_dir / name


def read_magic(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read(8)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check a packed Kimi-VL Q8 runtime before inference")
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--build-dir", type=Path, default=Path("build"))
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="do not fail if SOURCE_REVISION.txt records a different revision")
    args = ap.parse_args()

    model = args.model_dir.resolve()
    required = (
        "trunk.bin", "trunk.idx", "experts.bin", "experts.idx",
        "vision.bin", "vision.idx", "tiktoken.model", "tokenizer_config.json",
        "preprocessor_config.json", "config.json",
    )
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        print("FAIL: missing runtime files:", ", ".join(missing))
        return 1

    expected_magic = {
        "trunk.idx": b"KVLTRNK1",
        "vision.idx": b"KVLTRNK1",
        "experts.idx": b"KVLXPRT1",
    }
    for name, expected in expected_magic.items():
        got = read_magic(model / name)
        if got != expected:
            print(f"FAIL: {name} magic={got!r}, expected={expected!r}")
            return 1

    revision_file = model / "SOURCE_REVISION.txt"
    if revision_file.is_file():
        revision = revision_file.read_text(encoding="ascii", errors="strict").strip()
        if revision != PINNED_REVISION and not args.allow_unpinned:
            print(f"FAIL: source revision {revision} != validated {PINNED_REVISION}")
            print("Use --allow-unpinned only if you intentionally packed another checkpoint revision.")
            return 1
    else:
        revision = "unknown (SOURCE_REVISION.txt missing; older pack path?)"

    bins = [binary_path(args.build_dir, "kvl_vision"), binary_path(args.build_dir, "kvl_generate_vl")]
    missing_bins = [str(p) for p in bins if not p.is_file()]
    if missing_bins:
        print("FAIL: native binaries not found:", ", ".join(missing_bins))
        print("Build with: cmake -S . -B build -DKVL_USE_AVX2=ON && cmake --build build --config Release")
        return 1

    weight_names = ("trunk.bin", "experts.bin", "vision.bin")
    print("Kimi-VL low-RAM runtime doctor")
    print(f"  model_dir: {model}")
    print(f"  source_revision: {revision}")
    total = 0
    for name in weight_names:
        size = (model / name).stat().st_size
        total += size
        print(f"  {name}: {size / 1024**3:.3f} GiB")
    print(f"  packed_weight_total: {total / 1024**3:.3f} GiB")
    for p in bins:
        print(f"  binary: {p}")
    print("PASS: runtime structure and native binaries look ready for inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
