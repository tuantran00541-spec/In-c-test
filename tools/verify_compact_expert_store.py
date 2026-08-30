#!/usr/bin/env python3
"""Verify a compacted KVL Q8 expert store against its full-store source.

The verifier is deliberately stronger than comparing total file hashes: logical
(layer, expert) IDs must match the source minus the mask, every non-offset record
field must be identical, and every kept aligned record blob must compare
byte-for-byte.  This is an offline storage-equivalence check; it does not make a
model-quality claim and it does not replace a runtime full-store+mask vs
sparse-store+same-mask A/B smoke test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from compact_expert_store import CHUNK, read_index, read_mask


def _records_by_id(meta: dict) -> dict[tuple[int, int], tuple]:
    return {(int(r[0]), int(r[1])): r for r in meta["records"]}


def verify_compaction(
    source_bin: pathlib.Path,
    source_idx: pathlib.Path,
    sparse_bin: pathlib.Path,
    sparse_idx: pathlib.Path,
    mask: pathlib.Path,
) -> dict:
    src = read_index(source_idx)
    dst = read_index(sparse_idx)
    disabled = read_mask(mask)

    if src["n_layers"] != dst["n_layers"] or src["n_experts"] != dst["n_experts"]:
        raise ValueError(
            "logical store shape changed: "
            f"source={src['n_layers']}x{src['n_experts']} "
            f"sparse={dst['n_layers']}x{dst['n_experts']}"
        )
    if source_bin.stat().st_size != src["data_bytes"]:
        raise ValueError("source data size does not match source index")
    if sparse_bin.stat().st_size != dst["data_bytes"]:
        raise ValueError("sparse data size does not match sparse index")

    missing_disabled = sorted(disabled - src["ids"])
    if missing_disabled:
        raise ValueError(f"mask ids missing from source store, first={missing_disabled[:8]}")

    expected_ids = src["ids"] - disabled
    if dst["ids"] != expected_ids:
        missing = sorted(expected_ids - dst["ids"])
        extra = sorted(dst["ids"] - expected_ids)
        raise ValueError(
            f"sparse logical ids mismatch missing={missing[:8]} extra={extra[:8]}"
        )
    if dst["n_records"] != len(expected_ids):
        raise ValueError(
            f"sparse record count={dst['n_records']} expected={len(expected_ids)}"
        )

    src_records = _records_by_id(src)
    dst_records = _records_by_id(dst)
    src_digest = hashlib.sha256()
    dst_digest = hashlib.sha256()
    bytes_verified = 0

    with source_bin.open("rb") as sf, sparse_bin.open("rb") as df:
        for key in sorted(expected_ids):
            sr = src_records[key]
            dr = dst_records[key]
            # file_offset (field 2) is intentionally rewritten by compaction.
            if sr[:2] != dr[:2] or sr[3:] != dr[3:]:
                raise ValueError(f"record metadata mismatch for {key}: source={sr} sparse={dr}")

            n = int(sr[3])  # aligned read_bytes, including padding copied by compactor
            sf.seek(int(sr[2]))
            df.seek(int(dr[2]))
            remain = n
            key_tag = f"{key[0]}:{key[1]}:{n}\n".encode("ascii")
            src_digest.update(key_tag)
            dst_digest.update(key_tag)
            pos = 0
            while remain:
                take = min(CHUNK, remain)
                a = sf.read(take)
                b = df.read(take)
                if len(a) != take or len(b) != take:
                    raise IOError(f"short record read key={key} pos={pos} expected={take}")
                if a != b:
                    # Find the first mismatching byte inside this chunk for actionable evidence.
                    delta = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
                    raise ValueError(
                        f"record payload mismatch key={key} byte={pos + delta} "
                        f"source=0x{a[delta]:02x} sparse=0x{b[delta]:02x}"
                    )
                src_digest.update(a)
                dst_digest.update(b)
                remain -= take
                pos += take
                bytes_verified += take

    source_sha = src_digest.hexdigest()
    sparse_sha = dst_digest.hexdigest()
    if source_sha != sparse_sha:
        raise AssertionError("kept-record digests differ despite byte comparison")

    report = {
        "source_records": src["n_records"],
        "sparse_records": dst["n_records"],
        "disabled_records": len(disabled),
        "records_verified": len(expected_ids),
        "bytes_verified": bytes_verified,
        "source_bytes": source_bin.stat().st_size,
        "sparse_bytes": sparse_bin.stat().st_size,
        "byte_reduction": source_bin.stat().st_size - sparse_bin.stat().st_size,
        "byte_reduction_fraction": (
            (source_bin.stat().st_size - sparse_bin.stat().st_size) / source_bin.stat().st_size
            if source_bin.stat().st_size else 0.0
        ),
        "kept_records_sha256": source_sha,
        "byte_exact": True,
        "n_layers": src["n_layers"],
        "n_experts": src["n_experts"],
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_bin", type=pathlib.Path)
    ap.add_argument("source_idx", type=pathlib.Path)
    ap.add_argument("sparse_bin", type=pathlib.Path)
    ap.add_argument("sparse_idx", type=pathlib.Path)
    ap.add_argument("mask", type=pathlib.Path)
    ap.add_argument("--report", type=pathlib.Path)
    args = ap.parse_args()

    report = verify_compaction(
        args.source_bin, args.source_idx, args.sparse_bin, args.sparse_idx, args.mask
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    print(
        "KIMI_EXPERT_STORE_VERIFY_PASS "
        f"records={report['records_verified']} bytes={report['bytes_verified']} "
        f"reduction={report['byte_reduction_fraction']:.6f} "
        f"sha256={report['kept_records_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
