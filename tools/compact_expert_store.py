#!/usr/bin/env python3
"""Physically compact a KVL Q8 expert store using a logical KVL_MOE_MASK.

The output preserves logical layer/expert ids and copies every kept expert record
byte-for-byte into a new 4096-aligned experts.bin. Only file offsets and the
record count change. n_layers/n_experts remain unchanged, so router ids need no
remapping.

A canonical sibling .mask sidecar is emitted next to the output index. The native
runtime requires this sidecar for sparse Q8 stores, verifies that it exactly
matches the physically absent routed expert ids, and auto-loads it when
KVL_MOE_MASK is not explicitly set.

This tool is intentionally Q8-store-only for the current Kimi pruning research
lane. It never mutates the source store.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct

ALIGN = 4096
MAGIC = b"KVLXPRT1"
VERSION = 1
DTYPE_Q8_ROW = 3
HDR = struct.Struct("<8sIIIIIIQQ")
REC = struct.Struct("<IIQQQQQQQQQ")
CHUNK = 1024 * 1024


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def mask_sidecar_path(idx_path: pathlib.Path) -> pathlib.Path:
    if idx_path.suffix == ".idx":
        return idx_path.with_suffix(".mask")
    return pathlib.Path(str(idx_path) + ".mask")


def read_mask(path: pathlib.Path) -> set[tuple[int, int]]:
    disabled: set[tuple[int, int]] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected layer expert")
        layer, expert = map(int, parts)
        if layer < 0 or expert < 0:
            raise ValueError(f"{path}:{lineno}: negative layer/expert")
        key = (layer, expert)
        if key in disabled:
            raise ValueError(f"{path}:{lineno}: duplicate mask entry {key}")
        disabled.add(key)
    if not disabled:
        raise ValueError(f"{path}: no disabled experts")
    return disabled


def write_mask_sidecar(path: pathlib.Path, disabled: set[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# KVL_MOE_MASK_V1", "# Bound to the physically absent routed expert records."]
    lines.extend(f"{layer} {expert}" for layer, expert in sorted(disabled))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_index(path: pathlib.Path):
    raw = path.read_bytes()
    if len(raw) < HDR.size:
        raise ValueError(f"{path}: truncated header")
    h = HDR.unpack_from(raw, 0)
    magic, version, align, n_layers, n_experts, n_records, dtype, records_offset, data_bytes = h
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"{path}: unsupported expert index")
    if align != ALIGN or dtype != DTYPE_Q8_ROW or records_offset != HDR.size:
        raise ValueError(f"{path}: incompatible Q8 store header")
    expected = records_offset + n_records * REC.size
    if len(raw) != expected:
        raise ValueError(f"{path}: size={len(raw)} expected={expected}")
    records = []
    pos = records_offset
    seen = set()
    for i in range(n_records):
        r = REC.unpack_from(raw, pos)
        pos += REC.size
        layer, expert = r[:2]
        if layer >= n_layers or expert >= n_experts:
            raise ValueError(f"{path}: record {i} id out of range {(layer, expert)}")
        if (layer, expert) in seen:
            raise ValueError(f"{path}: duplicate record {(layer, expert)}")
        seen.add((layer, expert))
        if r[3] <= 0 or r[3] % ALIGN:
            raise ValueError(f"{path}: record {(layer, expert)} read_bytes not aligned")
        if r[2] % ALIGN:
            raise ValueError(f"{path}: record {(layer, expert)} file_offset not aligned")
        records.append(r)
    return {
        "header": h,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "n_records": n_records,
        "data_bytes": data_bytes,
        "records": records,
        "ids": seen,
    }


def copy_exact(src, dst, offset: int, n: int) -> None:
    src.seek(offset)
    remain = n
    while remain:
        block = src.read(min(CHUNK, remain))
        if not block:
            raise IOError(f"short source read at offset={offset} remain={remain}")
        dst.write(block)
        remain -= len(block)


def compact_store(src_bin: pathlib.Path, src_idx: pathlib.Path, mask: pathlib.Path,
                  dst_bin: pathlib.Path, dst_idx: pathlib.Path,
                  allow_missing_disabled: bool = False) -> dict:
    if dst_bin.resolve() == src_bin.resolve() or dst_idx.resolve() == src_idx.resolve():
        raise ValueError("destination must not overwrite source store")
    meta = read_index(src_idx)
    disabled = read_mask(mask)
    out_of_range = sorted(
        (l, e) for l, e in disabled
        if l >= meta["n_layers"] or e >= meta["n_experts"]
    )
    if out_of_range:
        raise ValueError(f"mask ids out of range, first={out_of_range[:8]}")
    missing = sorted(disabled - meta["ids"])
    if missing and not allow_missing_disabled:
        raise ValueError(f"mask entries missing from source store, first={missing[:8]}")

    actual_src_size = src_bin.stat().st_size
    if meta["data_bytes"] != actual_src_size:
        raise ValueError(
            f"source data size mismatch index={meta['data_bytes']} actual={actual_src_size}"
        )

    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    dst_idx.parent.mkdir(parents=True, exist_ok=True)
    kept_records = []
    removed_records = 0
    removed_read_bytes = 0

    with src_bin.open("rb") as src, dst_bin.open("wb") as dst:
        for r in meta["records"]:
            layer, expert = r[:2]
            if (layer, expert) in disabled:
                removed_records += 1
                removed_read_bytes += int(r[3])
                continue
            start = align_up(dst.tell())
            if start != dst.tell():
                dst.write(b"\0" * (start - dst.tell()))
            copy_exact(src, dst, int(r[2]), int(r[3]))
            nr = list(r)
            nr[2] = start
            kept_records.append(tuple(nr))

    new_size = dst_bin.stat().st_size
    header = (
        MAGIC, VERSION, ALIGN, meta["n_layers"], meta["n_experts"],
        len(kept_records), DTYPE_Q8_ROW, HDR.size, new_size,
    )
    with dst_idx.open("wb") as f:
        f.write(HDR.pack(*header))
        for r in kept_records:
            f.write(REC.pack(*r))

    sidecar = mask_sidecar_path(dst_idx)
    write_mask_sidecar(sidecar, disabled)

    report = {
        "source_records": meta["n_records"],
        "output_records": len(kept_records),
        "disabled_mask_entries": len(disabled),
        "disabled_missing_from_source": len(missing),
        "removed_records": removed_records,
        "source_bytes": actual_src_size,
        "output_bytes": new_size,
        "removed_read_bytes": removed_read_bytes,
        "byte_reduction": actual_src_size - new_size,
        "byte_reduction_fraction": (
            (actual_src_size - new_size) / actual_src_size if actual_src_size else 0.0
        ),
        "n_layers": meta["n_layers"],
        "n_experts": meta["n_experts"],
        "dtype": DTYPE_Q8_ROW,
        "mask_sidecar": str(sidecar),
        "mask_sidecar_entries": len(disabled),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_bin", type=pathlib.Path)
    ap.add_argument("source_idx", type=pathlib.Path)
    ap.add_argument("mask", type=pathlib.Path)
    ap.add_argument("output_bin", type=pathlib.Path)
    ap.add_argument("output_idx", type=pathlib.Path)
    ap.add_argument("--allow-missing-disabled", action="store_true")
    ap.add_argument("--report", type=pathlib.Path)
    args = ap.parse_args()

    report = compact_store(
        args.source_bin, args.source_idx, args.mask,
        args.output_bin, args.output_idx,
        allow_missing_disabled=args.allow_missing_disabled,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    print(
        "KIMI_EXPERT_STORE_COMPACT_PASS "
        f"records={report['source_records']}->{report['output_records']} "
        f"bytes={report['source_bytes']}->{report['output_bytes']} "
        f"reduction={report['byte_reduction_fraction']:.6f} "
        f"sidecar={report['mask_sidecar']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
