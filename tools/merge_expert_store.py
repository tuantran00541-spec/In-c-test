#!/usr/bin/env python3
"""Rewrite a KVL expert store so clusters of logical experts share one payload.

The output index still contains every original (layer, expert) record. Aliased
records point to the same copied payload, so existing C runtimes need no format
or router changes for this first structured-pruning experiment.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
from collections import defaultdict

MAGIC = b"KVLXPRT1"
VERSION = 1
ALIGN = 4096
HDR = struct.Struct("<8sIIIIIIQQ")
REC = struct.Struct("<IIQQQQQQQQQ")


def read_index(path: pathlib.Path):
    raw = path.read_bytes()
    if len(raw) < HDR.size:
        raise SystemExit("bad experts.idx: short header")
    h = HDR.unpack_from(raw, 0)
    magic, version, align, _n_layers, _n_experts, n_records, _dtype, records_offset, _data_bytes = h
    if magic != MAGIC or version != VERSION or align != ALIGN:
        raise SystemExit("bad experts.idx header")
    if len(raw) < records_offset + n_records * REC.size:
        raise SystemExit("bad experts.idx: truncated records")
    recs = [REC.unpack_from(raw, records_offset + i * REC.size) for i in range(n_records)]
    return h, recs


def load_mapping(path: pathlib.Path):
    obj = json.loads(path.read_text())
    if obj.get("version") != 1 or "layers" not in obj:
        raise SystemExit("unsupported pruning manifest")
    mapping = {}
    for layer_s, info in obj["layers"].items():
        mapping[int(layer_s)] = {int(k): int(v) for k, v in info.get("map", {}).items()}
    return obj, mapping


def align_file(out):
    pad = (-out.tell()) % ALIGN
    if pad:
        out.write(b"\0" * pad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_bin", type=pathlib.Path)
    ap.add_argument("src_idx", type=pathlib.Path)
    ap.add_argument("mapping_json", type=pathlib.Path)
    ap.add_argument("out_bin", type=pathlib.Path)
    ap.add_argument("out_idx", type=pathlib.Path)
    args = ap.parse_args()

    h, recs = read_index(args.src_idx)
    magic, version, align, n_layers, n_experts, n_records, dtype, _records_offset, data_bytes = h
    if args.src_bin.stat().st_size < data_bytes:
        raise SystemExit("experts.bin shorter than index declares")

    _manifest, layer_map = load_mapping(args.mapping_json)
    by_key = {(int(r[0]), int(r[1])): r for r in recs}
    if len(by_key) != len(recs):
        raise SystemExit("duplicate expert records in source index")

    prototype_for = {}
    for r in recs:
        layer, expert = int(r[0]), int(r[1])
        prototype = layer_map.get(layer, {}).get(expert, expert)
        if (layer, prototype) not in by_key:
            raise SystemExit(f"layer {layer} expert {expert}: prototype {prototype} not present")
        prototype_for[(layer, expert)] = prototype

    referenced = sorted({(layer, proto) for (layer, _expert), proto in prototype_for.items()})
    args.out_bin.parent.mkdir(parents=True, exist_ok=True)
    args.out_idx.parent.mkdir(parents=True, exist_ok=True)

    new_location = {}
    with args.src_bin.open("rb") as src, args.out_bin.open("wb") as out:
        for key in referenced:
            src_rec = by_key[key]
            file_offset, read_bytes = int(src_rec[2]), int(src_rec[3])
            if file_offset % ALIGN or read_bytes % ALIGN:
                raise SystemExit(f"{key}: source direct-I/O span is not aligned")
            align_file(out)
            new_off = out.tell()
            src.seek(file_offset)
            blob = src.read(read_bytes)
            if len(blob) != read_bytes:
                raise SystemExit(f"{key}: short expert payload read")
            out.write(blob)
            new_location[key] = new_off
        align_file(out)
        out_bytes = out.tell()

    out_records = []
    for r in recs:
        layer, expert = int(r[0]), int(r[1])
        proto = prototype_for[(layer, expert)]
        pr = by_key[(layer, proto)]
        out_records.append((
            layer, expert, int(new_location[(layer, proto)]), int(pr[3]), int(pr[4]),
            int(pr[5]), int(pr[6]), int(pr[7]), int(pr[8]), int(pr[9]), int(pr[10])
        ))

    with args.out_idx.open("wb") as f:
        f.write(HDR.pack(magic, version, align, n_layers, n_experts, n_records,
                         dtype, HDR.size, out_bytes))
        for r in out_records:
            f.write(REC.pack(*r))

    src_bytes = args.src_bin.stat().st_size
    ratio = out_bytes / src_bytes if src_bytes else 0.0
    by_layer = defaultdict(set)
    for layer, proto in referenced:
        by_layer[layer].add(proto)
    counts = ",".join(f"L{k}:{len(v)}" for k, v in sorted(by_layer.items()))
    print(
        f"EXPERT_MERGE_PASS logical_records={n_records} unique_payloads={len(referenced)} "
        f"bytes={out_bytes} source_bytes={src_bytes} ratio={ratio:.6f} {counts}"
    )
    print("NOTE: weight sharing / expert merging; quality must be measured on held-out prompts.")


if __name__ == "__main__":
    main()
