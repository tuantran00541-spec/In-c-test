#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROUTER = ROOT / "tools" / "make_router_prune_map.py"
MERGER = ROOT / "tools" / "merge_expert_store.py"

THDR = struct.Struct("<8sIIIIQQ")
TREC = struct.Struct("<IIII4IQQQ")
EHDR = struct.Struct("<8sIIIIIIQQ")
EREC = struct.Struct("<IIQQQQQQQQQ")
ALIGN = 4096


def bf16_bytes(a: np.ndarray) -> bytes:
    a = np.asarray(a, dtype=np.float32)
    bits = a.view(np.uint32)
    return (bits >> np.uint32(16)).astype("<u2").tobytes()


def make_trunk(root: pathlib.Path):
    binp, idxp = root / "trunk.bin", root / "trunk.idx"
    rows_by_layer = {
        1: np.array([[3, 0], [2.8, 0.1], [0, 3], [0.1, 2.8]], dtype=np.float32),
        2: np.array([[2, 2], [1.9, 2.1], [-2, 2], [-2.1, 1.9]], dtype=np.float32),
    }
    recs = []
    with binp.open("wb") as out:
        for layer, rows in rows_by_layer.items():
            pad = (-out.tell()) % ALIGN
            out.write(b"\0" * pad)
            off = out.tell()
            payload = bf16_bytes(rows)
            out.write(payload)
            read_bytes = ((len(payload) + ALIGN - 1) // ALIGN) * ALIGN
            out.write(b"\0" * (read_bytes - len(payload)))
            recs.append((layer, 30, 1, 2, rows.shape[0], rows.shape[1], 0, 0,
                         off, read_bytes, len(payload)))
    with idxp.open("wb") as f:
        f.write(THDR.pack(b"KVLTRNK1", 1, ALIGN, len(recs), 0, THDR.size, binp.stat().st_size))
        for r in recs:
            f.write(TREC.pack(*r))
    return binp, idxp


def make_experts(root: pathlib.Path):
    binp, idxp = root / "experts.bin", root / "experts.idx"
    recs = []
    with binp.open("wb") as out:
        for layer in (1, 2):
            for expert in range(4):
                off = out.tell()
                byte = bytes([(layer * 16 + expert) & 0xFF])
                payload = byte * 64
                out.write(payload)
                out.write(b"\0" * (ALIGN - len(payload)))
                recs.append((layer, expert, off, ALIGN, 64, 0, 16, 16, 16, 32, 16))
    with idxp.open("wb") as f:
        f.write(EHDR.pack(b"KVLXPRT1", 1, ALIGN, 3, 4, len(recs), 3, EHDR.size, binp.stat().st_size))
        for r in recs:
            f.write(EREC.pack(*r))
    return binp, idxp, recs


def read_expert_records(path: pathlib.Path):
    raw = path.read_bytes()
    h = EHDR.unpack_from(raw, 0)
    return h, [EREC.unpack_from(raw, h[7] + i * EREC.size) for i in range(h[5])]


def main():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        trunk_bin, trunk_idx = make_trunk(root)
        exp_bin, exp_idx, source_recs = make_experts(root)
        mapping = root / "map.json"
        out_bin, out_idx = root / "merged.bin", root / "merged.idx"

        p = subprocess.run(
            [sys.executable, str(ROUTER), str(trunk_bin), str(trunk_idx), str(mapping), "--keep", "2"],
            text=True, capture_output=True, check=True,
        )
        assert "PRUNE_MAP_PASS" in p.stdout
        obj = json.loads(mapping.read_text())
        assert obj["keep_per_layer"] == 2
        assert set(obj["layers"]) == {"1", "2"}
        for info in obj["layers"].values():
            assert len(info["prototypes"]) == 2
            assert len(set(info["map"].values())) == 2
            for proto in info["prototypes"]:
                assert info["map"][str(proto)] == proto

        p = subprocess.run(
            [sys.executable, str(MERGER), str(exp_bin), str(exp_idx), str(mapping), str(out_bin), str(out_idx)],
            text=True, capture_output=True, check=True,
        )
        assert "EXPERT_MERGE_PASS" in p.stdout
        assert out_bin.stat().st_size == 4 * ALIGN
        assert exp_bin.stat().st_size == 8 * ALIGN

        h, out_recs = read_expert_records(out_idx)
        assert h[5] == 8
        assert h[8] == out_bin.stat().st_size

        source = {(r[0], r[1]): r for r in source_recs}
        out_by_key = {(r[0], r[1]): r for r in out_recs}
        with exp_bin.open("rb") as src, out_bin.open("rb") as merged:
            for layer_s, info in obj["layers"].items():
                layer = int(layer_s)
                for e_s, proto in info["map"].items():
                    e = int(e_s)
                    sr = source[(layer, int(proto))]
                    rr = out_by_key[(layer, e)]
                    src.seek(sr[2]); want = src.read(64)
                    merged.seek(rr[2]); got = merged.read(64)
                    assert got == want

    print("EXPERT_STRUCTURED_PRUNING_TEST_PASS")


if __name__ == "__main__":
    main()
