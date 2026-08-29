#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compact_expert_store", ROOT / "tools" / "compact_expert_store.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    src_bin = td / "experts.bin"
    src_idx = td / "experts.idx"
    dst_bin = td / "experts-pruned.bin"
    dst_idx = td / "experts-pruned.idx"
    mask = td / "mask.txt"

    recs = []
    with src_bin.open("wb") as f:
        for layer, expert, byte in [
            (1, 0, b"A"), (1, 1, b"B"), (2, 0, b"C"), (2, 1, b"D")
        ]:
            start = mod.align_up(f.tell())
            f.write(b"\0" * (start - f.tell()))
            f.write(byte * mod.ALIGN)
            recs.append((
                layer, expert, start, mod.ALIGN, 64,
                0, 16, 16, 16, 32, 32,
            ))
    with src_idx.open("wb") as f:
        f.write(mod.HDR.pack(
            mod.MAGIC, mod.VERSION, mod.ALIGN, 3, 2, len(recs),
            mod.DTYPE_Q8_ROW, mod.HDR.size, src_bin.stat().st_size,
        ))
        for r in recs:
            f.write(mod.REC.pack(*r))

    mask.write_text("# KVL_MOE_MASK_V1\n1 1\n", encoding="utf-8")
    report = mod.compact_store(src_bin, src_idx, mask, dst_bin, dst_idx)
    assert report["source_records"] == 4
    assert report["output_records"] == 3
    assert report["removed_records"] == 1
    out = mod.read_index(dst_idx)
    assert out["ids"] == {(1, 0), (2, 0), (2, 1)}
    assert out["n_layers"] == 3 and out["n_experts"] == 2
    assert dst_bin.stat().st_size == 3 * mod.ALIGN

    expected = {(1, 0): b"A", (2, 0): b"C", (2, 1): b"D"}
    with dst_bin.open("rb") as f:
        for r in out["records"]:
            f.seek(r[2])
            assert f.read(1) == expected[(r[0], r[1])]

print("KIMI_EXPERT_STORE_COMPACT_TEST_PASS")
