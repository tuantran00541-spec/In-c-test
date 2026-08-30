#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compact_expert_store", ROOT / "tools" / "compact_expert_store.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)

probe = ROOT / "build" / ("kvl_probe.exe" if os.name == "nt" else "kvl_probe")
if not probe.is_file():
    raise SystemExit(f"missing built probe: {probe}")

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    src_bin = td / "experts.bin"
    src_idx = td / "experts.idx"
    dst_bin = td / "experts-sparse.bin"
    dst_idx = td / "experts-sparse.idx"
    mask = td / "mask.txt"

    recs = []
    with src_bin.open("wb") as f:
        for layer, expert, byte in [(1, 0, b"A"), (1, 1, b"B"), (2, 0, b"C")]:
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
    assert report["output_records"] == 2

    kept = subprocess.run(
        [str(probe), str(dst_bin), str(dst_idx), "1", "0"],
        text=True, capture_output=True,
    )
    assert kept.returncode == 0, (kept.stdout, kept.stderr)
    assert "L1/E0" in kept.stdout and "got=4096" in kept.stdout

    removed = subprocess.run(
        [str(probe), str(dst_bin), str(dst_idx), "1", "1"],
        text=True, capture_output=True,
    )
    assert removed.returncode == 1, (removed.stdout, removed.stderr)
    assert "expert L1/E1 not present" in removed.stderr

    kept2 = subprocess.run(
        [str(probe), str(dst_bin), str(dst_idx), "2", "0"],
        text=True, capture_output=True,
    )
    assert kept2.returncode == 0, (kept2.stdout, kept2.stderr)
    assert "L2/E0" in kept2.stdout and "got=4096" in kept2.stdout

print("KIMI_SPARSE_EXPERT_STORE_RUNTIME_TEST_PASS")
