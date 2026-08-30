#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

COMPACT_SPEC = importlib.util.spec_from_file_location(
    "compact_expert_store", ROOT / "tools" / "compact_expert_store.py"
)
compact = importlib.util.module_from_spec(COMPACT_SPEC)
assert COMPACT_SPEC.loader is not None
COMPACT_SPEC.loader.exec_module(compact)
sys.modules["compact_expert_store"] = compact

VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_compact_expert_store", ROOT / "tools" / "verify_compact_expert_store.py"
)
verify = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
VERIFY_SPEC.loader.exec_module(verify)

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
            start = compact.align_up(f.tell())
            f.write(b"\0" * (start - f.tell()))
            f.write(byte * compact.ALIGN)
            recs.append((
                layer, expert, start, compact.ALIGN, 64,
                0, 16, 16, 16, 32, 32,
            ))
    with src_idx.open("wb") as f:
        f.write(compact.HDR.pack(
            compact.MAGIC, compact.VERSION, compact.ALIGN, 3, 2, len(recs),
            compact.DTYPE_Q8_ROW, compact.HDR.size, src_bin.stat().st_size,
        ))
        for r in recs:
            f.write(compact.REC.pack(*r))

    mask.write_text("# KVL_MOE_MASK_V1\n1 1\n", encoding="utf-8")
    compact.compact_store(src_bin, src_idx, mask, dst_bin, dst_idx)

    report = verify.verify_compaction(src_bin, src_idx, dst_bin, dst_idx, mask)
    assert report["byte_exact"] is True
    assert report["source_records"] == 4
    assert report["sparse_records"] == 3
    assert report["disabled_records"] == 1
    assert report["records_verified"] == 3
    assert report["bytes_verified"] == 3 * compact.ALIGN
    assert report["byte_reduction"] == compact.ALIGN

    # Corrupt one kept sparse payload byte and require an actionable mismatch.
    with dst_bin.open("r+b") as f:
        f.seek(compact.ALIGN + 17)
        old = f.read(1)
        f.seek(compact.ALIGN + 17)
        f.write(bytes([old[0] ^ 0xFF]))
    try:
        verify.verify_compaction(src_bin, src_idx, dst_bin, dst_idx, mask)
    except ValueError as exc:
        assert "record payload mismatch" in str(exc)
    else:
        raise AssertionError("corrupted sparse payload was not detected")

print("KIMI_EXPERT_STORE_VERIFY_TEST_PASS")
