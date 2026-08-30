#!/usr/bin/env python3
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kimi_moe_act_reservoir import (  # noqa: E402
    ENDIAN_MARKER,
    HEADER,
    MAGIC,
    VERSION,
    build_balanced_reservoir,
    iter_trace,
)


def write_trace(path: Path, hidden: int, top_k: int, records) -> None:
    with path.open("wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, hidden, top_k, ENDIAN_MARKER))
        for event, layer, ids, weights, x in records:
            f.write(struct.pack("<Qi", int(event), int(layer)))
            f.write(struct.pack("<" + "i" * top_k, *[int(v) for v in ids]))
            f.write(struct.pack("<" + "f" * top_k, *[float(v) for v in weights]))
            f.write(np.asarray(x, dtype="<f4").tobytes())


class BinaryTraceTests(unittest.TestCase):
    def test_parser_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.act"
            write_trace(
                p,
                hidden=3,
                top_k=2,
                records=[(7, 4, [2, 5], [0.75, 0.25], [1.0, -2.0, 3.5])],
            )
            header, records = iter_trace(p)
            got = list(records)
        self.assertEqual((header.hidden, header.top_k), (3, 2))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].event, 7)
        self.assertEqual(got[0].layer, 4)
        np.testing.assert_array_equal(got[0].ids, np.array([2, 5], dtype=np.int32))
        np.testing.assert_allclose(got[0].weights, [0.75, 0.25], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(got[0].x, [1.0, -2.0, 3.5], rtol=0.0, atol=0.0)

    def test_balances_text_and_media_per_expert(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            text = td / "text.act"
            media = td / "media.act"
            text_records = []
            for i in range(10):
                text_records.append((i + 1, 6, [30, 20], [0.8, 0.2], [float(i), 1.0]))
            media_records = []
            for i in range(3):
                media_records.append((100 + i, 6, [30, 47], [0.6, 0.4], [-float(i), 2.0]))
            write_trace(text, hidden=2, top_k=2, records=text_records)
            write_trace(media, hidden=2, top_k=2, records=media_records)
            out = td / "out"
            manifest = build_balanced_reservoir(
                [("text", text), ("media", media)],
                out,
                capacity_per_kind=2,
                seed=7,
            )
            with np.load(out / "layer-06-expert-30.npz", allow_pickle=False) as data:
                kinds = np.asarray(data["kind"])
                names = [str(x) for x in data["meta_kind_names"].tolist()]
                x = np.asarray(data["x"])
        self.assertEqual(manifest["expert_count"], 3)
        self.assertEqual(manifest["trace_records"], {"text": 10, "media": 3})
        self.assertEqual(x.shape, (4, 2))
        text_id = names.index("text")
        media_id = names.index("media")
        self.assertEqual(int(np.sum(kinds == text_id)), 2)
        self.assertEqual(int(np.sum(kinds == media_id)), 2)
        row = next(e for e in manifest["experts"] if e["layer"] == 6 and e["expert"] == 30)
        self.assertEqual(row["per_kind"]["text"], {"seen": 10, "kept": 2})
        self.assertEqual(row["per_kind"]["media"], {"seen": 3, "kept": 2})

    def test_bad_magic_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.act"
            p.write_bytes(HEADER.pack(b"BADMAGIC", VERSION, 2, 1, ENDIAN_MARKER))
            with self.assertRaises(ValueError):
                iter_trace(p)

    def test_truncated_record_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trunc.act"
            p.write_bytes(HEADER.pack(MAGIC, VERSION, 2, 1, ENDIAN_MARKER) + struct.pack("<Qi", 1, 2))
            header, records = iter_trace(p)
            self.assertEqual(header.hidden, 2)
            with self.assertRaises(ValueError):
                list(records)

    def test_duplicate_selected_expert_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p = td / "dup.act"
            write_trace(p, hidden=2, top_k=2, records=[(1, 1, [3, 3], [0.5, 0.5], [0.0, 1.0])])
            with self.assertRaises(ValueError):
                build_balanced_reservoir([("text", p)], td / "out", 2, 1)


if __name__ == "__main__":
    unittest.main()
