#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "analyze_kimi_reap_vl.py"
spec = importlib.util.spec_from_file_location("analyze_kimi_reap_vl", TOOL)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


def write_trace(path: Path, rows: list[tuple[int, int, int, float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# event layer expert router_weight output_l2 saliency\n")
        for row in rows:
            f.write("%d %d %d %.9g %.9g %.9g\n" % row)


def disabled(mask_rows):
    return {(int(r["layer"]), int(r["expert"])) for r in mask_rows}


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        text = root / "text.tsv"
        vl = root / "vl.tsv"
        ids = root / "prompt.ids"

        # Expert 6 is a text specialist. Experts 0/1 are ordinary support.
        # Experts 2/3/4/5 are deliberately unseen and should be cut first.
        write_trace(text, [
            (1, 1, 6, 0.9, 100.0, 90.0),
            (2, 1, 6, 0.8, 100.0, 80.0),
            (3, 1, 0, 0.5, 2.0, 1.0),
        ])

        # Two prompt positions: event 1 is MEDIA_PAD, event 2 is normal VL text.
        # Expert 7 is important only on media and must therefore be protected by
        # max-over-modality scoring rather than averaged away.
        ids.write_text(f"{mod.MEDIA_PAD_ID}\n42\n", encoding="ascii")
        write_trace(vl, [
            (1, 1, 7, 0.95, 100.0, 95.0),
            (2, 1, 1, 0.5, 2.0, 1.0),
        ])

        rows = mod.read_trace(vl)
        prompt_ids = mod.read_prompt_ids(ids)
        assert mod.vl_event_kind(rows[0], prompt_ids) == "media"
        assert mod.vl_event_kind(rows[1], prompt_ids) == "vl_text"

        report, masks = mod.build_report(
            [("text", text)],
            [(vl, ids)],
            n_experts=8,
            n_groups=2,
            targets=[2, 4],
            max_disabled_per_layer=4,
            max_disabled_per_group=2,
        )
        assert report["kinds"] == ["media", "text", "vl_text"]

        by_slot = {(r["layer"], r["expert"]): r for r in report["experts"]}
        assert by_slot[(1, 6)]["by_kind"]["text"]["normalized_reap"] > 0.0
        assert by_slot[(1, 6)]["final_score"] > 0.0
        assert by_slot[(1, 7)]["by_kind"]["media"]["normalized_reap"] > 0.0
        assert by_slot[(1, 7)]["final_score"] > 0.0
        assert by_slot[(1, 2)]["final_score"] == 0.0

        m2 = disabled(masks[2])
        m4 = disabled(masks[4])
        assert len(m2) == 2 and len(m4) == 4
        assert m2 < m4, (m2, m4)
        assert (1, 6) not in m4, "text specialist was incorrectly pruned"
        assert (1, 7) not in m4, "media specialist was incorrectly pruned"
        assert m4 == {(1, 2), (1, 3), (1, 4), (1, 5)}, m4

        # Router-group caps must be honored by every snapshot.
        for mask in masks.values():
            counts = {}
            for row in mask:
                key = (int(row["layer"]), int(row["expert"]) // 4)
                counts[key] = counts.get(key, 0) + 1
            assert max(counts.values(), default=0) <= 2

        out = root / "out"
        mod.write_outputs(out, report, masks)
        assert (out / "report.json").is_file()
        assert (out / "ranking.tsv").is_file()
        assert (out / "mask-reap2.txt").is_file()
        assert (out / "mask-reap4.txt").is_file()

    print("KIMI_REAP_VL_UNIT_PASS")


if __name__ == "__main__":
    main()
