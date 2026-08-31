#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "analyze_kimi_moe_profile.py"
spec = importlib.util.spec_from_file_location("analyze_kimi_moe_profile", TOOL)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


def write_trace(path: Path, rows: list[tuple], *, v2: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        if v2:
            f.write(
                "# event layer expert router_weight output_l2 saliency "
                "output_max_abs\n"
            )
        else:
            f.write("# event layer expert router_weight output_l2 saliency\n")
        for row in rows:
            f.write(" ".join(str(value) for value in row) + "\n")


def by_slot(report: dict, layer: int, expert: int) -> dict:
    return next(
        row for row in report["experts"]
        if row["layer"] == layer and row["expert"] == expert
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        english = root / "english.tsv"
        vl = root / "vision.tsv"
        ids = root / "vision.ids"

        # Legacy six-column traces must remain readable. Expert 6 is the
        # English MAN/MSAN specialist and co-routes once with expert 0.
        write_trace(
            english,
            [
                (1, 1, 6, 0.5, 10.0, 5.0),
                (1, 1, 0, 0.5, 2.0, 1.0),
                (2, 1, 6, 0.25, 4.0, 1.0),
                (2, 1, 1, 0.75, 2.0, 1.5),
            ],
            v2=False,
        )

        # One media token followed by one VL-text token. The v2 max-absolute
        # column makes expert 7 an outlier candidate, but not a proven Super
        # Expert because hidden-state massive-activation layers are unknown.
        ids.write_text(f"{mod.MEDIA_PAD_ID}\n42\n", encoding="ascii")
        write_trace(
            vl,
            [
                (1, 1, 7, 0.9, 100.0, 90.0, 100.0),
                (1, 1, 2, 0.1, 1.0, 0.1, 0.5),
                (2, 1, 3, 0.6, 3.0, 1.8, 2.5),
                (2, 1, 4, 0.4, 2.0, 0.8, 1.5),
            ],
            v2=True,
        )

        legacy_rows = mod.read_trace(english)
        v2_rows = mod.read_trace(vl)
        assert legacy_rows[0].output_max_abs is None
        assert v2_rows[0].output_max_abs == 100.0

        report = mod.build_profile(
            [("english", english)],
            [("vision_description", vl, ids)],
            n_experts=8,
        )
        assert report["schema"] == "kimi-moe-multisignal-profile-v1"
        assert report["trace_formats"] == {"legacy_v1": 1, "v2": 1}
        assert report["domains"] == ["english", "vision_description"]
        assert report["modalities"] == ["media", "text", "vl_text"]

        e6 = by_slot(report, 1, 6)["by_domain"]["english"]["all"]
        assert e6["selected"] == 2
        assert math.isclose(e6["route_frequency"], 1.0)
        assert math.isclose(e6["router_weight_mean_abs"], 0.375)
        assert math.isclose(e6["reap"], 3.0)
        assert math.isclose(e6["man"], 7.0)
        assert math.isclose(e6["msan"], 58.0)
        assert e6["output_max_abs_max"] is None

        e7 = by_slot(report, 1, 7)
        media = e7["by_domain"]["vision_description"]["media"]
        assert media["selected"] == 1
        assert media["route_frequency"] == 1.0
        assert media["man"] == 100.0
        assert media["msan"] == 10000.0
        assert media["output_max_abs_max"] == 100.0
        assert media["output_max_abs_p95"] == 100.0

        candidates = {
            (row["layer"], row["expert"])
            for row in report["outlier_profile"]["super_expert_like_candidates"]
        }
        assert candidates == {(1, 7)}, candidates
        assert report["outlier_profile"]["paper_se_layer_condition_available"] is False
        assert e7["aggregate"]["output_max_abs_global_tail_count"] == 1
        assert e7["aggregate"]["output_max_abs_global_tail_frequency"] == 1.0

        pairs = report["coactivation"]["within_layer_pairs"]
        english_pair = next(
            row for row in pairs
            if row["domain"] == "english"
            and row["layer"] == 1
            and row["expert_a"] == 0
            and row["expert_b"] == 6
        )
        assert english_pair["count"] == 1
        assert math.isclose(english_pair["event_fraction"], 0.5)

        vectors = report["coactivation"]["sample_activation_vectors"]
        assert len(vectors) == 2
        english_vector = next(row for row in vectors if row["domain"] == "english")
        vector_values = {
            (row["layer"], row["expert"]): row["abs_router_weight_sum"]
            for row in english_vector["values"]
        }
        assert math.isclose(vector_values[(1, 6)], 0.75)

        # Coverage protection alternates independent domain rankings, so the
        # two domain specialists are both represented before ordinary slots.
        layer_order = report["coverage"]["round_robin_order"]["1"]
        assert layer_order[:2] == [6, 7], layer_order

        out = root / "out"
        mod.write_outputs(out, report)
        assert (out / "report.json").is_file()
        assert (out / "expert-profile.tsv").is_file()
        assert (out / "domain-profile.tsv").is_file()
        assert (out / "coactivation.tsv").is_file()
        assert (out / "sample-activation-matrix.json").is_file()

    print("KIMI_MOE_PROFILE_UNIT_PASS")


if __name__ == "__main__":
    main()
