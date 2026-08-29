#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "analyze_kimi_moe_trace.py"


def mask_rows(path: Path):
    return [
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        trace1 = td / "trace1.tsv"
        trace2 = td / "trace2.tsv"
        trace1.write_text(
            "# event\tlayer\texpert\trouter_weight\toutput_l2\tsaliency\n"
            "1\t1\t0\t0.5\t2.0\t1.0\n"
            "1\t1\t1\t0.5\t4.0\t2.0\n"
            "2\t1\t0\t0.5\t2.0\t1.0\n"
            "2\t1\t2\t0.5\t6.0\t3.0\n"
            "3\t2\t0\t0.4\t5.0\t2.0\n"
            "3\t2\t1\t0.6\t5.0\t3.0\n",
            encoding="utf-8",
        )
        trace2.write_text(
            "# event\tlayer\texpert\trouter_weight\toutput_l2\tsaliency\n"
            "1\t1\t0\t0.5\t3.0\t1.5\n"
            "1\t1\t1\t0.5\t2.0\t1.0\n"
            "2\t1\t0\t0.5\t3.0\t1.5\n"
            "2\t1\t3\t0.5\t2.0\t1.0\n"
            "3\t2\t0\t0.4\t5.0\t2.0\n"
            "3\t2\t2\t0.6\t5.0\t3.0\n",
            encoding="utf-8",
        )
        out = td / "out"
        proc = subprocess.run(
            [sys.executable, str(TOOL), str(trace1), str(trace2),
             "--n-experts", "8", "--keep", "6", "5", "--unseen-cap", "2", "6",
             "--out-dir", str(out), "--coactivation-top", "5"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if "KIMI_MOE_TRACE_ANALYSIS_PASS" not in proc.stdout:
            raise AssertionError(proc.stdout)

        expected6 = ["1 4", "1 5", "2 3", "2 4"]
        if mask_rows(out / "mask-keep6.txt") != expected6:
            raise AssertionError(mask_rows(out / "mask-keep6.txt"))
        if mask_rows(out / "mask-unseen-cap2.txt") != expected6:
            raise AssertionError(mask_rows(out / "mask-unseen-cap2.txt"))

        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        l1 = report["layers"]["1"]
        if l1["events"] != 4:
            raise AssertionError(l1["events"])
        if l1["seen_experts"] != 4 or l1["unseen_experts"] != 4:
            raise AssertionError((l1["seen_experts"], l1["unseen_experts"]))
        pairs = {(p["a"], p["b"]): p["count"] for p in l1["top_coactivation_pairs"]}
        if pairs.get((0, 1)) != 2 or pairs.get((0, 2)) != 1 or pairs.get((0, 3)) != 1:
            raise AssertionError(pairs)
        if l1["experts"][0]["saliency_sum"] != 5.0:
            raise AssertionError(l1["experts"][0])
        if l1["experts"][0]["trace_file_count"] != 2:
            raise AssertionError(l1["experts"][0])
        if l1["experts"][2]["trace_file_count"] != 1:
            raise AssertionError(l1["experts"][2])

        coverage = report["coverage"]
        if coverage["seen_slots"] != 7 or coverage["unseen_slots"] != 9:
            raise AssertionError(coverage)
        if coverage["trace_file_support_histogram"] != {"0": 9, "1": 4, "2": 3}:
            raise AssertionError(coverage["trace_file_support_histogram"])
        if report["masks"]["6"]["disabled_seen_count"] != 0:
            raise AssertionError(report["masks"]["6"])
        if report["unseen_cap_masks"]["6"]["disabled_count"] != 9:
            raise AssertionError(report["unseen_cap_masks"]["6"])
        if report["unseen_cap_masks"]["6"]["disabled_seen_count"] != 0:
            raise AssertionError(report["unseen_cap_masks"]["6"])
        if not (out / "layer-sensitivity.tsv").is_file():
            raise AssertionError("missing layer-sensitivity.tsv")

    print("KIMI_FUNCTIONAL_PRUNING_SYNTHETIC_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
