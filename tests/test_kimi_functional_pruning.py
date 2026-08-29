#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "analyze_kimi_moe_trace.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        trace = td / "trace.tsv"
        trace.write_text(
            "# event\tlayer\texpert\trouter_weight\toutput_l2\tsaliency\n"
            "1\t1\t0\t0.5\t2.0\t1.0\n"
            "1\t1\t1\t0.5\t4.0\t2.0\n"
            "2\t1\t0\t0.5\t2.0\t1.0\n"
            "2\t1\t2\t0.5\t6.0\t3.0\n"
            "3\t2\t0\t0.4\t5.0\t2.0\n"
            "3\t2\t1\t0.6\t5.0\t3.0\n",
            encoding="utf-8",
        )
        out = td / "out"
        proc = subprocess.run(
            [sys.executable, str(TOOL), str(trace), "--n-experts", "8",
             "--keep", "6", "5", "--out-dir", str(out), "--coactivation-top", "5"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if "KIMI_MOE_TRACE_ANALYSIS_PASS" not in proc.stdout:
            raise AssertionError(proc.stdout)

        mask6 = [
            line.split("#", 1)[0].strip()
            for line in (out / "mask-keep6.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        # Layer 1: experts 3..7 are unseen, so 3 and 4 are the first two pruned.
        # Layer 2: experts 2..7 are unseen, so 2 and 3 are first.
        expected6 = ["1 3", "1 4", "2 2", "2 3"]
        if mask6 != expected6:
            raise AssertionError((mask6, expected6))

        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        l1 = report["layers"]["1"]
        if l1["events"] != 2:
            raise AssertionError(l1["events"])
        pairs = {(p["a"], p["b"]): p["count"] for p in l1["top_coactivation_pairs"]}
        if pairs.get((0, 1)) != 1 or pairs.get((0, 2)) != 1:
            raise AssertionError(pairs)
        if l1["experts"][0]["saliency_sum"] != 2.0:
            raise AssertionError(l1["experts"][0])

    print("KIMI_FUNCTIONAL_PRUNING_SYNTHETIC_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
