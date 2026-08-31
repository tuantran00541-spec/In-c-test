#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "simulate_kimi_dynskip.py"
spec = importlib.util.spec_from_file_location("simulate_kimi_dynskip", TOOL)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sample = root / "text-cal"
        sample.mkdir(parents=True)
        ids = [163594, 101, 163601, 102, 163586, 163587, 103, 163601, 201, 163586, 163588, 105, 163601]
        (sample / "prompt.ids").write_text("\n".join(map(str, ids)) + "\n", encoding="ascii")

        # Two MoE layers, top-6 routes per token. Only the user content token at
        # position 8 is eligible for the content layer-2 policy below.
        lines = ["# synthetic"]
        event = 0
        weights = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
        for layer in (1, 2):
            for _pos in range(len(ids)):
                event += 1
                for expert, weight in enumerate(weights):
                    lines.append(f"{event}\t{layer}\t{expert}\t{weight}\t1\t1\t1")
        (sample / "trace.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        manifest = {
            "text_samples": [{
                "id": "demo",
                "domain": "demo",
                "prompt_ids": "/tmp/profile/text-cal/prompt.ids",
                "trace": "/tmp/profile/text-cal/trace.tsv",
            }],
            "vl_samples": [],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "policy.txt").write_text("content 2 0.14 5\n", encoding="utf-8")

        report = mod.simulate(root / "manifest.json", root, root / "policy.txt")
        assert report["samples"] == 1
        assert report["routed"] == len(ids) * 2 * 6
        assert report["skipped"] == 1
        row = next(r for r in report["rows"] if r["family"] == "content" and r["layer"] == 2)
        assert row["events"] == 1 and row["skipped"] == 1
        assert mod.classify_prompt(ids)[8] == mod.CONTENT

    print("KIMI_DYNSKIP_SIMULATOR_UNIT_PASS")


if __name__ == "__main__":
    main()
