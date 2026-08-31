#!/usr/bin/env python3
"""Rescore an existing hard VL intelligence summary without rerunning model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kimi_vl_semantic_score import score_semantic_answer
from run_kimi_dynskip_vl_intelligence import CASES


def rescore(report: dict) -> dict:
    specs = {c["id"]: c for c in CASES}
    rows = []
    for row in report["cases"]:
        spec = specs.get(row["id"])
        if spec is None:
            raise ValueError(f"unknown case id: {row['id']}")
        baseline = score_semantic_answer(
            row["baseline"]["text"], marker=spec["marker"],
            answer_kind=spec["answer_kind"], expected=spec["expected"],
        )
        candidate = score_semantic_answer(
            row["candidate"]["text"], marker=spec["marker"],
            answer_kind=spec["answer_kind"], expected=spec["expected"],
        )
        rows.append({
            "id": row["id"],
            "domain": row["domain"],
            "expected": spec["expected"],
            "baseline": baseline,
            "candidate": candidate,
            "baseline_strict_score": row["baseline"].get("score"),
            "candidate_strict_score": row["candidate"].get("score"),
            "retention_regression": baseline["semantic_correct"] and not candidate["semantic_correct"],
            "candidate_improvement": (not baseline["semantic_correct"]) and candidate["semantic_correct"],
            "token_exact": row["comparison"]["token_exact"],
        })
    base_ok = sum(int(r["baseline"]["semantic_correct"]) for r in rows)
    cand_ok = sum(int(r["candidate"]["semantic_correct"]) for r in rows)
    regressions = [r["id"] for r in rows if r["retention_regression"]]
    improvements = [r["id"] for r in rows if r["candidate_improvement"]]
    return {
        "schema": "kimi-dynskip-vl-intelligence-semantic-rescore-v1",
        "source_schema": report.get("schema"),
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "baseline_semantic_correct_cases": base_ok,
            "candidate_semantic_correct_cases": cand_ok,
            "baseline_semantic_retained_cases": base_ok - len(regressions),
            "semantic_retention_regressions": regressions,
            "semantic_candidate_improvements": improvements,
            "baseline_format_ok_cases": sum(int(r["baseline"]["format_ok"]) for r in rows),
            "candidate_format_ok_cases": sum(int(r["candidate"]["format_ok"]) for r in rows),
            "ambiguous_baseline_cases": [r["id"] for r in rows if r["baseline"]["ambiguous"]],
            "ambiguous_candidate_cases": [r["id"] for r in rows if r["candidate"]["ambiguous"]],
        },
        "claim_boundary": (
            "Semantic fallback is a conservative regression aid, not an external intelligence benchmark. "
            "Ambiguous answers are never guessed as correct."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    result = rescore(report)
    out = args.out or args.summary.with_name("intelligence-semantic-rescore.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    s = result["summary"]
    print(
        f"KIMI_DYNSKIP_INTELLIGENCE_SEMANTIC_RESCORE "
        f"baseline={s['baseline_semantic_correct_cases']}/{s['cases']} "
        f"candidate={s['candidate_semantic_correct_cases']}/{s['cases']} "
        f"retained={s['baseline_semantic_retained_cases']}/{s['baseline_semantic_correct_cases']} "
        f"regressions={s['semantic_retention_regressions']} out={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
