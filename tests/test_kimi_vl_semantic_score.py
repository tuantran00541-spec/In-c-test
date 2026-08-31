#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kimi_vl_semantic_score import score_semantic_answer


def main() -> None:
    strict = score_semantic_answer("Counts are 5, 4, 3.\nFINAL=37", marker="FINAL", answer_kind="integer", expected="37")
    assert strict["semantic_correct"] and strict["format_ok"] and strict["source"] == "marker"

    flexible = score_semantic_answer("So the result is 37.", marker="FINAL", answer_kind="integer", expected="37")
    assert flexible["semantic_correct"] and not flexible["format_ok"] and flexible["source"] == "tail_fallback"

    colon = score_semantic_answer("Reasoning...\nFINAL: 37", marker="FINAL", answer_kind="integer", expected="37")
    assert colon["semantic_correct"] and not colon["format_ok"] and colon["source"] == "marker"

    wrong_marker = score_semantic_answer("I considered 37, but corrected it. FINAL=36", marker="FINAL", answer_kind="integer", expected="37")
    assert not wrong_marker["semantic_correct"] and wrong_marker["extracted"] == "36"

    ambiguous = score_semantic_answer("FINAL=37\nFINAL=36", marker="FINAL", answer_kind="integer", expected="37")
    assert ambiguous["ambiguous"] and not ambiguous["semantic_correct"]

    word = score_semantic_answer("THIRD: Green", marker="THIRD", answer_kind="word", expected="green")
    assert word["semantic_correct"] and word["extracted"] == "green"

    body_only = score_semantic_answer("There are 37 candidates, but the answer is unclear.", marker="FINAL", answer_kind="integer", expected="37")
    assert not body_only["semantic_correct"]

    print("KIMI_VL_SEMANTIC_SCORE_UNIT_PASS")


if __name__ == "__main__":
    main()
