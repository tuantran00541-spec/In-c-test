#!/usr/bin/env python3
"""Hard multimodal reasoning A/B gate for Kimi-VL dynamic expert skipping.

The suite is deliberately different from simple description/counting smoke tests.
Each synthetic image has an independently known ground-truth answer and requires
some combination of visual parsing, arithmetic, spatial reasoning, planning,
ordering, pattern induction, Vietnamese instruction following, or multi-step
reasoning.

Baseline and candidate are both scored against ground truth. Token-exact equality
is recorded separately: candidate wording may drift without being counted as an
intelligence regression if the final structured answer remains correct.

This remains a small deterministic regression suite, not a general benchmark of
model intelligence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

EOS_ID = 163585
IM_END_ID = 163586

CASES = [
    {
        "id": "hard-weighted-inventory",
        "image": "weighted-inventory.png",
        "domain": "visual_counting_arithmetic",
        "prompt": (
            "Inspect the image carefully. Compute 2 times the number of red circles, "
            "plus 3 times the number of blue squares, plus 5 times the number of green triangles. "
            "Explain the counts and arithmetic briefly, then end with a separate line exactly as FINAL=<integer>."
        ),
        "max_new": 40,
        "answer_kind": "integer",
        "expected": "37",
        "marker": "FINAL",
    },
    {
        "id": "hard-vi-quadrant-arithmetic",
        "image": "vi-quadrants.png",
        "domain": "vietnamese_spatial_arithmetic",
        "prompt": (
            "Quan sát bốn góc được chia bởi hai đường đen. Hãy tính: "
            "(số tam giác đỏ ở góc trên-trái nhân số hình tròn vàng ở góc dưới-phải) "
            "cộng số hình vuông xanh lá ở góc dưới-trái. Giải thích ngắn gọn rồi kết thúc bằng "
            "một dòng riêng đúng dạng KET_QUA=<số nguyên>."
        ),
        "max_new": 40,
        "answer_kind": "integer",
        "expected": "19",
        "marker": "KET_QUA",
    },
    {
        "id": "hard-color-cycle-induction",
        "image": "color-cycle.png",
        "domain": "visual_pattern_induction",
        "prompt": (
            "This 5 by 5 grid follows one color rule: moving one cell right advances "
            "red -> blue -> green -> red, and moving one cell down also advances one step. "
            "Exactly one interior cell is blank. Infer its missing color. Reply only COLOR=red, COLOR=blue, or COLOR=green."
        ),
        "max_new": 12,
        "answer_kind": "word",
        "expected": "green",
        "marker": "COLOR",
    },
    {
        "id": "hard-maze-shortest-path",
        "image": "maze.png",
        "domain": "visual_planning",
        "prompt": (
            "The image is a 6 by 6 cell maze. The red cell is the start, the green cell is the goal, "
            "black cells are blocked, and white cells are open. You may move only one cell up, down, left, or right. "
            "Find the minimum number of moves from start to goal. Briefly explain the route length, then end with "
            "a separate line exactly as FINAL=<integer>."
        ),
        "max_new": 48,
        "answer_kind": "integer",
        "expected": "13",
        "marker": "FINAL",
    },
    {
        "id": "hard-size-ordering",
        "image": "size-ordering.png",
        "domain": "visual_comparison_ordering",
        "prompt": (
            "There are five colored circles of different diameters. Rank them by diameter from largest to smallest "
            "and identify the third-largest circle. Reply only THIRD=<color> using the English color name."
        ),
        "max_new": 12,
        "answer_kind": "word",
        "expected": "green",
        "marker": "THIRD",
    },
    {
        "id": "hard-dot-matrix-rule",
        "image": "dot-matrix.png",
        "domain": "visual_pattern_counting",
        "prompt": (
            "The image is a 3 by 3 matrix of panels containing black dots. In each complete row, "
            "the third panel contains the sum of the dots in the first two panels. The bottom-right panel is intentionally empty. "
            "Infer how many dots should replace it. Explain the row rule briefly, then end with a separate line exactly as FINAL=<integer>."
        ),
        "max_new": 40,
        "answer_kind": "integer",
        "expected": "5",
        "marker": "FINAL",
    },
]

MAZE_N = 6
MAZE_START = (0, 1)
MAZE_GOAL = (0, 4)
MAZE_BLOCKED = {(r, 3) for r in range(5)}

CACHE_RE = re.compile(
    r"kvl_cache: .*?req=(?P<requests>\d+) hit=(?P<hits>\d+) miss=(?P<misses>\d+) "
    r"hit_rate=(?P<hit_rate>[0-9.]+)% evict=(?P<evictions>\d+) "
    r"prefetch=(?P<prefetch_reads>\d+)/(?P<prefetch_batches>\d+) "
    r"reads=(?P<read_ops>\d+) bytes=(?P<bytes_mib>[0-9.]+) MiB"
)


def shortest_path_length(n: int, start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]]) -> int | None:
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), dist = q.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n):
                continue
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, dist + 1))
    return None


def validate_cases() -> None:
    ids = [c["id"] for c in CASES]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate hard-suite case id")
    if len(CASES) < 6:
        raise ValueError("hard suite unexpectedly small")
    for c in CASES:
        if c["answer_kind"] not in {"integer", "word"}:
            raise ValueError(f"{c['id']}: unsupported answer kind")
        if int(c["max_new"]) <= 1:
            raise ValueError(f"{c['id']}: max_new must exceed 1")
        if not c["expected"] or not c["marker"]:
            raise ValueError(f"{c['id']}: missing ground truth")
    if shortest_path_length(MAZE_N, MAZE_START, MAZE_GOAL, MAZE_BLOCKED) != 13:
        raise ValueError("maze ground truth drifted")
    # Inventory: 5*2 + 4*3 + 3*5 = 37. Quadrant: 3*5 + 4 = 19.
    if 5 * 2 + 4 * 3 + 3 * 5 != 37 or 3 * 5 + 4 != 19:
        raise ValueError("arithmetic ground truth drifted")


def run(cmd: list[str], *, env: dict[str, str] | None = None,
        stdout_path: Path | None = None, stderr_path: Path | None = None) -> None:
    stdout = stdout_path.open("w", encoding="utf-8") if stdout_path else None
    stderr = stderr_path.open("w", encoding="utf-8") if stderr_path else None
    try:
        subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, check=True, text=True)
    finally:
        if stdout:
            stdout.close()
        if stderr:
            stderr.close()


def parse_generated(path: Path) -> list[int]:
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("TOKEN "):
            ids.append(int(line.split()[1]))
    return ids


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def parse_cache(stderr_path: Path) -> dict[str, int | float]:
    text = stderr_path.read_text(encoding="utf-8")
    matches = list(CACHE_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"{stderr_path}: expected one kvl_cache line, got {len(matches)}")
    g = matches[0].groupdict()
    return {
        "requests": int(g["requests"]),
        "hits": int(g["hits"]),
        "misses": int(g["misses"]),
        "hit_rate_percent": float(g["hit_rate"]),
        "evictions": int(g["evictions"]),
        "prefetch_reads": int(g["prefetch_reads"]),
        "prefetch_batches": int(g["prefetch_batches"]),
        "read_ops": int(g["read_ops"]),
        "bytes_read_mib": float(g["bytes_mib"]),
    }


def parse_stats(path: Path) -> tuple[list[dict[str, str]], int, int, int]:
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    routed = sum(int(r["routed"]) for r in rows)
    skipped = sum(int(r["skipped"]) for r in rows)
    non_media = sum(int(r["skipped"]) for r in rows if r["family"] != "media")
    return rows, routed, skipped, non_media


def normalize_word(value: str) -> str:
    return re.sub(r"[^a-z]+", "", value.lower())


def score_answer(text: str, case: dict) -> dict:
    marker = re.escape(case["marker"])
    if case["answer_kind"] == "integer":
        m = re.search(rf"(?im)^\s*{marker}\s*=\s*(-?\d+)\s*[.!]?\s*$", text)
        extracted = m.group(1) if m else None
        correct = extracted == case["expected"]
    else:
        m = re.search(rf"(?im)^\s*{marker}\s*=\s*([A-Za-z]+)\s*[.!]?\s*$", text)
        extracted = normalize_word(m.group(1)) if m else None
        correct = extracted == normalize_word(case["expected"])
    return {
        "format_ok": m is not None,
        "extracted": extracted,
        "expected": case["expected"],
        "correct": bool(correct),
    }


def draw_shape(draw, kind: str, box: tuple[int, int, int, int], fill: str) -> None:
    x0, y0, x1, y1 = box
    if kind == "circle":
        draw.ellipse(box, fill=fill, outline="black", width=3)
    elif kind == "square":
        draw.rectangle(box, fill=fill, outline="black", width=3)
    elif kind == "triangle":
        draw.polygon(((x0 + x1) // 2, y0, x0, y1, x1, y1), fill=fill, outline="black")
    else:
        raise ValueError(kind)


def build_images(root: Path) -> None:
    from PIL import Image, ImageDraw

    root.mkdir(parents=True, exist_ok=True)

    # Case 1: 5 red circles, 4 blue squares, 3 green triangles.
    im = Image.new("RGB", (448, 448), "white"); d = ImageDraw.Draw(im)
    rows = [
        ("circle", "#e53935", 5, 65),
        ("square", "#1e88e5", 4, 205),
        ("triangle", "#43a047", 3, 345),
    ]
    for kind, color, count, cy in rows:
        spacing = 72
        total = (count - 1) * spacing
        start = 224 - total // 2
        for i in range(count):
            cx = start + i * spacing
            draw_shape(d, kind, (cx - 25, cy - 25, cx + 25, cy + 25), color)
    im.save(root / "weighted-inventory.png")

    # Case 2: quadrant arithmetic, intentionally language-agnostic image.
    im = Image.new("RGB", (448, 448), "white"); d = ImageDraw.Draw(im)
    d.line((224, 12, 224, 436), fill="black", width=7); d.line((12, 224, 436, 224), fill="black", width=7)
    placements = [
        ("triangle", "#e53935", [(70, 70), (155, 70), (112, 155)]),
        ("circle", "#1e88e5", [(292, 95), (375, 145)]),
        ("square", "#43a047", [(65, 285), (155, 285), (65, 375), (155, 375)]),
        ("circle", "#fdd835", [(275, 275), (365, 275), (320, 335), (275, 395), (365, 395)]),
    ]
    for kind, color, points in placements:
        for cx, cy in points:
            draw_shape(d, kind, (cx - 25, cy - 25, cx + 25, cy + 25), color)
    im.save(root / "vi-quadrants.png")

    # Case 3: red -> blue -> green cycle in both axes; blank at row 4, col 3 (green).
    im = Image.new("RGB", (448, 448), "white"); d = ImageDraw.Draw(im)
    colors = ["#e53935", "#1e88e5", "#43a047"]
    left, top, cell = 49, 49, 70
    blank = (3, 2)
    for r in range(5):
        for c in range(5):
            box = (left + c * cell, top + r * cell, left + (c + 1) * cell, top + (r + 1) * cell)
            fill = "white" if (r, c) == blank else colors[(r + c) % 3]
            d.rectangle(box, fill=fill, outline="black", width=4)
    im.save(root / "color-cycle.png")

    # Case 4: 6x6 maze. Wall at column 4 rows 1..5 forces a bottom detour of 13 moves.
    im = Image.new("RGB", (448, 448), "white"); d = ImageDraw.Draw(im)
    left, top, cell = 44, 44, 60
    for r in range(MAZE_N):
        for c in range(MAZE_N):
            pos = (r, c)
            if pos == MAZE_START:
                fill = "#e53935"
            elif pos == MAZE_GOAL:
                fill = "#43a047"
            elif pos in MAZE_BLOCKED:
                fill = "black"
            else:
                fill = "white"
            box = (left + c * cell, top + r * cell, left + (c + 1) * cell, top + (r + 1) * cell)
            d.rectangle(box, fill=fill, outline="#666666", width=3)
    im.save(root / "maze.png")

    # Case 5: distinct diameters; descending order is purple, red, green, yellow, blue.
    im = Image.new("RGB", (448, 448), "white"); d = ImageDraw.Draw(im)
    circles = [
        ("#8e24aa", 58, (95, 115)),
        ("#e53935", 50, (250, 110)),
        ("#43a047", 42, (360, 225)),
        ("#fdd835", 34, (210, 330)),
        ("#1e88e5", 26, (80, 330)),
    ]
    for color, radius, (cx, cy) in circles:
        d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline="black", width=3)
    im.save(root / "size-ordering.png")

    # Case 6: row3 = row1 + row2 in dot counts: [2,3,5], [1,4,5], [3,2,?] => 5.
    im = Image.new("RGB", (448, 448), "white"); d = ImageDraw.Draw(im)
    left, top, cell = 44, 44, 120
    matrix = [[2, 3, 5], [1, 4, 5], [3, 2, None]]
    dot_offsets = [(0, 0), (-25, -22), (25, -22), (-25, 24), (25, 24)]
    for r in range(3):
        for c in range(3):
            x0, y0 = left + c * cell, top + r * cell
            x1, y1 = x0 + cell, y0 + cell
            outline = "#e53935" if matrix[r][c] is None else "black"
            d.rectangle((x0, y0, x1, y1), fill="white", outline=outline, width=4)
            count = matrix[r][c]
            if count is None:
                continue
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            for dx, dy in dot_offsets[:count]:
                d.ellipse((cx + dx - 8, cy + dy - 8, cx + dx + 8, cy + dy + 8), fill="black")
    im.save(root / "dot-matrix.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path("build/dynskip-intelligence"))
    ap.add_argument("--evidence-dir", type=Path)
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--vision-binary", default="./build/kvl_vision")
    ap.add_argument("--baseline-binary", default="./build/kvl_generate_vl")
    ap.add_argument("--candidate-binary", default="./build/kvl_generate_vl_dynskip")
    ap.add_argument("--cache-bytes", type=int, default=536870912)
    ap.add_argument("--revision", default="unknown")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    validate_cases()
    print(
        f"KIMI_DYNSKIP_INTELLIGENCE_SUITE_VALID cases={len(CASES)} "
        f"max_new_total={sum(int(c['max_new']) for c in CASES)} maze_answer=13"
    )
    if args.validate_only:
        return 0
    if args.model_dir is None or args.evidence_dir is None or args.policy is None:
        raise SystemExit("--model-dir, --evidence-dir and --policy are required unless --validate-only")
    if args.cache_bytes <= 0:
        raise SystemExit("--cache-bytes must be positive")

    # Heavy dependencies intentionally imported only for the real run so validate-only remains stdlib-only.
    from kimi_image import write_patches
    from kimi_tokenizer import build_encoding, decode_generated, encode_image_chat

    model = args.model_dir.resolve(); work = args.work_dir.resolve(); evidence = args.evidence_dir.resolve(); policy = args.policy.resolve()
    work.mkdir(parents=True, exist_ok=True); evidence.mkdir(parents=True, exist_ok=True)
    image_root = work / "images"; build_images(image_root)
    enc, _, special = build_encoding(model)
    media_pad_id = special["<|media_pad|>"]
    rows = []

    for index, case in enumerate(CASES, 1):
        root = work / case["id"]; ev = evidence / case["id"]
        root.mkdir(parents=True, exist_ok=True); ev.mkdir(parents=True, exist_ok=True)
        patches = root / "patches.f32"; media = root / "media.f32"; ids_path = root / "prompt.ids"
        gh, gw = write_patches(model, image_root / case["image"], patches)
        media_tokens = (gh // 2) * (gw // 2)
        prompt_ids = encode_image_chat(enc, case["prompt"], media_tokens, "You are a helpful assistant")
        if sum(x == media_pad_id for x in prompt_ids) != media_tokens:
            raise RuntimeError(f"{case['id']}: media token count mismatch")
        ids_path.write_text("\n".join(map(str, prompt_ids)) + "\n", encoding="ascii")
        run([
            args.vision_binary, str(model / "vision.bin"), str(model / "vision.idx"),
            str(patches), str(gh), str(gw), str(media),
        ], stdout_path=ev / "vision.out", stderr_path=ev / "vision.err")

        def generation(variant: str, binary: str, dynamic: bool) -> dict:
            out = ev / f"{variant}.out"; err = ev / f"{variant}.err"
            env = os.environ.copy(); env.pop("KVL_MOE_MASK", None)
            stats_path = ev / "dynskip-stats.tsv"
            if dynamic:
                env["KVL_MOE_DYNSKIP_POLICY"] = str(policy)
                env["KVL_MOE_DYNSKIP_PROMPT_IDS"] = str(ids_path)
                env["KVL_MOE_DYNSKIP_STATS"] = str(stats_path)
            else:
                env.pop("KVL_MOE_DYNSKIP_POLICY", None); env.pop("KVL_MOE_DYNSKIP_PROMPT_IDS", None); env.pop("KVL_MOE_DYNSKIP_STATS", None)
            run([
                binary, str(model / "trunk.bin"), str(model / "trunk.idx"),
                str(model / "experts.bin"), str(model / "experts.idx"),
                str(ids_path), str(media), str(args.cache_bytes), str(case["max_new"]), "0", "1",
            ], env=env, stdout_path=out, stderr_path=err)
            ids = parse_generated(out)
            text = decode_generated(enc, ids, {EOS_ID, IM_END_ID}).strip()
            result = {
                "generated_ids": ids,
                "generated_tokens": len(ids),
                "text": text,
                "score": score_answer(text, case),
                "cache": parse_cache(err),
            }
            if dynamic:
                stat_rows, routed, skipped, non_media = parse_stats(stats_path)
                result.update({"routed": routed, "skipped": skipped, "non_media_skipped": non_media, "stats": stat_rows})
            return result

        baseline = generation("baseline", args.baseline_binary, False)
        candidate = generation("candidate", args.candidate_binary, True)
        div = first_divergence(baseline["generated_ids"], candidate["generated_ids"])
        retention_regression = baseline["score"]["correct"] and not candidate["score"]["correct"]
        result = {
            "id": case["id"], "domain": case["domain"], "prompt": case["prompt"],
            "expected": case["expected"], "max_new": case["max_new"],
            "grid": [gh, gw], "media_tokens": media_tokens, "prompt_tokens": len(prompt_ids),
            "baseline": baseline, "candidate": candidate,
            "comparison": {
                "token_exact": div is None,
                "first_divergence_position": div,
                "intelligence_retention_regression": retention_regression,
                "answer_agree": baseline["score"]["extracted"] == candidate["score"]["extracted"],
                "expert_request_delta": candidate["cache"]["requests"] - baseline["cache"]["requests"],
                "expert_store_bytes_delta_mib": candidate["cache"]["bytes_read_mib"] - baseline["cache"]["bytes_read_mib"],
            },
        }
        rows.append(result)
        (ev / "case-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"KIMI_DYNSKIP_INTELLIGENCE_CASE {index}/{len(CASES)} id={case['id']} "
            f"base_correct={int(baseline['score']['correct'])} cand_correct={int(candidate['score']['correct'])} "
            f"token_exact={int(div is None)} skipped={candidate['skipped']} non_media={candidate['non_media_skipped']} "
            f"base_answer={baseline['score']['extracted']} cand_answer={candidate['score']['extracted']}",
            flush=True,
        )

    baseline_correct = sum(int(r["baseline"]["score"]["correct"]) for r in rows)
    candidate_correct = sum(int(r["candidate"]["score"]["correct"]) for r in rows)
    regressions = [r["id"] for r in rows if r["comparison"]["intelligence_retention_regression"]]
    improvements = [r["id"] for r in rows if not r["baseline"]["score"]["correct"] and r["candidate"]["score"]["correct"]]
    baseline_format = sum(int(r["baseline"]["score"]["format_ok"]) for r in rows)
    candidate_format = sum(int(r["candidate"]["score"]["format_ok"]) for r in rows)
    aggregate = {
        "schema": "kimi-dynskip-vl-intelligence-v1",
        "model": "moonshotai/Kimi-VL-A3B-Instruct",
        "revision": args.revision,
        "policy": str(policy),
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "baseline_correct_cases": baseline_correct,
            "candidate_correct_cases": candidate_correct,
            "baseline_format_ok_cases": baseline_format,
            "candidate_format_ok_cases": candidate_format,
            "baseline_correct_retained_cases": baseline_correct - len(regressions),
            "intelligence_retention_regressions": regressions,
            "candidate_improvements": improvements,
            "token_exact_cases": sum(int(r["comparison"]["token_exact"]) for r in rows),
            "answer_agree_cases": sum(int(r["comparison"]["answer_agree"]) for r in rows),
            "non_media_skipped": sum(int(r["candidate"]["non_media_skipped"]) for r in rows),
            "routed": sum(int(r["candidate"]["routed"]) for r in rows),
            "skipped": sum(int(r["candidate"]["skipped"]) for r in rows),
            "expert_request_delta": sum(int(r["comparison"]["expert_request_delta"]) for r in rows),
            "expert_store_bytes_delta_mib": sum(float(r["comparison"]["expert_store_bytes_delta_mib"]) for r in rows),
            "baseline_generated_tokens": sum(len(r["baseline"]["generated_ids"]) for r in rows),
            "candidate_generated_tokens": sum(len(r["candidate"]["generated_ids"]) for r in rows),
        },
        "claim_boundary": (
            "Small deterministic hard-VL regression suite. Ground-truth score measures only these synthetic tasks; "
            "zero retention regressions does not prove global intelligence or quality preservation. Timing is not a benchmark."
        ),
    }
    out = evidence / "intelligence-summary.json"
    out.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    s = aggregate["summary"]
    print(
        "KIMI_DYNSKIP_INTELLIGENCE_COMPLETE "
        f"baseline_correct={s['baseline_correct_cases']}/{s['cases']} "
        f"candidate_correct={s['candidate_correct_cases']}/{s['cases']} "
        f"retention_regressions={len(s['intelligence_retention_regressions'])} "
        f"token_exact={s['token_exact_cases']}/{s['cases']} skipped={s['skipped']} non_media={s['non_media_skipped']}"
    )
    # A candidate must not lose a capability that the full-Q8 baseline demonstrated.
    # Do not fail solely because the baseline itself cannot solve a hard case.
    if regressions or s["non_media_skipped"] != 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
