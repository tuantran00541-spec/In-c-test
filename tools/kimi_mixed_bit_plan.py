#!/usr/bin/env python3
"""Choose a simple mixed-bit expert assignment from measured sensitivity JSON.

This tool consumes one or more `kimi-expert-quant-sensitivity-v1` JSON files
and solves a discrete storage/error trade-off with dynamic programming.  It is
an offline planning tool only; it does not modify weights or write a native
runtime store.

Each expert must choose exactly one available candidate bit width.  The caller
sets a projected byte budget.  The objective is the sum of a selected measured
error field (default: relative_l2).  This intentionally avoids hand-authored
importance weights until real sensitivity measurements exist.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Choice:
    bits: int
    bytes: int
    error: float


def _expert_key(doc: dict, fallback: str) -> str:
    meta = doc.get("metadata") or {}
    if "layer" in meta and "expert" in meta:
        return f"L{int(meta['layer']):02d}E{int(meta['expert']):02d}"
    return fallback


def load_choices(paths: list[Path], error_field: str) -> list[tuple[str, list[Choice]]]:
    experts = []
    seen = set()
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema") != "kimi-expert-quant-sensitivity-v1":
            raise ValueError(f"{path}: unsupported schema {doc.get('schema')!r}")
        key = _expert_key(doc, path.stem)
        if key in seen:
            raise ValueError(f"duplicate expert key {key}")
        seen.add(key)
        choices = []
        for c in doc.get("candidates", []):
            if error_field not in c:
                raise ValueError(f"{path}: candidate missing {error_field}")
            choices.append(
                Choice(
                    bits=int(c["bits"]),
                    bytes=int(c["projected_total_bytes_f16_scales"]),
                    error=float(c[error_field]),
                )
            )
        if not choices:
            raise ValueError(f"{path}: no candidates")
        choices.sort(key=lambda x: (-x.bits, x.bytes, x.error))
        experts.append((key, choices))
    return experts


def optimize(experts: list[tuple[str, list[Choice]]], budget_bytes: int, quantum: int) -> dict:
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    budget_units = budget_bytes // quantum
    if budget_units <= 0:
        raise ValueError("budget is smaller than one quantum")

    # DP maps used budget units -> (total_error, tuple(choice_index,...)).  Bytes
    # are rounded *up* to the quantum so a returned plan never exceeds the
    # requested byte budget merely because of discretization.
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for _, choices in experts:
        nxt: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (err, picks) in dp.items():
            for idx, c in enumerate(choices):
                units = (c.bytes + quantum - 1) // quantum
                new_used = used + units
                if new_used > budget_units:
                    continue
                new_err = err + c.error
                cur = nxt.get(new_used)
                if cur is None or new_err < cur[0]:
                    nxt[new_used] = (new_err, picks + (idx,))
        dp = nxt
        if not dp:
            raise ValueError("no feasible assignment under projected byte budget")

    # Primary objective error, secondary objective fewer projected bytes.
    used_units, (total_error, picks) = min(dp.items(), key=lambda kv: (kv[1][0], kv[0]))
    assignment = []
    exact_bytes = 0
    for (key, choices), idx in zip(experts, picks):
        c = choices[idx]
        exact_bytes += c.bytes
        assignment.append({"expert": key, "bits": c.bits, "projected_bytes": c.bytes, "error": c.error})
    if exact_bytes > budget_bytes:
        raise AssertionError("quantized DP plan exceeded exact byte budget")
    return {
        "schema": "kimi-mixed-bit-plan-v1",
        "projection_only": True,
        "budget_bytes": budget_bytes,
        "quantum_bytes": quantum,
        "projected_bytes": exact_bytes,
        "total_error": total_error,
        "expert_count": len(experts),
        "assignment": assignment,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--budget-bytes", required=True, type=int)
    ap.add_argument("--error-field", default="relative_l2")
    ap.add_argument("--quantum-bytes", type=int, default=4096)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    experts = load_choices(args.inputs, args.error_field)
    plan = optimize(experts, args.budget_bytes, args.quantum_bytes)
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
