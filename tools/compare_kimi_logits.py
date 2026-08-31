#!/usr/bin/env python3
"""Compare KVL float32 logit dump files exactly enough for pruning A/B diagnostics.

Format (little-endian, x86 runtime target):
  8 bytes magic = b"KVLLOG1\\0"
  uint32 version = 1
  uint32 vocab
  repeated records:
    uint32 record_index
    uint32 n_logits (must equal vocab)
    float32 logits[n_logits]
"""
from __future__ import annotations

import argparse
import array
import heapq
import json
import math
import struct
import sys
from pathlib import Path

MAGIC = b"KVLLOG1\0"
HEADER = struct.Struct("<8sII")
RECORD = struct.Struct("<II")


def read_dump(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: truncated header")
    magic, version, vocab = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != 1 or vocab <= 0:
        raise ValueError(f"{path}: bad header magic/version/vocab")
    pos = HEADER.size
    records = []
    while pos < len(data):
        if pos + RECORD.size > len(data):
            raise ValueError(f"{path}: truncated record header")
        record_index, n = RECORD.unpack_from(data, pos)
        pos += RECORD.size
        if n != vocab:
            raise ValueError(f"{path}: record {record_index} n={n} vocab={vocab}")
        need = n * 4
        if pos + need > len(data):
            raise ValueError(f"{path}: truncated record {record_index}")
        vals = array.array("f")
        vals.frombytes(data[pos:pos + need])
        if sys.byteorder != "little":
            vals.byteswap()
        pos += need
        if any(not math.isfinite(x) for x in vals):
            raise ValueError(f"{path}: non-finite logit in record {record_index}")
        records.append((record_index, vals))
    if not records:
        raise ValueError(f"{path}: no records")
    return vocab, records


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(float(x) - m) for x in values))


def top_ids(values, k):
    k = min(k, len(values))
    return heapq.nlargest(k, range(len(values)), key=values.__getitem__)


def compare_record(a, b, topk):
    if len(a) != len(b):
        raise ValueError("logit vector size mismatch")
    n = len(a)
    max_abs = 0.0
    sq = 0.0
    for x, y in zip(a, b):
        d = float(x) - float(y)
        ad = abs(d)
        if ad > max_abs:
            max_abs = ad
        sq += d * d
    rms = math.sqrt(sq / n) if n else 0.0

    za = logsumexp(a)
    zb = logsumexp(b)
    kl_ab = 0.0
    kl_ba = 0.0
    js = 0.0
    max_prob_delta = 0.0
    probability_overlap = 0.0
    total_variation_l1 = 0.0
    for x, y in zip(a, b):
        pa = math.exp(float(x) - za)
        pb = math.exp(float(y) - zb)
        dp = abs(pa - pb)
        probability_overlap += min(pa, pb)
        total_variation_l1 += dp
        if dp > max_prob_delta:
            max_prob_delta = dp
        if pa > 0.0 and pb > 0.0:
            kl_ab += pa * math.log(pa / pb)
            kl_ba += pb * math.log(pb / pa)
        m = 0.5 * (pa + pb)
        if pa > 0.0:
            js += 0.5 * pa * math.log(pa / m)
        if pb > 0.0:
            js += 0.5 * pb * math.log(pb / m)

    ta = top_ids(a, topk)
    tb = top_ids(b, topk)
    inter = set(ta) & set(tb)
    argmax_a = ta[0]
    argmax_b = tb[0]
    return {
        "max_abs_logit_delta": max_abs,
        "rms_logit_delta": rms,
        "argmax_baseline": argmax_a,
        "argmax_variant": argmax_b,
        "argmax_agree": argmax_a == argmax_b,
        "topk": min(topk, n),
        "topk_overlap": len(inter),
        "topk_overlap_fraction": len(inter) / min(topk, n),
        "kl_baseline_to_variant": kl_ab,
        "kl_variant_to_baseline": kl_ba,
        "js_divergence": js,
        "max_probability_delta": max_prob_delta,
        "probability_overlap": probability_overlap,
        "total_variation": 0.5 * total_variation_l1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("variant", type=Path)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.topk <= 0:
        raise SystemExit("--topk must be positive")

    vocab_a, a = read_dump(args.baseline)
    vocab_b, b = read_dump(args.variant)
    if vocab_a != vocab_b:
        raise ValueError(f"vocab mismatch: {vocab_a} vs {vocab_b}")
    if [i for i, _ in a] != [i for i, _ in b]:
        raise ValueError("record indices differ")

    records = []
    for (idx, av), (_, bv) in zip(a, b):
        r = compare_record(av, bv, args.topk)
        r["record_index"] = idx
        records.append(r)
    report = {
        "baseline": str(args.baseline),
        "variant": str(args.variant),
        "vocab": vocab_a,
        "records": records,
        "summary": {
            "records": len(records),
            "argmax_agree_records": sum(int(r["argmax_agree"]) for r in records),
            "max_abs_logit_delta": max(r["max_abs_logit_delta"] for r in records),
            "max_js_divergence": max(r["js_divergence"] for r in records),
            "min_topk_overlap_fraction": min(r["topk_overlap_fraction"] for r in records),
            "mean_probability_overlap": sum(r["probability_overlap"] for r in records) / len(records),
            "min_probability_overlap": min(r["probability_overlap"] for r in records),
            "max_total_variation": max(r["total_variation"] for r in records),
            "esap_claim_boundary": (
                "probability_overlap equals token-level ESAP only when baseline and variant "
                "records use identical teacher-forced prefixes"
            ),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    s = report["summary"]
    print(
        "KIMI_LOGIT_COMPARE_PASS "
        f"records={s['records']} argmax_agree={s['argmax_agree_records']} "
        f"max_abs={s['max_abs_logit_delta']:.9g} max_js={s['max_js_divergence']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
