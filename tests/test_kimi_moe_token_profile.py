#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "kimi_moe_token_profile.py"
spec = importlib.util.spec_from_file_location("kimi_moe_token_profile", TOOL)
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
assert spec.loader is not None; spec.loader.exec_module(mod)


def write_trace(path: Path, rows: list[tuple]) -> None:
    path.write_text("\n".join(" ".join(map(str, row)) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    ids = [
        mod.IM_SYSTEM_ID, 101, mod.IM_MIDDLE_ID, 102, mod.IM_END_ID,
        mod.IM_USER_ID, 103, mod.IM_MIDDLE_ID,
        mod.MEDIA_START_ID, 104, mod.MEDIA_CONTENT_ID, mod.MEDIA_PAD_ID, mod.MEDIA_END_ID,
        201, 202, mod.IM_END_ID,
        mod.IM_ASSISTANT_ID, 105, mod.IM_MIDDLE_ID,
    ]
    expected = [
        "system", "system", "system", "system", "system",
        "user_control", "user_control", "user_control",
        "media_control", "media_control", "media_control", "media_pad", "media_control",
        "user_content", "user_content", "user_control",
        "assistant_transition", "assistant_transition", "assistant_transition",
    ]
    assert mod.classify_prompt_tokens(ids) == expected
    assert mod.event_position(14, 1, len(ids)) == 13
    assert mod.event_position(len(ids) + 12, 2, len(ids)) == 11

    with tempfile.TemporaryDirectory() as td:
        root = Path(td); ids_path = root / "prompt.ids"; trace = root / "trace.tsv"; out = root / "out"
        ids_path.write_text("\n".join(map(str, ids)) + "\n", encoding="ascii")
        rows = [
            (1, 1, 0, 0.5, 10.0, 5.0, 50.0), (1, 1, 1, 0.4, 1.0, 0.4, 1.0),
            (12, 1, 2, 0.3, 8.0, 2.4, 20.0), (12, 1, 3, 0.2, 2.0, 0.4, 2.0),
            (14, 1, 4, 0.6, 7.0, 4.2, 15.0), (14, 1, 5, 0.1, 1.0, 0.1, 1.0),
            (17, 1, 6, 0.7, 6.0, 4.2, 12.0), (17, 1, 7, 0.1, 1.0, 0.1, 1.0),
        ]
        write_trace(trace, rows)
        report = mod.build_token_profile([("demo", trace, ids_path)], n_experts=8)
        assert report["schema"] == "kimi-moe-token-aware-profile-v1"
        assert report["token_families"] == ["content", "control", "media"]
        assert report["trace_formats"] == {"legacy_v1": 0, "v2": 1}
        content = report["by_family"]["content"]
        slot4 = next(x for x in content if x["layer"] == 1 and x["expert"] == 4)
        assert slot4["metrics"]["selected"] == 1
        media = report["by_family"]["media"]
        slot2 = next(x for x in media if x["layer"] == 1 and x["expert"] == 2)
        assert slot2["metrics"]["selected"] == 1
        assert any(x["token_family"] == "content" and x["expert_a"] == 4 and x["expert_b"] == 5 for x in report["coactivation"])
        mod.write_outputs(out, report)
        for name in ("token-aware-report.json", "token-family-profile.tsv", "token-class-profile.tsv", "domain-token-family-profile.tsv", "token-family-coactivation.tsv"):
            assert (out / name).is_file() and (out / name).stat().st_size > 0

    print("KIMI_MOE_TOKEN_PROFILE_UNIT_PASS")


if __name__ == "__main__":
    main()
