#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_kimi_pruning_multi_screen.py"
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("multi", TOOL)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def metrics(text_prompts=14, text_exact=14, text_argmax=14, text_top10=1.0, text_js=0.001,
            vl_cases=6, vl_exact=6, vl_argmax=6, vl_top10=1.0, vl_js=0.001):
    text = {
        "prompts": text_prompts, "first_token_exact": text_exact,
        "logit_argmax_agree": text_argmax, "logit_min_topk_overlap": text_top10,
        "logit_max_js_divergence": text_js,
    }
    vl = {
        "cases": vl_cases, "first_token_exact": vl_exact,
        "logit_argmax_agree": vl_argmax, "logit_min_topk_overlap": vl_top10,
        "logit_max_js_divergence": vl_js,
    }
    return text, vl


def main():
    t, v = metrics()
    ok, failures = mod.screen_pass(t, v)
    assert ok and failures == []
    t, v = metrics(vl_exact=5)
    ok, failures = mod.screen_pass(t, v)
    assert not ok and any(x.startswith("vl_exact=") for x in failures)
    t, v = metrics(text_js=0.006)
    ok, failures = mod.screen_pass(t, v)
    assert not ok and any(x.startswith("text_js=") for x in failures)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        a = td / "a.mask"
        b = td / "b.mask"
        a.write_text("1 1\n", encoding="utf-8")
        b.write_text("1 1\n2 2\n", encoding="utf-8")
        rows = mod.candidate_specs([f"a={a}", f"b={b}"])
        assert [r["disabled_count"] for r in rows] == [1, 2]
        bad = td / "bad.mask"
        bad.write_text("3 3\n", encoding="utf-8")
        try:
            mod.candidate_specs([f"a={a}", f"bad={bad}"])
        except ValueError as e:
            assert "nested" in str(e)
        else:
            raise AssertionError("non-nested masks should fail")
    print("KIMI_MULTI_SCREEN_TEST_PASS")


if __name__ == "__main__":
    main()
