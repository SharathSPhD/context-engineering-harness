"""SWE-bench Verified adapter — load, render, heuristic-score, harness hook."""
from __future__ import annotations

import logging

import pytest

from src.benchmarks import ModelOutput
from src.benchmarks.adapters.swebench import verified as _verified  # noqa: F401
from src.benchmarks.adapters.swebench.verified import (
    SWEBenchVerifiedAdapter,
    SwebenchHarnessUnavailable,
)
from src.benchmarks.registry import all_names, get


def test_swebench_adapter_registered():
    assert "swe_bench_verified" in all_names()
    assert get("swe_bench_verified") is SWEBenchVerifiedAdapter


def test_swebench_synthetic_examples_carry_gold_patch_and_target_file():
    adapter = SWEBenchVerifiedAdapter(default_n=4)
    examples = adapter.load_examples(seed=0)
    assert len(examples) == 4
    for ex in examples:
        gt = ex.ground_truth
        assert isinstance(gt, dict)
        assert gt["patch"].startswith("diff --git")
        assert gt["file_path"]
        assert gt["fail_to_pass"]


def test_swebench_render_prompt_treatment_includes_discipline_clause():
    adapter = SWEBenchVerifiedAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    on = adapter.render_prompt(ex, condition="harness_on")
    off = adapter.render_prompt(ex, condition="harness_off")
    assert "MINIMAL unified diff" in on
    assert "MINIMAL unified diff" not in off
    assert "Likely target file" in on
    assert ex.prompt in on and ex.prompt in off


def test_swebench_score_perfect_on_gold_patch_replay():
    adapter = SWEBenchVerifiedAdapter(default_n=2)
    ex = adapter.load_examples(seed=0)[0]
    gold = ex.ground_truth["patch"]
    score, ok, _ = adapter.score(ex, ModelOutput(text=gold))
    assert score == 1.0
    assert ok is True


def test_swebench_score_partial_credit_on_right_file_wrong_lines():
    adapter = SWEBenchVerifiedAdapter(default_n=2)
    ex = adapter.load_examples(seed=0)[0]
    target = ex.ground_truth["file_path"]
    pred = (
        f"diff --git a/{target} b/{target}\n"
        f"--- a/{target}\n"
        f"+++ b/{target}\n"
        "@@ -1,1 +1,1 @@\n"
        "-old totally unrelated line\n"
        "+new totally unrelated line\n"
    )
    score, ok, _ = adapter.score(ex, ModelOutput(text=pred))
    # Right file (overlap = 1.0) but no matching lines → 0.5*1.0 + 0.5*0.0 = 0.5,
    # which lands exactly at the default threshold (passes).
    assert 0.4 <= score <= 0.6
    assert ok is True


def test_swebench_score_zero_on_unrelated_file():
    adapter = SWEBenchVerifiedAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    pred = (
        "diff --git a/totally/unrelated.py b/totally/unrelated.py\n"
        "--- a/totally/unrelated.py\n"
        "+++ b/totally/unrelated.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    score, ok, _ = adapter.score(ex, ModelOutput(text=pred))
    assert score == 0.0
    assert ok is False


def test_swebench_score_zero_on_empty_output():
    adapter = SWEBenchVerifiedAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    score, ok, _ = adapter.score(ex, ModelOutput(text=""))
    assert score == 0.0
    assert ok is False


def test_swebench_load_real_falls_back_when_disabled(monkeypatch, caplog):
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    adapter = SWEBenchVerifiedAdapter(default_n=2, load_real=True)
    with caplog.at_level(logging.WARNING):
        examples = adapter.load_examples(seed=0)
    assert len(examples) == 2
    assert any(
        "SWE-bench Verified real loader unavailable" in rec.message
        for rec in caplog.records
    )


def test_verify_with_swebench_harness_raises_when_unavailable():
    """When the docker harness package isn't installed, verify_* must
    raise SwebenchHarnessUnavailable so callers can degrade explicitly
    instead of silently treating heuristic scores as validated outcomes.
    """
    adapter = SWEBenchVerifiedAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    try:
        import swebench.harness.run_evaluation  # noqa: F401
    except ImportError:
        with pytest.raises(SwebenchHarnessUnavailable):
            adapter.verify_with_swebench_harness(ex, ModelOutput(text=""))
    else:  # pragma: no cover — only fires in environments with swebench installed
        pytest.skip("swebench harness installed; cannot exercise the unavailable branch")
