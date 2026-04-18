"""Hallucination-family adapters: HaluEval, TruthfulQA, FACTS-Grounding."""
from __future__ import annotations

import logging

import pytest

from src.benchmarks import ModelOutput, MultiSeedRunner, RunnerConfig
from src.benchmarks.adapters.hallu import (  # noqa: F401  (registration side effect)
    facts_grounding as _facts,
    halu_eval as _halu,
    truthful_qa as _truthful,
)
from src.benchmarks.adapters.hallu.facts_grounding import FactsGroundingAdapter
from src.benchmarks.adapters.hallu.halu_eval import (
    HaluEvalDiscriminateAdapter,
    HaluEvalQAAdapter,
)
from src.benchmarks.adapters.hallu.truthful_qa import TruthfulQAAdapter
from src.benchmarks.registry import all_names, get


def test_hallu_adapters_registered():
    for name in (
        "halu_eval_qa",
        "halu_eval_discriminate",
        "truthful_qa",
        "facts_grounding",
    ):
        assert name in all_names(), f"adapter {name} should self-register"
    assert get("halu_eval_qa") is HaluEvalQAAdapter
    assert get("halu_eval_discriminate") is HaluEvalDiscriminateAdapter
    assert get("truthful_qa") is TruthfulQAAdapter
    assert get("facts_grounding") is FactsGroundingAdapter


# --- HaluEval QA ----------------------------------------------------------
def test_haluqa_synthetic_examples_have_paired_distractor():
    adapter = HaluEvalQAAdapter(default_n=5)
    examples = adapter.load_examples(seed=0)
    assert len(examples) == 5
    for ex in examples:
        assert isinstance(ex.ground_truth, str) and ex.ground_truth
        assert ex.metadata["hallucinated"] != ex.ground_truth


def test_haluqa_score_correct_partial_and_abstention():
    adapter = HaluEvalQAAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    correct = ModelOutput(text=f"The answer is {ex.ground_truth}.")
    abstain = ModelOutput(text="I do not know.")
    wrong = ModelOutput(text="Definitely something else.")
    s_ok, ok_ok, _ = adapter.score(ex, correct)
    s_ab, ok_ab, _ = adapter.score(ex, abstain)
    s_no, ok_no, _ = adapter.score(ex, wrong)
    assert (s_ok, ok_ok) == (1.0, True)
    assert (s_ab, ok_ab) == (0.5, False)
    assert (s_no, ok_no) == (0.0, False)


def test_haluqa_render_prompt_treatment_includes_abstention_clause():
    adapter = HaluEvalQAAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    on = adapter.render_prompt(ex, condition="harness_on")
    off = adapter.render_prompt(ex, condition="harness_off")
    assert "I do not know" in on
    assert "I do not know" not in off


# --- HaluEval discriminate -----------------------------------------------
def test_halu_discriminate_balanced_polarities():
    adapter = HaluEvalDiscriminateAdapter(default_n=10)
    examples = adapter.load_examples(seed=0)
    polarities = [ex.metadata["polarity"] for ex in examples]
    assert polarities.count("true") == polarities.count("false") == 5


def test_halu_discriminate_score_extracts_verdict():
    adapter = HaluEvalDiscriminateAdapter(default_n=2)
    examples = adapter.load_examples(seed=0)
    true_ex = next(ex for ex in examples if ex.ground_truth == "TRUE")
    false_ex = next(ex for ex in examples if ex.ground_truth == "FALSE")
    s_tt, ok_tt, _ = adapter.score(true_ex, ModelOutput(text="My verdict: TRUE"))
    s_tf, ok_tf, _ = adapter.score(true_ex, ModelOutput(text="The answer is FALSE"))
    s_ff, ok_ff, _ = adapter.score(false_ex, ModelOutput(text="FALSE."))
    assert (s_tt, ok_tt) == (1.0, True)
    assert (s_tf, ok_tf) == (0.0, False)
    assert (s_ff, ok_ff) == (1.0, True)


# --- TruthfulQA -----------------------------------------------------------
def test_truthful_qa_synthetic_has_misconception_metadata():
    adapter = TruthfulQAAdapter(default_n=4)
    examples = adapter.load_examples(seed=0)
    assert len(examples) == 4
    for ex in examples:
        assert ex.metadata["misconception"]
        assert _is_misconception_distinct(ex.ground_truth, ex.metadata["misconception"])


def _is_misconception_distinct(truthful: str, misconception: str) -> bool:
    return truthful.lower().strip() != misconception.lower().strip()


def test_truthful_qa_score_truthful_vs_misconception():
    adapter = TruthfulQAAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    truthful = ModelOutput(text=ex.ground_truth)
    misc = ModelOutput(text=ex.metadata["misconception"])
    s_t, ok_t, _ = adapter.score(ex, truthful)
    s_m, ok_m, _ = adapter.score(ex, misc)
    assert ok_t is True and s_t == 1.0
    assert ok_m is False and s_m == 0.0


def test_truthful_qa_treatment_prompt_includes_misconception_first_pattern():
    adapter = TruthfulQAAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    on = adapter.render_prompt(ex, condition="harness_on")
    off = adapter.render_prompt(ex, condition="harness_off")
    assert "Misconception" in on
    assert "Misconception" not in off


# --- FACTS-Grounding ------------------------------------------------------
def test_facts_grounding_examples_carry_passage_in_context():
    adapter = FactsGroundingAdapter(default_n=4)
    examples = adapter.load_examples(seed=0)
    assert len(examples) == 4
    for ex in examples:
        assert ex.context
        assert ex.ground_truth
        assert ex.ground_truth.lower() in ex.context.lower()


def test_facts_grounding_treatment_prompt_includes_insufficient_clause():
    adapter = FactsGroundingAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    on = adapter.render_prompt(ex, condition="harness_on")
    off = adapter.render_prompt(ex, condition="harness_off")
    assert "INSUFFICIENT" in on
    assert "INSUFFICIENT" not in off
    assert ex.context in on
    assert ex.context in off


def test_facts_grounding_score_grounded_vs_hallucinated():
    adapter = FactsGroundingAdapter(default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    grounded = ModelOutput(text=f"From the passage: {ex.ground_truth}.")
    fabricated = ModelOutput(text="Made-up answer that is not in the passage.")
    s_g, ok_g, _ = adapter.score(ex, grounded)
    s_f, ok_f, _ = adapter.score(ex, fabricated)
    assert ok_g is True and s_g == 1.0
    assert ok_f is False and s_f == 0.0


# --- Real loader fallback semantics --------------------------------------
def test_haluqa_load_real_falls_back_when_disabled(monkeypatch, caplog):
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    adapter = HaluEvalQAAdapter(default_n=3, load_real=True)
    with caplog.at_level(logging.WARNING):
        examples = adapter.load_examples(seed=0)
    assert len(examples) == 3
    assert any("HaluEval QA real loader unavailable" in rec.message for rec in caplog.records)


def test_truthful_qa_load_real_falls_back_when_disabled(monkeypatch, caplog):
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    adapter = TruthfulQAAdapter(default_n=2, load_real=True)
    with caplog.at_level(logging.WARNING):
        examples = adapter.load_examples(seed=0)
    assert len(examples) == 2
    assert any("TruthfulQA real loader unavailable" in rec.message for rec in caplog.records)


def test_facts_grounding_load_real_falls_back_when_disabled(monkeypatch, caplog):
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    adapter = FactsGroundingAdapter(default_n=2, load_real=True)
    with caplog.at_level(logging.WARNING):
        examples = adapter.load_examples(seed=0)
    assert len(examples) == 2
    assert any(
        "FACTS-Grounding real loader unavailable" in rec.message for rec in caplog.records
    )


# --- End-to-end via the runner -------------------------------------------
def test_runner_drives_haluqa_with_oracle_caller():
    """Treatment caller knows the answer; baseline confidently hallucinates.

    Asserts the runner produces a non-zero treatment - baseline delta on
    the hallucination family without touching the network.
    """
    adapter = HaluEvalQAAdapter(default_n=10)

    def caller(*, prompt, model, max_tokens, system="", seed=None):  # noqa: ARG001
        examples = adapter.load_examples(seed=seed or 0)
        for ex in examples:
            if ex.prompt in prompt:
                if "If you are not certain" in prompt:
                    return ModelOutput(text=ex.ground_truth)
                return ModelOutput(text=ex.metadata["hallucinated"])
        return ModelOutput(text="UNKNOWN")

    runner = MultiSeedRunner(
        adapter=adapter,
        model_caller=caller,
        config=RunnerConfig(max_tokens=64, bootstrap_n=200, permutation_n=200),
    )
    treatment = runner.run_condition(condition="harness_on", model="m", seed=0, n_examples=10)
    baseline = runner.run_condition(condition="harness_off", model="m", seed=0, n_examples=10)
    assert treatment.accuracy >= 0.9
    assert baseline.accuracy <= 0.1
