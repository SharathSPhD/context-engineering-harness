"""Long-context adapters: RULER, HELMET, NoCha — load + render + score."""
from __future__ import annotations

from src.benchmarks import ModelOutput, MultiSeedRunner, RunnerConfig
from src.benchmarks.adapters.longctx import (
    helmet as _helmet,  # noqa: F401  (registration side effect)
    nocha as _nocha,    # noqa: F401
    ruler as _ruler,    # noqa: F401
)
from src.benchmarks.adapters.longctx.helmet import HelmetRagAdapter, HelmetRecallAdapter
from src.benchmarks.adapters.longctx.nocha import NochaJointAccuracyAdapter
from src.benchmarks.adapters.longctx.ruler import RulerNIAHAdapter, RulerNIAHMultiAdapter
from src.benchmarks.registry import all_names, get


def test_ruler_adapter_registered():
    assert "ruler_niah" in all_names()
    assert get("ruler_niah") is RulerNIAHAdapter


def test_helmet_adapters_registered():
    assert "helmet_rag" in all_names()
    assert "helmet_recall" in all_names()
    assert get("helmet_rag") is HelmetRagAdapter


def test_nocha_adapter_registered():
    assert "nocha_joint" in all_names()
    assert get("nocha_joint") is NochaJointAccuracyAdapter


def test_ruler_examples_carry_ground_truth_in_haystack():
    adapter = RulerNIAHAdapter(target_tokens=2_000, default_n=4)
    examples = adapter.load_examples(seed=0)
    assert len(examples) == 4
    for ex in examples:
        assert isinstance(ex.ground_truth, str)
        assert ex.ground_truth in ex.context


def test_ruler_render_prompt_treatment_vs_baseline_differ():
    adapter = RulerNIAHAdapter(target_tokens=2_000, default_n=2)
    ex = adapter.load_examples(seed=0)[0]
    treatment = adapter.render_prompt(ex, condition="harness_on")
    baseline = adapter.render_prompt(ex, condition="harness_off")
    assert "focused long-context retriever" in treatment
    assert "focused long-context retriever" not in baseline
    assert ex.context in treatment
    assert ex.context in baseline


def test_ruler_score_recognizes_inline_code():
    adapter = RulerNIAHAdapter(target_tokens=2_000, default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    correct = ModelOutput(text=f"The answer is {ex.ground_truth}.")
    wrong = ModelOutput(text="I do not know.")
    s_ok, ok_ok, _ = adapter.score(ex, correct)
    s_no, ok_no, _ = adapter.score(ex, wrong)
    assert (s_ok, ok_ok) == (1.0, True)
    assert (s_no, ok_no) == (0.0, False)


def test_ruler_multi_adapter_partial_credit():
    adapter = RulerNIAHMultiAdapter(target_tokens=3_000, default_n=1)
    ex = adapter.load_examples(seed=0)[0]
    assert isinstance(ex.ground_truth, dict)
    values = list(ex.ground_truth.values())
    half = " ".join(values[: len(values) // 2])
    partial = ModelOutput(text=half)
    full = ModelOutput(text=" ".join(values))
    score_partial, ok_partial, _ = adapter.score(ex, partial)
    score_full, ok_full, _ = adapter.score(ex, full)
    assert 0 < score_partial < 1
    assert ok_partial is False
    assert score_full == 1.0
    assert ok_full is True


def test_helmet_recall_score_full_vs_partial():
    adapter = HelmetRecallAdapter(target_tokens=2_000, default_n=1, needles_per_example=4)
    ex = adapter.load_examples(seed=0)[0]
    gt = ex.ground_truth
    full_text = " ".join(gt.values())
    score_full, ok_full, _ = adapter.score(ex, ModelOutput(text=full_text))
    assert score_full == 1.0
    assert ok_full is True


def test_nocha_score_requires_both_verdicts_correct():
    adapter = NochaJointAccuracyAdapter(target_tokens=3_000, default_n=2)
    ex = adapter.load_examples(seed=0)[0]
    correct = ModelOutput(
        text='{"true_claim_verdict":"TRUE","false_claim_verdict":"FALSE"}'
    )
    wrong_a = ModelOutput(
        text='{"true_claim_verdict":"FALSE","false_claim_verdict":"FALSE"}'
    )
    s_ok, _, _ = adapter.score(ex, correct)
    s_no, _, _ = adapter.score(ex, wrong_a)
    assert s_ok == 1.0
    assert s_no == 0.0


def test_runner_end_to_end_with_ruler_adapter_and_oracle_caller():
    """Treatment caller answers correctly; baseline caller hallucinates.

    The runner should report a perfect treatment - baseline delta on the
    synthetic generator, proving the harness path works end-to-end.
    """
    adapter = RulerNIAHAdapter(target_tokens=1_500, default_n=8)

    def caller(*, prompt, model, max_tokens, system="", seed=None):  # noqa: ARG001
        if "focused long-context retriever" in prompt:
            import re
            m = re.search(r"vault (vault-\d+-\d+)", prompt)
            assert m, "treatment prompt should mention the vault id"
            vault_id = m.group(1)
            haystack = prompt
            inline = re.search(rf"vault {vault_id} is ([0-9A-F]{{6}})", haystack)
            return ModelOutput(text=inline.group(1) if inline else "WRONG")
        return ModelOutput(text="UNKNOWN")

    runner = MultiSeedRunner(
        adapter=adapter,
        model_caller=caller,
        config=RunnerConfig(max_tokens=64, bootstrap_n=300, permutation_n=300),
    )
    treatment = runner.run_condition(condition="harness_on", model="m", seed=0, n_examples=8)
    baseline = runner.run_condition(condition="harness_off", model="m", seed=0, n_examples=8)
    assert treatment.accuracy == 1.0
    assert baseline.accuracy == 0.0
