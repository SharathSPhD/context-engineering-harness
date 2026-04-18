"""End-to-end runner tests using a deterministic synthetic adapter + fixture caller."""
from __future__ import annotations

from dataclasses import dataclass

from src.benchmarks import (
    BenchmarkAdapter,
    BenchmarkExample,
    HypothesisSpec,
    ModelOutput,
    MultiSeedRunner,
    RunnerConfig,
    TargetDirection,
)
from src.benchmarks.registry import all_names, get, register


@dataclass
class _FactRetrievalAdapter(BenchmarkAdapter):
    """Synthetic adapter: 'treatment' returns the right answer; 'baseline'
    returns a near-miss so we can verify the runner stats end-to-end."""

    name: str = "_fact_retrieval_test"

    def load_examples(self, *, n: int | None = None, seed: int = 0) -> list[BenchmarkExample]:
        n = n or 20
        return [
            BenchmarkExample(
                id=f"fact-{i:03d}",
                prompt=f"What is fact {i}?",
                ground_truth=f"answer-{i}",
                metadata={"i": i},
            )
            for i in range(n)
        ]

    def render_prompt(self, example: BenchmarkExample, *, condition: str) -> str:
        return f"[{condition}] {example.prompt}"

    def system_prompt(self, *, condition: str) -> str:
        return f"You are running under condition={condition}"

    def score(self, example: BenchmarkExample, output: ModelOutput) -> tuple[float, bool, str]:
        gt = str(example.ground_truth)
        pred = output.text.strip()
        ok = pred == gt
        return (1.0 if ok else 0.0), ok, pred


def _fixture_caller(*, prompt: str, model: str, max_tokens: int, system: str = "", seed: int | None = None) -> ModelOutput:  # noqa: ARG001
    """Treatment ⇒ correct; baseline ⇒ correct on the first 5 examples only.

    This guarantees treatment dominates baseline by 0.75 - 0.25 = 0.50 on n=20.
    """
    fact_idx = int(prompt.rsplit(" ", 1)[-1].rstrip("?"))
    if "[treatment]" in prompt:
        return ModelOutput(text=f"answer-{fact_idx}", input_tokens=10, output_tokens=5)
    if fact_idx < 5:
        return ModelOutput(text=f"answer-{fact_idx}", input_tokens=10, output_tokens=5)
    return ModelOutput(text="WRONG", input_tokens=10, output_tokens=5)


def _make_runner() -> MultiSeedRunner:
    adapter = _FactRetrievalAdapter()
    return MultiSeedRunner(
        adapter=adapter,
        model_caller=_fixture_caller,
        config=RunnerConfig(max_tokens=64, bootstrap_n=500, permutation_n=500),
    )


def test_run_condition_returns_one_run_with_paired_results():
    runner = _make_runner()
    run = runner.run_condition(condition="treatment", model="m", seed=0, n_examples=10)
    assert run.n == 10
    assert run.accuracy == 1.0
    assert all(r.condition == "treatment" for r in run.results)


def test_run_hypothesis_treatment_beats_baseline_with_significance():
    runner = _make_runner()
    spec = HypothesisSpec(
        hypothesis_id="HX",
        description="treatment retrieves correctly more often than baseline",
        adapter_name="_fact_retrieval_test",
        treatment_condition="treatment",
        baseline_condition="baseline",
        metric="accuracy",
        direction=TargetDirection.GREATER,
        delta=0.10,
        n_examples=20,
        seeds=(0, 1, 2),
        models=("m",),
        significance_alpha=0.05,
    )
    outcome = runner.run_hypothesis(spec)
    assert outcome.treatment_metric == 1.0
    assert outcome.baseline_metric == 5 / 20
    assert outcome.delta_observed == 0.75
    assert outcome.target_met is True
    assert outcome.p_value < 0.05
    assert outcome.cohens_d > 0
    assert outcome.n_seeds_used == 3


def test_run_hypothesis_target_not_met_when_delta_below_threshold():
    runner = _make_runner()
    spec = HypothesisSpec(
        hypothesis_id="HX_strict",
        description="needs delta >= 0.95 to pass",
        adapter_name="_fact_retrieval_test",
        treatment_condition="treatment",
        baseline_condition="baseline",
        metric="accuracy",
        direction=TargetDirection.GREATER,
        delta=0.95,
        n_examples=20,
        seeds=(0,),
        models=("m",),
    )
    outcome = runner.run_hypothesis(spec)
    assert outcome.delta_observed == 0.75
    assert outcome.target_met is False


def test_run_hypothesis_equiv_direction_passes_when_runs_match():
    """Equivalence: treatment ≈ baseline within delta is the win condition."""

    class _IdAdapter(_FactRetrievalAdapter):
        name = "_id_adapter_for_equiv"

        def render_prompt(self, example, *, condition):
            return example.prompt  # condition does not matter ⇒ identical outputs

    def caller(*, prompt, model, max_tokens, system="", seed=None):
        i = int(prompt.rsplit(" ", 1)[-1].rstrip("?"))
        return ModelOutput(text=f"answer-{i}", input_tokens=1, output_tokens=1)

    runner = MultiSeedRunner(adapter=_IdAdapter(), model_caller=caller)
    spec = HypothesisSpec(
        hypothesis_id="HX_equiv",
        description="conditions yield equivalent accuracy within 0.01",
        adapter_name="_id_adapter_for_equiv",
        treatment_condition="t",
        baseline_condition="b",
        direction=TargetDirection.EQUIV,
        delta=0.01,
        n_examples=10,
        seeds=(0,),
        models=("m",),
    )
    outcome = runner.run_hypothesis(spec)
    assert outcome.delta_observed == 0.0
    assert outcome.target_met is True


def test_runner_records_token_usage_and_latency():
    runner = _make_runner()
    run = runner.run_condition(condition="treatment", model="m", seed=0, n_examples=5)
    assert run.total_input_tokens == 50
    assert run.total_output_tokens == 25
    assert all(r.latency_ms >= 0.0 for r in run.results)


def test_runner_swallows_caller_exceptions_and_records_error():
    class _FlakyAdapter(_FactRetrievalAdapter):
        name = "_flaky"

    def boom(**kwargs):  # noqa: ARG001
        raise RuntimeError("boom")

    runner = MultiSeedRunner(adapter=_FlakyAdapter(), model_caller=boom)
    run = runner.run_condition(condition="x", model="m", seed=0, n_examples=2)
    assert run.n == 2
    assert all(r.error == "boom" for r in run.results)
    assert all(r.correct is False for r in run.results)


def test_registry_register_and_lookup_roundtrip():
    @register
    class _Demo(BenchmarkAdapter):
        name = "_demo_for_registry_test"

        def load_examples(self, *, n=None, seed=0):
            return []

        def render_prompt(self, example, *, condition):
            return ""

        def score(self, example, output):
            return 0.0, False, ""

    assert get("_demo_for_registry_test") is _Demo
    assert "_demo_for_registry_test" in all_names()
