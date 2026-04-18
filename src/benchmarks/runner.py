"""MultiSeedRunner — executes a hypothesis end-to-end.

For each (model, seed):
  1. Load N examples from the adapter (deterministic per seed).
  2. Run treatment condition on each example via the injected ModelCaller.
  3. Run baseline condition on the same examples.
  4. Score both outputs through `adapter.score(...)`.
  5. Aggregate per-example paired score vectors.

Then across all seeds:
  6. Pool per-example paired scores.
  7. Compute bootstrap CI on the difference, paired permutation p-value, and
     paired Cohen's d.
  8. Emit a `HypothesisOutcome` with `target_met` decided by the spec.

The runner is fully deterministic given a seed and a deterministic
ModelCaller (for tests we use a fixture caller; for real runs the dev tree
wraps the CLIBudgetScheduler, which has its own retry/cache layer).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .base import BenchmarkAdapter, BenchmarkResult, BenchmarkRun, ModelCaller
from .hypothesis import HypothesisOutcome, HypothesisSpec, TargetDirection
from .stats import bootstrap_ci, cohens_d, paired_permutation_test

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    max_tokens: int = 1024
    bootstrap_n: int = 10_000
    permutation_n: int = 10_000
    bootstrap_ci_level: float = 0.95
    request_timeout_s: int = 300


class MultiSeedRunner:
    """Owns the seed loop and the stats roll-up. Adapter does the I/O & scoring."""

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        model_caller: ModelCaller,
        *,
        config: RunnerConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.model_caller = model_caller
        self.config = config or RunnerConfig()

    def run_condition(
        self,
        *,
        condition: str,
        model: str,
        seed: int,
        n_examples: int | None,
    ) -> BenchmarkRun:
        examples = self.adapter.load_examples(n=n_examples, seed=seed)
        results: list[BenchmarkResult] = []
        system = self.adapter.system_prompt(condition=condition)
        for ex in examples:
            prompt = self.adapter.render_prompt(ex, condition=condition)
            t0 = time.perf_counter()
            try:
                output = self.model_caller(
                    prompt=prompt,
                    model=model,
                    max_tokens=self.config.max_tokens,
                    system=system,
                    seed=seed,
                )
                err = ""
            except Exception as exc:  # noqa: BLE001 — adapter must surface, not crash the run
                logger.exception("adapter %s example %s failed: %s", self.adapter.name, ex.id, exc)
                from .base import ModelOutput

                output = ModelOutput(text="", metadata={"error": str(exc)})
                err = str(exc)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            score, correct, prediction = self.adapter.score(ex, output)
            results.append(
                BenchmarkResult(
                    example_id=ex.id,
                    condition=condition,
                    seed=seed,
                    prediction=prediction,
                    score=float(score),
                    correct=bool(correct),
                    input_tokens=int(output.input_tokens),
                    output_tokens=int(output.output_tokens),
                    latency_ms=elapsed_ms,
                    error=err,
                    metadata=dict(output.metadata),
                )
            )
        return BenchmarkRun(
            adapter_name=self.adapter.name,
            model=model,
            condition=condition,
            seed=seed,
            results=results,
        )

    def run_hypothesis(self, spec: HypothesisSpec) -> HypothesisOutcome:
        treatment_runs: list[BenchmarkRun] = []
        baseline_runs: list[BenchmarkRun] = []
        for model in spec.models:
            for seed in spec.seeds:
                t = self.run_condition(
                    condition=spec.treatment_condition,
                    model=model,
                    seed=seed,
                    n_examples=spec.n_examples,
                )
                b = self.run_condition(
                    condition=spec.baseline_condition,
                    model=model,
                    seed=seed,
                    n_examples=spec.n_examples,
                )
                treatment_runs.append(t)
                baseline_runs.append(b)

        paired_t, paired_b = _pair_runs(treatment_runs, baseline_runs, spec.metric)
        treatment_metric = float(sum(paired_t) / len(paired_t)) if paired_t else 0.0
        baseline_metric = float(sum(paired_b) / len(paired_b)) if paired_b else 0.0
        delta_observed = treatment_metric - baseline_metric

        diffs = [t - b for t, b in zip(paired_t, paired_b)]
        _, ci_low, ci_high = bootstrap_ci(
            diffs,
            ci=self.config.bootstrap_ci_level,
            n_bootstrap=self.config.bootstrap_n,
        )
        p_value = paired_permutation_test(
            paired_t, paired_b, n_permutations=self.config.permutation_n
        )
        d = cohens_d(paired_t, paired_b)

        target_met = _verdict(spec, delta_observed, p_value)
        return HypothesisOutcome(
            spec=spec,
            treatment_metric=treatment_metric,
            baseline_metric=baseline_metric,
            delta_observed=delta_observed,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            cohens_d=d,
            target_met=target_met,
            n_examples_used=len(paired_t),
            n_seeds_used=len(set(r.seed for r in treatment_runs)),
            extra={
                "treatment_runs": len(treatment_runs),
                "baseline_runs": len(baseline_runs),
                "models": list(spec.models),
            },
        )


def _pair_runs(
    treatment_runs: list[BenchmarkRun],
    baseline_runs: list[BenchmarkRun],
    metric: str,
) -> tuple[list[float], list[float]]:
    """Align paired per-example scores by (example_id, seed, model).

    Examples without a matching baseline are dropped (and logged); this can
    happen if the adapter is non-deterministic or the model emits an error
    only on one side.
    """
    if metric not in ("accuracy", "score"):
        raise ValueError(f"unsupported metric {metric!r}; use 'accuracy' or 'score'")

    def _index(runs: list[BenchmarkRun]) -> dict[tuple[str, int, str], BenchmarkResult]:
        idx: dict[tuple[str, int, str], BenchmarkResult] = {}
        for run in runs:
            for r in run.results:
                idx[(r.example_id, r.seed, run.model)] = r
        return idx

    t_idx = _index(treatment_runs)
    b_idx = _index(baseline_runs)
    paired_t: list[float] = []
    paired_b: list[float] = []
    for key, r_t in t_idx.items():
        r_b = b_idx.get(key)
        if r_b is None:
            logger.warning("dropping unpaired treatment example %s", key)
            continue
        if metric == "accuracy":
            paired_t.append(1.0 if r_t.correct else 0.0)
            paired_b.append(1.0 if r_b.correct else 0.0)
        else:
            paired_t.append(r_t.score)
            paired_b.append(r_b.score)
    return paired_t, paired_b


def _verdict(spec: HypothesisSpec, delta: float, p_value: float) -> bool:
    significant = p_value < spec.significance_alpha
    if spec.direction == TargetDirection.GREATER:
        return significant and delta >= spec.delta
    if spec.direction == TargetDirection.LESS:
        return significant and delta <= -spec.delta
    if spec.direction == TargetDirection.EQUIV:
        return abs(delta) <= spec.delta
    return False
