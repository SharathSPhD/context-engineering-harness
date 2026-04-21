"""Per-example JSONL checkpointed runner for live-HF benchmark re-runs.

Unlike :class:`src.benchmarks.runner.MultiSeedRunner`, this runner owns the
(adapter, model, seed, condition, example_id) loop explicitly so it can:

* Append one JSONL line per completed call to a checkpoint file.
* Skip any (model, seed, condition, example_id) already present in the
  checkpoint when restarted, so a mid-run quota kill does NOT discard
  partial results.
* Catch :class:`tools.dev.scheduler.QuotaExhausted` at the inner call
  site, flush a final partial-results JSON, and exit cleanly.

The final JSON payload schema matches ``experiments/v2/p6a/run._run_one_bundle``
so downstream summary aggregation stays unchanged.

This module is deliberately decoupled from ``experiments/v2/p6a`` because the
same wiring is reused by Phase 2 (hallucination) and Phase 3 (SWE-bench) in
the live-HF re-run plan.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkExample,
    BenchmarkResult,
    BenchmarkRun,
    ModelCaller,
    ModelOutput,
)
from src.benchmarks.hypothesis import HypothesisOutcome, HypothesisSpec, TargetDirection
from src.benchmarks.runner import RunnerConfig, _pair_runs, _verdict
from src.benchmarks.stats import bootstrap_ci, cohens_d, paired_permutation_test
from tools.dev.scheduler import QuotaExhausted

logger = logging.getLogger(__name__)


class PartialRunExit(RuntimeError):
    """Bubbled up from :meth:`CheckpointedBundleRunner.run_bundle` when the
    Claude quota window is exhausted mid-bundle.

    Callers should catch this, log the reason, and move on to either the
    next bundle or a clean ``sys.exit(0)`` — the runner has already
    written a ``*_partial.json`` receipt to disk.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        partial_path: Path,
        completed_keys: int,
        window_summary: dict,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.partial_path = partial_path
        self.completed_keys = completed_keys
        self.window_summary = window_summary


@dataclass(frozen=True)
class CheckpointRecord:
    """One completed (adapter, model, seed, condition, example_id) row."""

    adapter: str
    model: str
    seed: int
    condition: str
    example_id: str
    score: float
    correct: bool
    prediction: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ts: str
    git_sha: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, int, str, str]:
        return (self.adapter, self.model, self.seed, self.condition, self.example_id)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def load_checkpoint(path: Path) -> dict[tuple[str, str, int, str, str], CheckpointRecord]:
    """Read every line of the checkpoint JSONL into a replayable dict.

    Malformed lines are logged and skipped so a partially-written line
    from a SIGKILL does not block the resume.
    """
    if not path.exists():
        return {}
    out: dict[tuple[str, str, int, str, str], CheckpointRecord] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for i, raw in enumerate(fh):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "checkpoint line %d at %s malformed (%s); skipping",
                    i,
                    path,
                    exc,
                )
                continue
            try:
                rec = CheckpointRecord(
                    adapter=obj["adapter"],
                    model=obj["model"],
                    seed=int(obj["seed"]),
                    condition=obj["condition"],
                    example_id=obj["example_id"],
                    score=float(obj.get("score", 0.0)),
                    correct=bool(obj.get("correct", False)),
                    prediction=str(obj.get("prediction", "")),
                    input_tokens=int(obj.get("input_tokens", 0)),
                    output_tokens=int(obj.get("output_tokens", 0)),
                    latency_ms=float(obj.get("latency_ms", 0.0)),
                    ts=str(obj.get("ts", "")),
                    git_sha=str(obj.get("git_sha", "unknown")),
                    error=str(obj.get("error", "")),
                    metadata=dict(obj.get("metadata", {})),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "checkpoint line %d at %s missing fields (%s); skipping",
                    i,
                    path,
                    exc,
                )
                continue
            out[rec.key()] = rec
    return out


def _append_checkpoint(path: Path, rec: CheckpointRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec), ensure_ascii=False, default=str))
        fh.write("\n")


@dataclass
class CheckpointedBundleRunner:
    """Run one bundle (one adapter × cross of models×seeds×conditions) with resume.

    The public contract is one method, :meth:`run_bundle`, which returns a
    JSON-serialisable payload matching the ``experiments/v2/p6a/run._run_one_bundle``
    schema plus a ``provenance`` block and a ``status`` field of
    ``"complete"`` or ``"partial_quota"``.

    This is cheap to instantiate; the caller typically keeps one instance
    per phase-level runner script.
    """

    label: str
    adapter: BenchmarkAdapter
    caller: ModelCaller
    spec: HypothesisSpec
    checkpoint_path: Path
    runner_cfg: RunnerConfig
    resume: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._checkpoint: dict[tuple[str, str, int, str, str], CheckpointRecord] = (
            load_checkpoint(self.checkpoint_path) if self.resume else {}
        )
        self._git_sha: str = _git_sha()
        if self._checkpoint:
            logger.info(
                "checkpointed-runner resuming: %d existing records at %s",
                len(self._checkpoint),
                self.checkpoint_path,
            )

    # ----------------------------------------------------------------- public

    def run_bundle(self) -> dict[str, Any]:
        """Run the full cross of (model, seed, condition, example); return payload.

        Raises :class:`PartialRunExit` if the quota exhausts mid-bundle.
        On success, returns the complete payload dict.
        """
        spec = self.spec
        adapter = self.adapter
        t0 = time.perf_counter()

        treatment_runs: list[BenchmarkRun] = []
        baseline_runs: list[BenchmarkRun] = []
        conditions = (spec.treatment_condition, spec.baseline_condition)
        try:
            for model in spec.models:
                for seed in spec.seeds:
                    examples = adapter.load_examples(n=spec.n_examples, seed=seed)
                    per_condition_runs: dict[str, BenchmarkRun] = {}
                    for condition in conditions:
                        run = self._run_one_condition(
                            model=model,
                            seed=seed,
                            condition=condition,
                            examples=examples,
                        )
                        per_condition_runs[condition] = run
                    treatment_runs.append(per_condition_runs[spec.treatment_condition])
                    baseline_runs.append(per_condition_runs[spec.baseline_condition])
            status = "complete"
        except QuotaExhausted as qe:
            elapsed = time.perf_counter() - t0
            partial_payload = self._build_payload(
                status="partial_quota",
                treatment_runs=treatment_runs,
                baseline_runs=baseline_runs,
                elapsed_s=elapsed,
                quota_note=qe.reason,
                window_summary=qe.window_summary,
            )
            partial_path = self._write_partial(partial_payload)
            completed_keys = len(self._checkpoint)
            raise PartialRunExit(
                f"quota exhausted during {self.label}: {qe.reason}",
                reason=qe.reason,
                partial_path=partial_path,
                completed_keys=completed_keys,
                window_summary=qe.window_summary,
            ) from qe

        elapsed = time.perf_counter() - t0
        payload = self._build_payload(
            status=status,
            treatment_runs=treatment_runs,
            baseline_runs=baseline_runs,
            elapsed_s=elapsed,
        )
        return payload

    # ----------------------------------------------------------------- inner loop

    def _run_one_condition(
        self,
        *,
        model: str,
        seed: int,
        condition: str,
        examples: list[BenchmarkExample],
    ) -> BenchmarkRun:
        adapter = self.adapter
        results: list[BenchmarkResult] = []
        system = adapter.system_prompt(condition=condition)
        for ex in examples:
            key = (adapter.name, model, seed, condition, ex.id)
            if key in self._checkpoint:
                rec = self._checkpoint[key]
                results.append(self._result_from_record(rec, ex))
                continue

            prompt = adapter.render_prompt(ex, condition=condition)
            t_call = time.perf_counter()
            err = ""
            try:
                output = self.caller(
                    prompt=prompt,
                    model=model,
                    max_tokens=self.runner_cfg.max_tokens,
                    system=system,
                    seed=seed,
                )
            except QuotaExhausted:
                raise
            except Exception as exc:  # noqa: BLE001 — surface, don't crash the bundle
                logger.exception(
                    "caller failed on %s / %s / %s / %s / %s: %s",
                    adapter.name, model, seed, condition, ex.id, exc,
                )
                output = ModelOutput(text="", metadata={"error": str(exc)})
                err = str(exc)
            latency_ms = (time.perf_counter() - t_call) * 1000.0

            score, correct, prediction = adapter.score(ex, output)
            result = BenchmarkResult(
                example_id=ex.id,
                condition=condition,
                seed=seed,
                prediction=prediction,
                score=float(score),
                correct=bool(correct),
                input_tokens=int(output.input_tokens),
                output_tokens=int(output.output_tokens),
                latency_ms=latency_ms,
                error=err,
                metadata=dict(output.metadata),
            )
            results.append(result)

            rec = CheckpointRecord(
                adapter=adapter.name,
                model=model,
                seed=seed,
                condition=condition,
                example_id=ex.id,
                score=result.score,
                correct=result.correct,
                prediction=result.prediction,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                ts=_utcnow_iso(),
                git_sha=self._git_sha,
                error=err,
                metadata=dict(result.metadata),
            )
            _append_checkpoint(self.checkpoint_path, rec)
            self._checkpoint[rec.key()] = rec

        return BenchmarkRun(
            adapter_name=adapter.name,
            model=model,
            condition=condition,
            seed=seed,
            results=results,
        )

    @staticmethod
    def _result_from_record(
        rec: CheckpointRecord, ex: BenchmarkExample
    ) -> BenchmarkResult:
        return BenchmarkResult(
            example_id=ex.id,
            condition=rec.condition,
            seed=rec.seed,
            prediction=rec.prediction,
            score=rec.score,
            correct=rec.correct,
            input_tokens=rec.input_tokens,
            output_tokens=rec.output_tokens,
            latency_ms=rec.latency_ms,
            error=rec.error,
            metadata=dict(rec.metadata),
        )

    # ----------------------------------------------------------------- stats + payload

    def _compute_outcome(
        self,
        treatment_runs: list[BenchmarkRun],
        baseline_runs: list[BenchmarkRun],
    ) -> HypothesisOutcome | None:
        spec = self.spec
        if not treatment_runs or not baseline_runs:
            return None
        paired_t, paired_b = _pair_runs(treatment_runs, baseline_runs, spec.metric)
        if not paired_t:
            return None
        treatment_metric = float(sum(paired_t) / len(paired_t))
        baseline_metric = float(sum(paired_b) / len(paired_b))
        delta_observed = treatment_metric - baseline_metric
        diffs = [t - b for t, b in zip(paired_t, paired_b)]
        _, ci_low, ci_high = bootstrap_ci(
            diffs,
            ci=self.runner_cfg.bootstrap_ci_level,
            n_bootstrap=self.runner_cfg.bootstrap_n,
        )
        p_value = paired_permutation_test(
            paired_t, paired_b, n_permutations=self.runner_cfg.permutation_n
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

    def _build_payload(
        self,
        *,
        status: str,
        treatment_runs: list[BenchmarkRun],
        baseline_runs: list[BenchmarkRun],
        elapsed_s: float,
        quota_note: str = "",
        window_summary: dict | None = None,
    ) -> dict[str, Any]:
        outcome = self._compute_outcome(treatment_runs, baseline_runs)
        payload: dict[str, Any] = {
            "label": self.label,
            "status": status,
            "spec": _asdict_spec(self.spec),
            "outcome": _asdict_outcome(outcome) if outcome is not None else None,
            "treatment_runs": [_summarise_run(r) for r in treatment_runs],
            "baseline_runs": [_summarise_run(r) for r in baseline_runs],
            "wallclock_s": round(elapsed_s, 3),
            "ts": _utcnow_iso(),
            "provenance": dict(self.provenance),
            "metadata": {
                **self.extra_metadata,
                "git_sha": self._git_sha,
                "checkpoint_path": str(self.checkpoint_path),
                "checkpointed_records": len(self._checkpoint),
            },
        }
        if status == "partial_quota":
            payload["partial"] = {
                "reason": quota_note,
                "window_summary": window_summary or {},
            }
        return payload

    def _write_partial(self, payload: dict[str, Any]) -> Path:
        out_dir = self.checkpoint_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.label}_partial.json"
        path.write_text(json.dumps(payload, indent=2))
        return path


# ---- schema helpers mirroring experiments/v2/p6a/run.py for compatibility ----


def _summarise_run(run: BenchmarkRun) -> dict[str, Any]:
    return {
        "adapter": run.adapter_name,
        "model": run.model,
        "condition": run.condition,
        "seed": run.seed,
        "n": run.n,
        "mean_score": round(run.mean_score, 4),
        "accuracy": round(run.accuracy, 4),
        "total_input_tokens": run.total_input_tokens,
        "total_output_tokens": run.total_output_tokens,
    }


def _asdict_spec(spec: HypothesisSpec) -> dict[str, Any]:
    d = asdict(spec)
    direction = spec.direction
    d["direction"] = (
        direction.value if hasattr(direction, "value") else str(direction)
    )
    return d


def _asdict_outcome(outcome: HypothesisOutcome) -> dict[str, Any]:
    return {
        "treatment_metric": round(outcome.treatment_metric, 4),
        "baseline_metric": round(outcome.baseline_metric, 4),
        "delta_observed": round(outcome.delta_observed, 4),
        "ci_low": round(outcome.ci_low, 4),
        "ci_high": round(outcome.ci_high, 4),
        "p_value": round(outcome.p_value, 6),
        "cohens_d": round(outcome.cohens_d, 4),
        "target_met": bool(outcome.target_met),
        "n_examples_used": outcome.n_examples_used,
        "n_seeds_used": outcome.n_seeds_used,
        "extra": outcome.extra,
    }
