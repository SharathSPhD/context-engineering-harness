"""P6-A entry point — multi-seed × multi-model re-runs of H1 & H2.

Usage:
    # default: free, deterministic, runs in seconds, writes JSON artifacts
    uv run --active python -m experiments.v2.p6a.run

    # only H1 (RULER)
    uv run --active python -m experiments.v2.p6a.run --hypotheses H1

    # cap the example count (smoke runs)
    uv run --active python -m experiments.v2.p6a.run --n-examples 5

    # *real* CLI calls through CLIBudgetScheduler — costs $$:
    uv run --active python -m experiments.v2.p6a.run --live

The runner emits, under `experiments/results/p6a/`, one JSON file per
HypothesisSpec describing:
    - the spec we ran (full echo)
    - the per-(model, seed, condition) BenchmarkRun summary
    - the rolled-up HypothesisOutcome (delta, CI, p, Cohen's d, target_met)
    - run metadata (mode, scheduler status, elapsed wallclock)

These files are the inputs for P7 (statistical analysis & figures).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmarks.registry import get as get_adapter
from src.benchmarks.runner import MultiSeedRunner, RunnerConfig
from src.benchmarks.adapters import (  # noqa: F401  (registers all adapters)
    hallu,
    longctx,
    swebench,
)

from .callers import LiveCLICaller, MockHarnessCaller
from .specs import DEFAULT_MODELS, DEFAULT_SEEDS, all_specs, h1_specs, h2_specs

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results" / "p6a"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_caller(mode: str):
    if mode == "mock":
        return MockHarnessCaller(), None
    if mode == "live":
        # Deferred import: only paid when the operator opts in.
        from tools.dev.scheduler import CLIBudgetScheduler, SchedulerConfig

        sched = CLIBudgetScheduler(
            SchedulerConfig(
                cache_root=".cache/llm",
                ledger_path=".cache/cost_ledger.db",
                journal_path="tools/dev/orchestration/attractor_journal.jsonl",
                max_input_tokens_per_window=2_000_000,
            )
        )
        return LiveCLICaller(sched), sched
    raise ValueError(f"unknown mode {mode!r}")


def _select_specs(which: list[str], models: tuple[str, ...], seeds: tuple[int, ...]):
    if not which or which == ["all"]:
        return all_specs(models=models, seeds=seeds)
    out = []
    if "H1" in which:
        out.extend(h1_specs(models=models, seeds=seeds))
    if "H2" in which:
        out.extend(h2_specs(models=models, seeds=seeds))
    if not out:
        raise ValueError(
            f"no specs selected by {which!r}; valid: ['H1', 'H2', 'all']"
        )
    return out


def _run_one_bundle(
    bundle,
    *,
    caller,
    n_override: int | None,
    runner_cfg: RunnerConfig,
) -> dict[str, Any]:
    spec = bundle.spec
    if n_override is not None:
        spec_n = n_override
    else:
        spec_n = spec.n_examples
    AdapterCls = get_adapter(spec.adapter_name)
    adapter = AdapterCls(**bundle.adapter_kwargs)

    # Patch n_examples in-place per CLI override (HypothesisSpec is frozen by
    # convention but is a plain dataclass; we rebuild rather than mutate).
    from src.benchmarks.hypothesis import HypothesisSpec

    spec_used = HypothesisSpec(
        hypothesis_id=spec.hypothesis_id,
        description=spec.description,
        adapter_name=spec.adapter_name,
        treatment_condition=spec.treatment_condition,
        baseline_condition=spec.baseline_condition,
        metric=spec.metric,
        direction=spec.direction,
        delta=spec.delta,
        n_examples=spec_n,
        seeds=spec.seeds,
        models=spec.models,
        significance_alpha=spec.significance_alpha,
        notes=spec.notes,
    )

    runner = MultiSeedRunner(adapter=adapter, model_caller=caller, config=runner_cfg)
    t0 = time.perf_counter()
    outcome = runner.run_hypothesis(spec_used)
    elapsed = time.perf_counter() - t0

    # Re-emit per-condition runs alongside the outcome for downstream stats.
    treatment_runs = []
    baseline_runs = []
    for model in spec_used.models:
        for seed in spec_used.seeds:
            t = runner.run_condition(
                condition=spec_used.treatment_condition,
                model=model,
                seed=seed,
                n_examples=spec_used.n_examples,
            )
            b = runner.run_condition(
                condition=spec_used.baseline_condition,
                model=model,
                seed=seed,
                n_examples=spec_used.n_examples,
            )
            treatment_runs.append(_summarise_run(t))
            baseline_runs.append(_summarise_run(b))

    return {
        "label": bundle.label,
        "spec": _asdict_spec(spec_used),
        "outcome": _asdict_outcome(outcome),
        "treatment_runs": treatment_runs,
        "baseline_runs": baseline_runs,
        "wallclock_s": round(elapsed, 3),
        "ts": _utcnow_iso(),
    }


def _summarise_run(run) -> dict[str, Any]:
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


def _asdict_spec(spec) -> dict[str, Any]:
    d = asdict(spec)
    d["direction"] = spec.direction.value if hasattr(spec.direction, "value") else str(spec.direction)
    return d


def _asdict_outcome(outcome) -> dict[str, Any]:
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


def _write(payload: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = payload["label"]
    fname = f"{label}.json"
    path = out_dir / fname
    path.write_text(json.dumps(payload, indent=2))
    return path


def _write_summary(all_payloads: list[dict[str, Any]], out_dir: Path, *, meta: dict) -> Path:
    summary = {
        "meta": meta,
        "results": [
            {
                "label": p["label"],
                "hypothesis_id": p["spec"]["hypothesis_id"],
                "adapter": p["spec"]["adapter_name"],
                "models": p["spec"]["models"],
                "seeds": p["spec"]["seeds"],
                "n_examples": p["spec"]["n_examples"],
                "outcome": p["outcome"],
                "wallclock_s": p["wallclock_s"],
            }
            for p in all_payloads
        ],
        "ts": _utcnow_iso(),
    }
    path = out_dir / "_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("mock", "live"),
        default="mock",
        help="mock = deterministic CI runner; live = real claude CLI via scheduler",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="alias for --mode live",
    )
    p.add_argument(
        "--hypotheses",
        nargs="+",
        default=["all"],
        choices=["H1", "H2", "all"],
        help="which hypothesis families to re-run",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="model names (e.g. claude-haiku-4-5 claude-sonnet-4-6)",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="random seeds (default 0 1 2)",
    )
    p.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="override per-spec n_examples (useful for smoke runs)",
    )
    p.add_argument(
        "--bootstrap-n",
        type=int,
        default=2_000,
        help="bootstrap samples for CI (mock mode is happy at 2000)",
    )
    p.add_argument(
        "--permutation-n",
        type=int,
        default=2_000,
        help="permutations for paired test",
    )
    p.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR),
        help="output directory for JSON artifacts",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    mode = "live" if args.live else args.mode
    caller, scheduler = _build_caller(mode)

    runner_cfg = RunnerConfig(
        max_tokens=1024,
        bootstrap_n=args.bootstrap_n,
        permutation_n=args.permutation_n,
    )

    bundles = _select_specs(
        args.hypotheses, models=tuple(args.models), seeds=tuple(args.seeds)
    )
    out_dir = Path(args.out_dir).resolve()
    logger.info(
        "p6a runner starting: mode=%s, %d specs, models=%s, seeds=%s, out=%s",
        mode, len(bundles), args.models, args.seeds, out_dir,
    )

    payloads: list[dict[str, Any]] = []
    for b in bundles:
        logger.info("running %s ...", b.label)
        payload = _run_one_bundle(
            b,
            caller=caller,
            n_override=args.n_examples,
            runner_cfg=runner_cfg,
        )
        path = _write(payload, out_dir)
        logger.info(
            "  -> %s   delta=%.4f  p=%.4f  d=%.3f  target_met=%s",
            path.name,
            payload["outcome"]["delta_observed"],
            payload["outcome"]["p_value"],
            payload["outcome"]["cohens_d"],
            payload["outcome"]["target_met"],
        )
        payloads.append(payload)

    meta = {
        "mode": mode,
        "scheduler_status": (scheduler.status() if scheduler is not None else None),
        "models": list(args.models),
        "seeds": list(args.seeds),
        "n_examples_override": args.n_examples,
        "bootstrap_n": args.bootstrap_n,
        "permutation_n": args.permutation_n,
    }
    summary = _write_summary(payloads, out_dir, meta=meta)
    logger.info("wrote summary to %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
