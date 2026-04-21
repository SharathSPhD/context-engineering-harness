"""Live-HF runner for Phase 1 (RULER) and Phase 2 (hallucination) re-runs.

Entry point:
    uv run --active python -m experiments.v2.p6a.run_live_hf --dry-run
    uv run --active python -m experiments.v2.p6a.run_live_hf --scope ruler --live
    uv run --active python -m experiments.v2.p6a.run_live_hf --scope hallu --live
    uv run --active python -m experiments.v2.p6a.run_live_hf --scope all_real_capable --live

Side-by-side artifact policy: every output writes ``*_live.json`` so the
pre-plan synthetic receipts stay byte-identical. A ``provenance`` block is
always attached and records ``{source: "huggingface", hf_dataset_id, ...,
load_real: true, fail_fast_on_quota: true, git_sha, ts}``.

Graceful quota handling: the scheduler is configured with
``fail_fast_on_quota=True``. When the 5-hour Claude window is exhausted
mid-bundle, the runner writes ``{label}_live_partial.json`` (checkpointed
state rolled into a full payload), logs the window summary, and exits with
code 0 so CI / cron don't alarm.

Dry-run mode: loads one example per (adapter, condition, tier), estimates
total input/output tokens and a rough USD cost using the local tokenizer
and documented Anthropic list prices. Zero API calls are made.

Exit codes (live path):
  0  — bundles completed (or checkpointed partial on quota exhaustion).
  1  — unhandled runtime failure; caller should inspect logs.
  2  — pre-registration violation; a CLI arg diverged from the locked
       ``scope=full_battery`` *or* ``scope=core4`` defaults without
       ``--allow-override-preregistration``.
  3  — provenance integrity error; probe example for a live bundle
       returned ``metadata["source"] != "huggingface"`` under
       ``load_real=True``. Refusing to mislabel synthetic data as HF.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmarks.adapters import (  # noqa: F401  — registers all adapters
    hallu,
    longctx,
    swebench,
)
from src.benchmarks.base import BenchmarkAdapter
from src.benchmarks.registry import get as get_adapter
from src.benchmarks.runner import RunnerConfig
from src.utils.tokenizer import count_tokens
from tools.dev.runners import CheckpointedBundleRunner, PartialRunExit

from .callers import LiveCLICaller, MockHarnessCaller
from .specs import DEFAULT_MODELS
from .specs_live import (
    HF_IDS,
    LIVE_DEFAULT_HALLU_N,
    LIVE_DEFAULT_RULER_N,
    LIVE_DEFAULT_SEEDS,
    LIVE_DEFAULT_SWEB_N,
    LIVE_DEFAULT_TIERS,
    LIVE_EXT_HALLU_N,
    LIVE_EXT_RULER_N,
    LIVE_EXT_RULER_TIERS,
    LIVE_EXT_SWEB_TIMEOUT_S,
    P6ASpecBundle,
    core4_specs,
    full_battery_specs,
    h1_ruler_live_specs,
    h1b_ruler_multi_live_specs,
    h_swebench_verified_live_n15_spec,
    hallu_live_specs,
    power_ext_specs,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
P6A_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "p6a"
CHECKPOINT_ROOT = REPO_ROOT / ".cache" / "live_hf_checkpoints"

# Rough Anthropic list prices in USD/token as of this plan's authoring.
# These are order-of-magnitude cost signals, not invoice-grade accounting.
# Cache-read is tracked separately by the scheduler; here we apply the
# headline input rate because dry-run assumes cold cache (worst case).
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Haiku 4.5 list prices
    "claude-haiku-4-5": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    # Sonnet 4.x list prices
    "claude-sonnet-4-6": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "claude-sonnet-4-5": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
}
FALLBACK_PRICING = {"input_per_mtok": 3.00, "output_per_mtok": 15.00}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _provenance_for(
    bundle: P6ASpecBundle, *, example_source: str | None = None
) -> dict[str, Any]:
    """Build the provenance block for a bundle receipt.

    ``example_source`` is pulled from the first live-loaded example's
    ``metadata["source"]`` (e.g. ``"huggingface"``, ``"synthetic"``,
    ``"synthetic+real_corpus"``). When set, it overrides the spec's
    claim so a silent synthetic fallback cannot produce a receipt that
    claims ``source="huggingface"`` while actually having served
    synthetic rows. When ``None`` (pre-load call-site), we default to
    ``"huggingface"`` because the bundle's strict_hf guard would have
    raised if HF had failed.
    """
    adapter_name = bundle.spec.adapter_name
    hf_info = HF_IDS.get(adapter_name, {})
    return {
        "source": example_source or "huggingface",
        "adapter_name": adapter_name,
        "adapter_kwargs": dict(bundle.adapter_kwargs),
        "hf_dataset_id": hf_info.get("dataset", ""),
        "hf_config": hf_info.get("config", ""),
        "hf_task": hf_info.get("task", ""),
        "load_real": bool(bundle.adapter_kwargs.get("load_real", False)),
        "strict_hf": bool(bundle.adapter_kwargs.get("strict_hf", False)),
        "git_sha": _git_sha(),
        "ts": _utcnow_iso(),
    }


class PreregistrationViolation(RuntimeError):
    """Raised when CLI args try to override a locked pre-registered scope."""


class ProvenanceIntegrityError(RuntimeError):
    """Raised when a bundle's first-example source is not ``huggingface``.

    This is the live-HF pipeline's "fail closed" signal: if the probe
    returns anything other than a genuine HF row under
    ``load_real=True``, we refuse to write a receipt that would
    silently mislabel synthetic data as HuggingFace-sourced.
    """


_PREREG_LOCKED_SCOPES = ("full_battery", "core4", "power_ext")


def _preregistration_locks(scope: str) -> dict[str, Any]:
    """Return the pre-registered values locked for each scope.

    * ``full_battery`` / ``core4`` — v2.1 locks: RULER N=15, hallu
      N=15, sweb N=15, seeds=(0,1), tiers=(8192, 16384), two models.
    * ``power_ext`` — v2.1.1 amendment: RULER N=30 (only 16K tier),
      hallu N=30, sweb N=15, scheduler_timeout_s=900. Same seeds and
      models. The amendment is pre-execution (see Appendix~G) and
      must stay locked so the addendum stays reproducible.
    """
    if scope == "power_ext":
        return {
            "ruler_n": LIVE_EXT_RULER_N,
            "hallu_n": LIVE_EXT_HALLU_N,
            "sweb_n": LIVE_DEFAULT_SWEB_N,
            "seeds": LIVE_DEFAULT_SEEDS,
            "tiers": LIVE_EXT_RULER_TIERS,
            "models": tuple(DEFAULT_MODELS),
            "scheduler_timeout_s": LIVE_EXT_SWEB_TIMEOUT_S,
        }
    # full_battery + core4 share the same v2.1 locks.
    return {
        "ruler_n": LIVE_DEFAULT_RULER_N,
        "hallu_n": LIVE_DEFAULT_HALLU_N,
        "sweb_n": LIVE_DEFAULT_SWEB_N,
        "seeds": LIVE_DEFAULT_SEEDS,
        "tiers": LIVE_DEFAULT_TIERS,
        "models": tuple(DEFAULT_MODELS),
        # v2.1 used the scheduler default of 300 s. Not part of the
        # v2.1 lock, so we skip it for full_battery/core4.
        "scheduler_timeout_s": None,
    }


def _enforce_preregistration(
    scope: str,
    *,
    args_ruler_n: int,
    args_hallu_n: int,
    args_sweb_n: int,
    args_seeds: tuple[int, ...],
    args_tiers: tuple[int, ...],
    args_models: tuple[str, ...],
    args_scheduler_timeout_s: int,
    allow_override: bool,
) -> None:
    """Reject CLI overrides of any locked pre-registered scope.

    Three scopes are registered:

    * ``full_battery`` (v2.1) — N=15 per RULER / hallu / sweb bundle,
      seeds=(0, 1), tiers=(8192, 16384), models=(haiku, sonnet).
    * ``core4`` (v2.1) — same locks as full_battery; smaller bundle
      count, unchanged per-bundle parameters.
    * ``power_ext`` (v2.1.1) — N=30 for RULER 16K and TQA, N=15 for
      SWE-bench, seeds=(0, 1), tiers=(16384,), models unchanged,
      ``scheduler_timeout_s=900``.

    Any CLI deviation turns the receipt into a new (unregistered)
    protocol; we fail loudly unless the operator passes
    ``--allow-override-preregistration`` and therefore accepts that
    the receipt is *not* the locked run.
    """
    if scope not in _PREREG_LOCKED_SCOPES or allow_override:
        return
    lock = _preregistration_locks(scope)
    mismatches: list[str] = []
    if args_ruler_n != lock["ruler_n"]:
        mismatches.append(
            f"--ruler-n={args_ruler_n} != locked {lock['ruler_n']}"
        )
    if args_hallu_n != lock["hallu_n"]:
        mismatches.append(
            f"--hallu-n={args_hallu_n} != locked {lock['hallu_n']}"
        )
    if args_sweb_n != lock["sweb_n"]:
        mismatches.append(
            f"--sweb-n={args_sweb_n} != locked {lock['sweb_n']}"
        )
    if args_seeds != lock["seeds"]:
        mismatches.append(
            f"--seeds={list(args_seeds)} != locked {list(lock['seeds'])}"
        )
    if args_tiers != lock["tiers"]:
        mismatches.append(
            f"--tiers={list(args_tiers)} != locked {list(lock['tiers'])}"
        )
    if args_models != lock["models"]:
        mismatches.append(
            f"--models={list(args_models)} != locked {list(lock['models'])}"
        )
    expected_timeout = lock["scheduler_timeout_s"]
    if expected_timeout is not None and args_scheduler_timeout_s != expected_timeout:
        mismatches.append(
            f"--scheduler-timeout-s={args_scheduler_timeout_s} != locked "
            f"{expected_timeout}"
        )
    if mismatches:
        raise PreregistrationViolation(
            f"scope={scope} is pre-registered and cannot take CLI "
            "overrides without --allow-override-preregistration.\n"
            "  " + "\n  ".join(mismatches)
        )


# --- scope ------------------------------------------------------------


def _select_bundles(
    scope: str,
    *,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    ruler_n: int,
    hallu_n: int,
    sweb_n: int,
    tiers: tuple[int, ...],
) -> list[P6ASpecBundle]:
    ruler = h1_ruler_live_specs(
        models=models, seeds=seeds, n_examples=ruler_n, target_tokens_tiers=tiers
    )
    ruler_multi = h1b_ruler_multi_live_specs(
        models=models, seeds=seeds, n_examples=ruler_n, target_tokens_tiers=tiers
    )
    hallu = hallu_live_specs(models=models, seeds=seeds, n_examples=hallu_n)
    sweb = [
        h_swebench_verified_live_n15_spec(
            models=models, seeds=seeds, n_examples=sweb_n
        )
    ]
    if scope == "ruler":
        return ruler + ruler_multi
    if scope == "hallu":
        return hallu
    if scope == "swebench":
        return sweb
    if scope == "all_real_capable":
        return ruler + ruler_multi + hallu
    if scope == "core4":
        return core4_specs(
            models=models,
            seeds=seeds,
            ruler_n=ruler_n,
            hallu_n=hallu_n,
            sweb_n=sweb_n,
            target_tokens_tiers=tiers,
        )
    if scope == "power_ext":
        # tiers is ignored here: the ext RULER bundle is locked to
        # 16384 only (the 8K tier already cleared both gates at N=15
        # and re-running it is pure token burn). The pre-registration
        # guard already rejects --tiers overrides for this scope.
        return power_ext_specs(
            models=models,
            seeds=seeds,
            ruler_n=ruler_n,
            hallu_n=hallu_n,
            sweb_n=sweb_n,
        )
    if scope == "full_battery":
        return full_battery_specs(
            models=models,
            seeds=seeds,
            ruler_n=ruler_n,
            hallu_n=hallu_n,
            sweb_n=sweb_n,
            target_tokens_tiers=tiers,
        )
    raise ValueError(f"unknown scope {scope!r}")


# --- dry run ----------------------------------------------------------


@dataclass
class DryRunEstimate:
    label: str
    adapter_name: str
    tier: str
    n_calls_total: int
    avg_input_tokens_per_call: int
    avg_output_tokens_per_call: int
    total_input_tokens: int
    total_output_tokens: int
    usd_by_model: dict[str, float]
    usd_total: float
    sample_prompt_tokens: dict[str, int]


def _estimate_output_tokens(adapter_name: str, max_tokens: int) -> int:
    """Heuristic: how many output tokens will a harness_off/on call *average*?

    RULER: single short code → ~5-20 tokens average (treatment) but the
    scheduler still budgets for ``max_tokens``. Use 30 as a conservative
    average across both conditions.

    Hallu: TruthfulQA/FACTS/HaluEval generate 1-3 sentences on average →
    cap at ``max_tokens`` or 120 tokens, whichever is lower.
    """
    if adapter_name.startswith("ruler"):
        return min(max_tokens, 40)
    if adapter_name in {"truthful_qa", "halu_eval_qa"}:
        return min(max_tokens, 120)
    if adapter_name == "facts_grounding":
        return min(max_tokens, 80)
    if adapter_name == "swe_bench_verified":
        # Patches are longer than QA but bounded by max_tokens; real
        # gold-patches average ~300 tokens in SWE-bench Verified.
        return min(max_tokens, 400)
    return min(max_tokens, 120)


def _dry_run_one_bundle(
    bundle: P6ASpecBundle, *, runner_cfg: RunnerConfig
) -> DryRunEstimate:
    """Load 1 example per condition, measure tokens, multiply out the cross."""
    AdapterCls = get_adapter(bundle.spec.adapter_name)
    adapter: BenchmarkAdapter = AdapterCls(**bundle.adapter_kwargs)
    # Load a single example — cheap, uses a real HF fetch if load_real=True,
    # otherwise uses the synthetic generator at the same tier.
    try:
        examples = adapter.load_examples(n=1, seed=bundle.spec.seeds[0])
    except Exception as exc:  # noqa: BLE001 — surface in receipt, not crash
        logger.warning(
            "dry-run: failed to load example for %s (%s); falling back to synthetic shape",
            bundle.label,
            exc,
        )
        examples = []

    sample_prompt_tokens: dict[str, int] = {}
    if examples:
        ex = examples[0]
        for cond in (bundle.spec.treatment_condition, bundle.spec.baseline_condition):
            p = adapter.render_prompt(ex, condition=cond)
            s = adapter.system_prompt(condition=cond)
            tok = count_tokens((s + "\n" + p) if s else p)
            sample_prompt_tokens[cond] = int(tok)
    else:
        # Fallback: estimate from target_tokens if set, else a cheap default.
        tier = int(bundle.adapter_kwargs.get("target_tokens", 2_000))
        sample_prompt_tokens = {
            bundle.spec.treatment_condition: tier + 80,
            bundle.spec.baseline_condition: tier + 40,
        }

    spec = bundle.spec
    n_calls_per_condition = (
        len(spec.models) * len(spec.seeds) * spec.n_examples
    )
    n_calls_total = 2 * n_calls_per_condition  # treatment + baseline

    avg_input = int(
        (sample_prompt_tokens[spec.treatment_condition]
         + sample_prompt_tokens[spec.baseline_condition]) // 2
    )
    avg_output = _estimate_output_tokens(spec.adapter_name, runner_cfg.max_tokens)
    total_input = avg_input * n_calls_total
    total_output = avg_output * n_calls_total

    usd_by_model: dict[str, float] = {}
    usd_total = 0.0
    calls_per_model = n_calls_total // max(1, len(spec.models))
    in_tok_per_model = avg_input * calls_per_model
    out_tok_per_model = avg_output * calls_per_model
    for m in spec.models:
        price = MODEL_PRICING.get(m, FALLBACK_PRICING)
        cost = (
            (in_tok_per_model / 1_000_000) * price["input_per_mtok"]
            + (out_tok_per_model / 1_000_000) * price["output_per_mtok"]
        )
        usd_by_model[m] = round(cost, 4)
        usd_total += cost

    tier_label = str(bundle.adapter_kwargs.get("target_tokens", "n/a"))
    return DryRunEstimate(
        label=bundle.label,
        adapter_name=spec.adapter_name,
        tier=tier_label,
        n_calls_total=n_calls_total,
        avg_input_tokens_per_call=avg_input,
        avg_output_tokens_per_call=avg_output,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        usd_by_model=usd_by_model,
        usd_total=round(usd_total, 4),
        sample_prompt_tokens=sample_prompt_tokens,
    )


def _print_dry_run_table(estimates: list[DryRunEstimate]) -> None:
    header = (
        f"{'BUNDLE':<32} {'ADAPTER':<20} {'TIER':<8} "
        f"{'CALLS':>6} {'INTOK':>10} {'OUTTOK':>8} {'USD':>10}"
    )
    print(header)
    print("-" * len(header))
    total_in = total_out = 0
    total_usd = 0.0
    for e in estimates:
        print(
            f"{e.label:<32} {e.adapter_name:<20} {e.tier:<8} "
            f"{e.n_calls_total:>6} {e.total_input_tokens:>10,} "
            f"{e.total_output_tokens:>8,} ${e.usd_total:>8.2f}"
        )
        total_in += e.total_input_tokens
        total_out += e.total_output_tokens
        total_usd += e.usd_total
    print("-" * len(header))
    print(
        f"{'TOTAL':<32} {'':<20} {'':<8} "
        f"{sum(e.n_calls_total for e in estimates):>6} "
        f"{total_in:>10,} {total_out:>8,} ${total_usd:>8.2f}"
    )
    print()
    print(
        "[dry-run] Per-model breakdown (rough list-price USD; cache read "
        "not applied):"
    )
    per_model: dict[str, float] = {}
    for e in estimates:
        for m, usd in e.usd_by_model.items():
            per_model[m] = per_model.get(m, 0.0) + usd
    for m, usd in sorted(per_model.items()):
        print(f"    {m:<26} ${usd:>8.2f}")
    print()
    print(
        "[dry-run] NOTE: estimates assume cold cache (no prompt caching "
        "savings) and ignore the scheduler's 2M input-token window cap. "
        "Real costs will be lower if the cache hits."
    )


# --- live run ---------------------------------------------------------


def _build_scheduler(*, fail_fast_on_quota: bool, timeout_s: int = 300):
    from tools.dev.scheduler import CLIBudgetScheduler, SchedulerConfig

    # ``max_input_tokens_per_window`` is a local safety rail, not the
    # Anthropic-side rate limit. The 2M/window default was sized for a
    # metered subscription tier; the full-quota run_live_hf pipeline
    # bumps it to 20M so a warm disk cache plus a >2M-token RULER tier
    # does not artificially stall the pipeline. Real API rate limits
    # are still enforced by the scheduler's HALT regime detector
    # against 429 responses.
    #
    # ``timeout_s`` is the per-CLI-call subprocess timeout (not a
    # scheduler budget). The scheduler's default of 300 s is fine
    # for short RULER/TQA prompts but the Claude CLI's SessionStart
    # hook can take >5 min to complete on SWE-bench-sized inputs;
    # the v2.1.1 power-extension run bumps it to 900 s so those
    # aborts go away.
    return CLIBudgetScheduler(
        SchedulerConfig(
            cache_root=".cache/llm",
            ledger_path=".cache/cost_ledger.db",
            journal_path="tools/dev/orchestration/attractor_journal.jsonl",
            max_input_tokens_per_window=20_000_000,
            fail_fast_on_quota=fail_fast_on_quota,
            max_retries=3,
            timeout_s=timeout_s,
        )
    )


def _run_live(
    bundles: list[P6ASpecBundle],
    *,
    runner_cfg: RunnerConfig,
    out_dir: Path,
    continue_on_partial: bool,
    scheduler_timeout_s: int = 300,
) -> dict[str, Any]:
    # Only require HF_TOKEN to be set for actually calling out; dry-run
    # doesn't need it but load_real=True might, so we surface it early.
    if os.getenv("HF_TOKEN") is None and os.getenv("HUGGING_FACE_HUB_TOKEN") is None:
        logger.warning(
            "neither HF_TOKEN nor HUGGING_FACE_HUB_TOKEN is set; real "
            "dataset loads may fail for gated datasets"
        )

    scheduler = _build_scheduler(
        fail_fast_on_quota=True, timeout_s=scheduler_timeout_s
    )
    caller = LiveCLICaller(scheduler)

    out_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    run_index: dict[str, Any] = {
        "ts_start": _utcnow_iso(),
        "git_sha": _git_sha(),
        "out_dir": str(out_dir),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "bundles": [],
    }

    for bundle in bundles:
        checkpoint_path = CHECKPOINT_ROOT / f"{bundle.label}.jsonl"
        AdapterCls = get_adapter(bundle.spec.adapter_name)
        adapter = AdapterCls(**bundle.adapter_kwargs)

        # Defense-in-depth: load a probe example now and confirm the
        # data genuinely came from HF. strict_hf would already have
        # raised on a failed HF fetch, but this probe also catches the
        # case where the adapter silently serves synthetic rows (e.g.
        # an older adapter missing the strict_hf wiring). Any mismatch
        # here is fatal — we'd rather crash the bundle than publish a
        # receipt with a falsified provenance.source field.
        probe = adapter.load_examples(n=1, seed=bundle.spec.seeds[0])
        probe_source = (
            (probe[0].metadata or {}).get("source") if probe else None
        )
        if bundle.adapter_kwargs.get("load_real") and probe_source != "huggingface":
            raise ProvenanceIntegrityError(
                f"bundle {bundle.label!r}: load_real=True but probe example "
                f"source={probe_source!r} (expected 'huggingface'). Refusing "
                "to run; check strict_hf wiring on the adapter."
            )
        provenance = _provenance_for(bundle, example_source=probe_source)

        runner = CheckpointedBundleRunner(
            label=bundle.label,
            adapter=adapter,
            caller=caller,
            spec=bundle.spec,
            checkpoint_path=checkpoint_path,
            runner_cfg=runner_cfg,
            resume=True,
            provenance=provenance,
            extra_metadata={
                "scheduler_status": scheduler.status(),
                "fail_fast_on_quota": True,
                "scheduler_timeout_s": scheduler_timeout_s,
            },
        )

        logger.info(
            "live-hf runner starting bundle %s (adapter=%s, n=%d, models=%s)",
            bundle.label,
            bundle.spec.adapter_name,
            bundle.spec.n_examples,
            list(bundle.spec.models),
        )
        t0 = time.perf_counter()
        try:
            payload = runner.run_bundle()
            elapsed = time.perf_counter() - t0
            out_path = out_dir / f"{bundle.label}.json"
            out_path.write_text(json.dumps(payload, indent=2))
            run_index["bundles"].append(
                {
                    "label": bundle.label,
                    "status": payload["status"],
                    "out_path": str(out_path),
                    "wallclock_s": round(elapsed, 2),
                }
            )
            logger.info(
                "bundle %s done in %.1fs status=%s -> %s",
                bundle.label, elapsed, payload["status"], out_path.name,
            )
        except PartialRunExit as pe:
            elapsed = time.perf_counter() - t0
            logger.warning(
                "bundle %s hit QuotaExhausted after %.1fs: %s (partial at %s)",
                bundle.label, elapsed, pe.reason, pe.partial_path,
            )
            run_index["bundles"].append(
                {
                    "label": bundle.label,
                    "status": "partial_quota",
                    "out_path": str(pe.partial_path),
                    "wallclock_s": round(elapsed, 2),
                    "partial_reason": pe.reason,
                    "window_summary": pe.window_summary,
                }
            )
            if not continue_on_partial:
                logger.warning(
                    "continue_on_partial=False; stopping the live runner "
                    "here. Re-run later to resume from checkpoint."
                )
                break

    run_index["ts_end"] = _utcnow_iso()
    run_index["scheduler_status"] = scheduler.status()
    return run_index


# --- summaries --------------------------------------------------------


def _write_live_summary(run_index: dict[str, Any], *, out_dir: Path) -> Path:
    """Emit p6a/_summary_live.json — non-destructive companion to _summary.json."""
    out_path = out_dir / "_summary_live.json"
    payload = {
        "ts": _utcnow_iso(),
        "git_sha": run_index.get("git_sha"),
        "source": "live_hf_rerun",
        "bundles": run_index.get("bundles", []),
        "scheduler_status": run_index.get("scheduler_status"),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# --- CLI --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scope",
        choices=(
            "ruler",
            "hallu",
            "swebench",
            "all_real_capable",
            "core4",
            "power_ext",
            "full_battery",
        ),
        default="full_battery",
        help=(
            "which benchmark family to re-run against real HF data; "
            "'full_battery' is the locked 7-bundle v2.1 run (4 RULER + 2 "
            "hallu + SWE-bench); 'core4' is the minimum-viable v2.1 subset; "
            "'power_ext' is the v2.1.1 power-extension amendment "
            "(H1_ruler_16384_live_ext N=30, H_TQA_live_v2_ext N=30, "
            "H_SWEB_live_ext N=15 with scheduler_timeout_s=900)"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print expected input_tokens + rough USD cost; no API calls",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="actually call the Claude CLI via the scheduler (costs $$)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="model names (default: Haiku + Sonnet)",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(LIVE_DEFAULT_SEEDS),
        help="random seeds (default: 0 1)",
    )
    p.add_argument(
        "--ruler-n",
        type=int,
        default=LIVE_DEFAULT_RULER_N,
        help=f"n_examples per RULER bundle per seed (default: {LIVE_DEFAULT_RULER_N})",
    )
    p.add_argument(
        "--hallu-n",
        type=int,
        default=LIVE_DEFAULT_HALLU_N,
        help=f"n_examples per hallucination bundle per seed (default: {LIVE_DEFAULT_HALLU_N})",
    )
    p.add_argument(
        "--sweb-n",
        type=int,
        default=LIVE_DEFAULT_SWEB_N,
        help=f"n_examples for SWE-bench Verified bundle per seed (default: {LIVE_DEFAULT_SWEB_N})",
    )
    p.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        default=list(LIVE_DEFAULT_TIERS),
        help=(
            "RULER token tiers. Default matches the locked pre-registration: "
            f"{' '.join(str(t) for t in LIVE_DEFAULT_TIERS)}. 32768 is NOT on "
            "simonjegou/ruler and would silently fall back to synthetic — "
            "don't use it."
        ),
    )
    p.add_argument(
        "--allow-override-preregistration",
        action="store_true",
        help=(
            "Allow CLI flags to override the locked full_battery / core4 / "
            "power_ext pre-registration (N, seeds, tiers, models, scheduler "
            "timeout). Default rejects any override so operators cannot "
            "accidentally publish a non-preregistered run under any "
            "pre-registered label."
        ),
    )
    p.add_argument(
        "--scheduler-timeout-s",
        type=int,
        default=300,
        help=(
            "per-CLI-call subprocess timeout (seconds). Default 300 s is "
            "fine for RULER/TQA-sized prompts. The v2.1.1 power_ext scope "
            f"locks this at {LIVE_EXT_SWEB_TIMEOUT_S} s to clear the Claude "
            "CLI SessionStart-hook abort path on SWE-bench inputs."
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="per-call max output tokens passed to the scheduler",
    )
    p.add_argument(
        "--bootstrap-n",
        type=int,
        default=2_000,
        help="bootstrap samples for CI",
    )
    p.add_argument(
        "--permutation-n",
        type=int,
        default=2_000,
        help="permutations for paired test",
    )
    p.add_argument(
        "--continue-on-partial",
        action="store_true",
        help="keep running the next bundle after a QuotaExhausted hit",
    )
    p.add_argument(
        "--out-dir",
        default=str(P6A_RESULTS_DIR),
        help="output directory for JSON receipts (default: experiments/results/p6a)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    if not args.dry_run and not args.live:
        print(
            "error: one of --dry-run or --live is required. Use --dry-run "
            "to estimate cost without making any API calls.",
            file=sys.stderr,
        )
        return 2

    try:
        _enforce_preregistration(
            args.scope,
            args_ruler_n=args.ruler_n,
            args_hallu_n=args.hallu_n,
            args_sweb_n=args.sweb_n,
            args_seeds=tuple(args.seeds),
            args_tiers=tuple(args.tiers),
            args_models=tuple(args.models),
            args_scheduler_timeout_s=args.scheduler_timeout_s,
            allow_override=args.allow_override_preregistration,
        )
    except PreregistrationViolation as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    bundles = _select_bundles(
        args.scope,
        models=tuple(args.models),
        seeds=tuple(args.seeds),
        ruler_n=args.ruler_n,
        hallu_n=args.hallu_n,
        sweb_n=args.sweb_n,
        tiers=tuple(args.tiers),
    )
    runner_cfg = RunnerConfig(
        max_tokens=args.max_tokens,
        bootstrap_n=args.bootstrap_n,
        permutation_n=args.permutation_n,
    )

    if args.dry_run:
        logger.info(
            "dry-run: %d bundles, scope=%s, models=%s, seeds=%s",
            len(bundles), args.scope, args.models, args.seeds,
        )
        estimates = [
            _dry_run_one_bundle(b, runner_cfg=runner_cfg) for b in bundles
        ]
        _print_dry_run_table(estimates)
        return 0

    # --live path
    out_dir = Path(args.out_dir).resolve()
    try:
        run_index = _run_live(
            bundles,
            runner_cfg=runner_cfg,
            out_dir=out_dir,
            continue_on_partial=args.continue_on_partial,
            scheduler_timeout_s=args.scheduler_timeout_s,
        )
    except ProvenanceIntegrityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    summary_path = _write_live_summary(run_index, out_dir=out_dir)
    logger.info("live-hf runner finished; summary at %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
