"""Offline scorer for v2.1.1 power-extension checkpoint JSONLs.

Why this exists
---------------
The v2.1.1 power-extension run ran partway through its budget window and
was halted by the user ("quota is done...consolidate at this stage").
The ``CheckpointedBundleRunner`` emits a ``*_partial.json`` stub with a
null ``outcome`` block when it bails on a ``QuotaExhausted`` — that is
correct for live-resumable state, but it means the final live summary
has no numeric outcome for the ``_ext`` bundles, even when the
underlying JSONL has enough rows to score.

This tool re-reads each ``_ext`` checkpoint JSONL **without** making any
Claude CLI calls and computes a finalized paired outcome (paired delta,
95% bootstrap CI on the diff, paired permutation p-value, paired
Cohen's d, and pass-gate verdict) using the same primitives as
``CheckpointedBundleRunner._compute_outcome``.

What it does / does not do
--------------------------
* **Does**: pair rows by ``(example_id, seed, model)`` across the
  ``harness_on`` / ``harness_off`` conditions (the same ``_pair_runs``
  contract as ``src/benchmarks/runner.py``), drop unpaired rows, and
  compute the stat block. Unbalanced per-cell coverage (common in
  v2.1.1 because Sonnet seed 1 for RULER-16K never extended past
  N=15) is handled honestly: the pooled pair count = ``min(on, off)``
  per cell, summed across cells, and is reported as
  ``n_paired_total`` so readers can't confuse a partial extension
  with a full N=30.
* **Does**: emit per-model sub-block outcomes alongside the pooled
  number, because the pre-registered read is "per-model paired test,
  pooled reported alongside" and pooling across Haiku@30 +
  Sonnet-seed-0@30 + Sonnet-seed-1@15 muddies the headline.
* **Does not**: touch Hugging Face. The ``example_id`` alignment
  is entirely local to the checkpoint file. No HF traffic, no Claude
  traffic, no token spend — this is a re-read of already-billed bits.

Outputs
-------
One JSON file per input checkpoint, written next to the existing
``_partial.json``:

* ``.cache/live_hf_checkpoints/H1_ruler_16384_live_ext_score.json``
* ``.cache/live_hf_checkpoints/H_TQA_live_v2_ext_score.json``
* ``.cache/live_hf_checkpoints/H_SWEB_live_ext_score.json``

Each payload includes the pooled outcome, per-model outcomes, the raw
per-cell sample counts, and a ``caveats`` list surfacing any
unbalanced-cell or zero-new-data issues so downstream paper / article
writers cannot accidentally over-claim.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.benchmarks.base import BenchmarkResult, BenchmarkRun
from src.benchmarks.hypothesis import HypothesisSpec, TargetDirection
from src.benchmarks.runner import _pair_runs, _verdict
from src.benchmarks.stats import bootstrap_ci, cohens_d, paired_permutation_test

from experiments.v2.p6a.specs_live import (
    LIVE_DEFAULT_SEEDS,
    LIVE_DEFAULT_SWEB_N,
    LIVE_EXT_HALLU_N,
    LIVE_EXT_RULER_N,
    h1_ruler_16384_ext_spec,
    h_swebench_verified_live_ext_spec,
    h_tqa_live_v2_ext_spec,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / ".cache" / "live_hf_checkpoints"

# stats wiring — matches the RunnerConfig dataclass defaults in
# src/benchmarks/runner.py (bootstrap_n=10_000, permutation_n=10_000,
# ci_level=0.95). NOTE: run_live_hf.py's argparse layer overrides
# these to 2_000 on the CLI default path; the v2.1 live bundles were
# therefore scored with 2_000 resamples on the wire, while the v2.1.1
# *_ext rows in Table~T8 were scored with 10_000 resamples offline
# from this script. The difference is below the reported decimal
# precision for all v2.1.1 numbers and does not change any gate
# verdict, but replayers who want byte-identical *_score.json files
# should keep these at 10_000; the replay recipes in Appendix~G and
# docs/release_v2.1.1_power_ext.md also pass --bootstrap-n 10000
# --permutation-n 10000 to run_live_hf.py to align both tools.
BOOTSTRAP_N: int = 10_000
PERMUTATION_N: int = 10_000
CI_LEVEL: float = 0.95


@dataclass
class _Bundle:
    """One ext-bundle scoring config."""

    label: str
    checkpoint_path: Path
    spec: HypothesisSpec
    expected_n_per_cell: int  # per (model, seed, condition)
    caveats: list[str] = field(default_factory=list)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(path.read_text().splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            logger.warning("malformed line %d at %s: %s", i, path, exc)
    return rows


def _row_to_result(row: dict[str, Any]) -> BenchmarkResult:
    return BenchmarkResult(
        example_id=str(row["example_id"]),
        condition=str(row["condition"]),
        seed=int(row["seed"]),
        prediction=str(row.get("prediction", "")),
        score=float(row.get("score", 0.0)),
        correct=bool(row.get("correct", False)),
        input_tokens=int(row.get("input_tokens", 0)),
        output_tokens=int(row.get("output_tokens", 0)),
        latency_ms=float(row.get("latency_ms", 0.0)),
        error=str(row.get("error", "")),
        metadata=dict(row.get("metadata", {})),
    )


def _group_runs(
    rows: list[dict[str, Any]],
    condition: str,
) -> list[BenchmarkRun]:
    """Group rows by (adapter, model, seed) for one condition into BenchmarkRuns."""
    by_key: dict[tuple[str, str, int], list[BenchmarkResult]] = {}
    for row in rows:
        if row["condition"] != condition:
            continue
        key = (str(row["adapter"]), str(row["model"]), int(row["seed"]))
        by_key.setdefault(key, []).append(_row_to_result(row))
    runs: list[BenchmarkRun] = []
    for (adapter_name, model, seed), results in sorted(by_key.items()):
        # sort so pairing is deterministic across re-scores
        results.sort(key=lambda r: r.example_id)
        runs.append(
            BenchmarkRun(
                adapter_name=adapter_name,
                model=model,
                condition=condition,
                seed=seed,
                results=results,
            )
        )
    return runs


def _score_paired(
    paired_t: list[float],
    paired_b: list[float],
    spec: HypothesisSpec,
) -> dict[str, Any]:
    if not paired_t or not paired_b or len(paired_t) != len(paired_b):
        return {
            "n_paired": len(paired_t),
            "treatment_metric": None,
            "baseline_metric": None,
            "delta_observed": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "cohens_d": None,
            "target_met": False,
            "note": "insufficient or unbalanced paired data",
        }
    treatment_metric = sum(paired_t) / len(paired_t)
    baseline_metric = sum(paired_b) / len(paired_b)
    delta = treatment_metric - baseline_metric
    diffs = [t - b for t, b in zip(paired_t, paired_b)]
    _, ci_low, ci_high = bootstrap_ci(diffs, ci=CI_LEVEL, n_bootstrap=BOOTSTRAP_N)
    p_value = paired_permutation_test(paired_t, paired_b, n_permutations=PERMUTATION_N)
    d = cohens_d(paired_t, paired_b)
    target_met = _verdict(spec, delta, p_value)
    return {
        "n_paired": len(paired_t),
        "treatment_metric": round(treatment_metric, 4),
        "baseline_metric": round(baseline_metric, 4),
        "delta_observed": round(delta, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round(d, 4),
        "target_met": bool(target_met),
    }


def _per_cell_counts(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Return ``{model: {seed_cond: n}}`` for transparency in outputs."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        model = str(r["model"])
        key = f"seed{r['seed']}_{r['condition']}"
        out.setdefault(model, {}).setdefault(key, 0)
        out[model][key] += 1
    return out


def _score_bundle(bundle: _Bundle) -> dict[str, Any]:
    rows = _load_rows(bundle.checkpoint_path)
    treatment = _group_runs(rows, bundle.spec.treatment_condition)
    baseline = _group_runs(rows, bundle.spec.baseline_condition)

    # Pooled across all models
    paired_t, paired_b = _pair_runs(treatment, baseline, bundle.spec.metric)
    pooled = _score_paired(paired_t, paired_b, bundle.spec)

    # Per-model
    per_model: dict[str, Any] = {}
    for model in bundle.spec.models:
        t_m = [r for r in treatment if r.model == model]
        b_m = [r for r in baseline if r.model == model]
        if not t_m or not b_m:
            per_model[model] = {
                "n_paired": 0,
                "treatment_metric": None,
                "baseline_metric": None,
                "delta_observed": None,
                "ci_low": None,
                "ci_high": None,
                "p_value": None,
                "cohens_d": None,
                "target_met": False,
                "note": "no rows for this model in checkpoint",
            }
            continue
        pt, pb = _pair_runs(t_m, b_m, bundle.spec.metric)
        per_model[model] = _score_paired(pt, pb, bundle.spec)

    errored = sum(1 for r in rows if r.get("error"))
    counts = _per_cell_counts(rows)
    caveats = list(bundle.caveats)

    # Flag unbalanced cells so paper / article writers cannot over-claim.
    expected = bundle.expected_n_per_cell
    imbalances: list[str] = []
    for model, cells in counts.items():
        for key, n in cells.items():
            if n != expected:
                imbalances.append(f"{model} {key}={n} (expected {expected})")
    if imbalances:
        caveats.append(
            "Per-cell sample imbalance vs. pre-registered N: "
            + "; ".join(sorted(imbalances))
        )
    if errored:
        caveats.append(
            f"{errored} row(s) recorded with ``error != ''`` (scored 0 at CLI "
            "level per checkpoint schema); included in paired totals as-is."
        )

    return {
        "label": bundle.label,
        "checkpoint_path": str(bundle.checkpoint_path),
        "spec_snapshot": {
            "hypothesis_id": bundle.spec.hypothesis_id,
            "treatment_condition": bundle.spec.treatment_condition,
            "baseline_condition": bundle.spec.baseline_condition,
            "metric": bundle.spec.metric,
            "direction": (
                bundle.spec.direction.value
                if isinstance(bundle.spec.direction, TargetDirection)
                else str(bundle.spec.direction)
            ),
            "delta": bundle.spec.delta,
            "n_examples_pre_reg": bundle.spec.n_examples,
            "seeds_pre_reg": list(bundle.spec.seeds),
            "models_pre_reg": list(bundle.spec.models),
            "significance_alpha": bundle.spec.significance_alpha,
        },
        "n_rows_total": len(rows),
        "n_rows_errored": errored,
        "per_cell_counts": counts,
        "pooled_outcome": pooled,
        "per_model_outcome": per_model,
        "caveats": caveats,
    }


def _bundles() -> list[_Bundle]:
    ruler_spec = h1_ruler_16384_ext_spec(n_examples=LIVE_EXT_RULER_N)
    tqa_spec = h_tqa_live_v2_ext_spec(n_examples=LIVE_EXT_HALLU_N)
    sweb_spec = h_swebench_verified_live_ext_spec(n_examples=LIVE_DEFAULT_SWEB_N)
    return [
        _Bundle(
            label=ruler_spec.label,
            checkpoint_path=CKPT_DIR / f"{ruler_spec.label}.jsonl",
            spec=ruler_spec.spec,
            expected_n_per_cell=LIVE_EXT_RULER_N,
            caveats=[
                "v2.1.1 power extension of the v2.1 N=15-per-cell run "
                "(n_pair=60 pooled). First 15 rows per (model, seed, "
                "condition) are seeded from H1_ruler_16384_live.jsonl "
                "(same seeds => deterministic "
                "superset); the additional rows were billed in v2.1.1."
            ],
        ),
        _Bundle(
            label=tqa_spec.label,
            checkpoint_path=CKPT_DIR / f"{tqa_spec.label}.jsonl",
            spec=tqa_spec.spec,
            expected_n_per_cell=LIVE_EXT_HALLU_N,
            caveats=[
                "v2.1.1 power extension of the v2.1 N=15-per-cell run "
                "(n_pair=60 pooled = 2 models x 2 seeds x 15 examples). "
                "Seed file H_TQA_live_v2.jsonl copied verbatim; v2.1.1 "
                "live run hit QuotaExhausted before any new rows were "
                "billed, so this checkpoint is byte-identical to v2.1 "
                "and the outcome below is the v2.1 n_pair=60 result "
                "re-reported."
            ],
        ),
        _Bundle(
            label=sweb_spec.label,
            checkpoint_path=CKPT_DIR / f"{sweb_spec.label}.jsonl",
            spec=sweb_spec.spec,
            expected_n_per_cell=LIVE_DEFAULT_SWEB_N,
            caveats=[
                "v2.1.1 infrastructure rescue of the v2.1 SWE-bench run. "
                "Seed file built by filtering error rows out of "
                "H_SWEB_live_n15.jsonl; v2.1.1 live run hit QuotaExhausted "
                "before any new rows were billed, so this checkpoint "
                "contains only the successful v2.1 rows (haiku-only) and "
                "no sonnet coverage. Pooled scoring below is therefore a "
                "haiku-only paired read; the ``n_paired`` count will "
                "under-count the pre-registered (15 × 2 × 2) cells."
            ],
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    outputs: list[Path] = []
    for bundle in _bundles():
        if not bundle.checkpoint_path.exists():
            logger.warning("skipping %s: %s missing", bundle.label, bundle.checkpoint_path)
            continue
        payload = _score_bundle(bundle)
        out_path = bundle.checkpoint_path.parent / f"{bundle.label}_score.json"
        out_path.write_text(json.dumps(payload, indent=2))
        outputs.append(out_path)
        pooled = payload["pooled_outcome"]
        logger.info(
            "%s: pooled n=%s delta=%s p=%s d=%s target_met=%s",
            bundle.label,
            pooled.get("n_paired"),
            pooled.get("delta_observed"),
            pooled.get("p_value"),
            pooled.get("cohens_d"),
            pooled.get("target_met"),
        )
    logger.info("wrote %d score file(s)", len(outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
