"""Compute paired SWE-bench outcome stats from the SWEB checkpoint JSONL.

**v2.1 run (``H_SWEB_live_n15``)** — N=15 instances × 2 seeds × 2 models ×
2 conditions = 120 target records. Claude-haiku-4-5 finished fully (60
records, 30 paired observations) before the second Anthropic rolling-
window closed; ~77 % of haiku attempts and 100 % of sonnet attempts
aborted at the CLI ``SessionStart`` hook under the 300 s default
subprocess timeout. Pre-registered imputation scored aborts as 0 on both
sides; the paired-delta on haiku was +0.109 (p=0.032, d_z=0.488).

**v2.1.1 infra-rescue run (``H_SWEB_live_ext``)** — same pre-registered N,
seeds, models, and scorer, but the CLI subprocess timeout is raised to
900 s so the ``SessionStart`` hook completes. The goal is to eliminate
the CLI-abort pathway that dominated the v2.1 error rates, so the
paired-delta reflects actual model performance rather than infrastructure
flakiness.

This module writes ``swe_bench_outcomes.json`` (v2.1 run, unchanged for
addendum-reproducibility) **and** ``swe_bench_outcomes_ext.json`` (v2.1.1
rescue) side-cars under ``experiments/results/p6a/`` so Phase 3 paper /
table regeneration can pick up both outcomes without re-parsing the raw
JSONLs. Both calls are idempotent.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEB_JSONL = REPO_ROOT / ".cache" / "live_hf_checkpoints" / "H_SWEB_live_n15.jsonl"
SWEB_JSONL_EXT = REPO_ROOT / ".cache" / "live_hf_checkpoints" / "H_SWEB_live_ext.jsonl"
OUT_PATH = REPO_ROOT / "experiments" / "results" / "p6a" / "swe_bench_outcomes.json"
OUT_PATH_EXT = REPO_ROOT / "experiments" / "results" / "p6a" / "swe_bench_outcomes_ext.json"

_V21_CAVEAT = (
    "claude CLI 2.1 SessionStart hook reliably aborts SWE-bench prompts when "
    "the startup exceeds the 300s submit timeout; ~77% of haiku attempts and "
    "100% of sonnet attempts were blocked at the CLI layer. full_errors_as_zero "
    "treats those aborts conservatively as score=0.0 on both sides (per "
    "pre-registration); clean_pairs_only restricts to (seed, instance) combos "
    "where neither side aborted, which loses most of the signal."
)

_V211_CAVEAT = (
    "v2.1.1 infra-rescue rerun: same pre-registered N=15, seeds=(0,1), models, "
    "and scorer, but the CLI subprocess timeout is raised from 300s to 900s so "
    "the Claude CLI SessionStart hook completes before being killed. Clean "
    "(non-error) rows from the v2.1 run (``H_SWEB_live_n15.jsonl``) were seeded "
    "into this checkpoint to re-use previously-billed observations; the runner "
    "only re-billed the (seed, instance, model, condition) tuples that errored "
    "in v2.1. full_errors_as_zero retains the pre-registered conservative "
    "imputation (score=0 on both sides for any remaining error); clean_pairs_only "
    "reports the delta on the rows where both sides returned a scored patch."
)


def _paired_permutation_p(diffs: np.ndarray, n_iter: int = 10000, seed: int = 42) -> float:
    if len(diffs) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_iter, len(diffs)))
    perm_means = (signs * diffs).mean(axis=1)
    observed = diffs.mean()
    return float((np.abs(perm_means) >= abs(observed)).mean())


def _cohens_dz(diffs: np.ndarray) -> float:
    if len(diffs) < 2:
        return 0.0
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else 0.0


def compute_sweb_outcomes(
    *,
    jsonl_path: Path = SWEB_JSONL,
    caveat: str = _V21_CAVEAT,
) -> dict[str, Any]:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    per_model: dict[str, Any] = {}
    for model in sorted({r["model"] for r in rows}):
        model_rows = [r for r in rows if r["model"] == model]
        pairs: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in model_rows:
            pairs[(r["seed"], r["example_id"])][r.get("condition", "?")] = r
        complete_pairs = [
            (v["harness_on"], v["harness_off"])
            for v in pairs.values()
            if "harness_on" in v and "harness_off" in v
        ]
        n_attempted = len(complete_pairs)
        on_scores = np.array([p[0]["score"] for p in complete_pairs])
        off_scores = np.array([p[1]["score"] for p in complete_pairs])
        diffs = on_scores - off_scores
        errs_on = int(sum(1 for p in complete_pairs if p[0].get("error")))
        errs_off = int(sum(1 for p in complete_pairs if p[1].get("error")))
        clean_pairs = [
            (a, b)
            for a, b in complete_pairs
            if not a.get("error") and not b.get("error")
        ]
        n_clean = len(clean_pairs)
        clean_on = np.array([p[0]["score"] for p in clean_pairs]) if n_clean else np.array([])
        clean_off = np.array([p[1]["score"] for p in clean_pairs]) if n_clean else np.array([])
        clean_diff = clean_on - clean_off

        per_model[model] = {
            "n_paired_attempts": n_attempted,
            "n_total_records": len(model_rows),
            "per_condition_error_rate_on": round(errs_on / n_attempted, 4) if n_attempted else None,
            "per_condition_error_rate_off": round(errs_off / n_attempted, 4) if n_attempted else None,
            "full_errors_as_zero": {
                "harness_on_mean": round(float(on_scores.mean()), 4) if n_attempted else None,
                "harness_off_mean": round(float(off_scores.mean()), 4) if n_attempted else None,
                "delta_observed": round(float(diffs.mean()), 4) if n_attempted else None,
                "cohens_dz": round(_cohens_dz(diffs), 4),
                "paired_permutation_p": round(_paired_permutation_p(diffs), 4),
                "n_nonzero_on": int((on_scores > 0).sum()),
                "n_nonzero_off": int((off_scores > 0).sum()),
            },
            "clean_pairs_only": {
                "n_clean_pairs": n_clean,
                "harness_on_mean": round(float(clean_on.mean()), 4) if n_clean else None,
                "harness_off_mean": round(float(clean_off.mean()), 4) if n_clean else None,
                "delta_observed": round(float(clean_diff.mean()), 4) if n_clean else None,
                "cohens_dz": round(_cohens_dz(clean_diff), 4),
                "paired_permutation_p": round(_paired_permutation_p(clean_diff), 4),
            },
        }
    return {
        "source_jsonl": str(jsonl_path.relative_to(REPO_ROOT)),
        "caveat": caveat,
        "per_model": per_model,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    v21_out = compute_sweb_outcomes(jsonl_path=SWEB_JSONL, caveat=_V21_CAVEAT)
    OUT_PATH.write_text(json.dumps(v21_out, indent=2))
    logger.info("wrote %s", OUT_PATH.relative_to(REPO_ROOT))
    if SWEB_JSONL_EXT.exists():
        v211_out = compute_sweb_outcomes(jsonl_path=SWEB_JSONL_EXT, caveat=_V211_CAVEAT)
        OUT_PATH_EXT.write_text(json.dumps(v211_out, indent=2))
        logger.info("wrote %s", OUT_PATH_EXT.relative_to(REPO_ROOT))
    else:
        logger.info(
            "skipping %s: %s not present yet",
            OUT_PATH_EXT.relative_to(REPO_ROOT),
            SWEB_JSONL_EXT.relative_to(REPO_ROOT),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
