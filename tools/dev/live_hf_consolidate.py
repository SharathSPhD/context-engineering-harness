"""Consolidate live-HF rerun artifacts into a comprehensive summary.

Merges completed bundle JSONs (``experiments/results/p6a/*_live.json``) and
partial bundle JSONs (``.cache/live_hf_checkpoints/*_partial.json``) into a
single ``experiments/results/p6a/_summary_live.json``, and writes a
``experiments/results/p6a/live_run_index.json`` with cumulative quota
accounting pulled from the scheduler ledger.

Idempotent: safe to run multiple times. Non-destructive: leaves the
original ``*_live.json`` files and synthetic ``_summary.json`` untouched.

Usage::

    uv run --active python -m tools.dev.live_hf_consolidate

"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from tools.dev.scheduler import CLIBudgetScheduler, SchedulerConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "p6a"
CHECKPOINT_DIR = REPO_ROOT / ".cache" / "live_hf_checkpoints"
JOURNAL_PATH = REPO_ROOT / "tools" / "dev" / "orchestration" / "attractor_journal.jsonl"


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


_HF_REVISION_CACHE: dict[str, str] = {}


def _resolve_hf_revision(dataset_id: str | None) -> str | None:
    """Resolve a Hugging Face dataset repo id to its current HEAD commit SHA.

    Returns ``None`` when ``dataset_id`` is missing, when ``huggingface_hub``
    is not installed, or when the network call fails. Memoised per-process.
    """
    if not dataset_id:
        return None
    if dataset_id in _HF_REVISION_CACHE:
        return _HF_REVISION_CACHE[dataset_id]
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        sha = HfApi().dataset_info(dataset_id).sha
    except Exception as exc:  # noqa: BLE001 — hub surfaces many error classes
        logger.warning("hf_revision lookup failed for %s: %s", dataset_id, exc)
        return None
    if sha:
        _HF_REVISION_CACHE[dataset_id] = sha
    return sha


def _stamp_hf_revision(prov: dict[str, Any] | None) -> dict[str, Any] | None:
    """Add ``provenance.hf_revision`` in place when ``source=huggingface``."""
    if not prov:
        return prov
    if prov.get("source") != "huggingface":
        return prov
    if prov.get("hf_revision"):
        return prov
    sha = _resolve_hf_revision(prov.get("hf_dataset_id"))
    if sha:
        prov["hf_revision"] = sha
    return prov


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _extract_completed_entry(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    outcome = data.get("outcome") or {}
    return {
        "label": data.get("label", path.stem),
        "status": data.get("status", "complete"),
        "out_path": str(path),
        "wallclock_s": data.get("wallclock_s"),
        "outcome": {
            "treatment_metric": outcome.get("treatment_metric"),
            "baseline_metric": outcome.get("baseline_metric"),
            "delta_observed": outcome.get("delta_observed"),
            "p_value": outcome.get("p_value"),
            "cohens_d": outcome.get("cohens_d"),
            "target_met": outcome.get("target_met"),
            "n_examples_used": outcome.get("n_examples_used"),
            "n_seeds_used": outcome.get("n_seeds_used"),
        },
        "provenance": _stamp_hf_revision(data.get("provenance")),
    }


def _extract_partial_entry(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    partial = data.get("partial") or {}
    md = data.get("metadata") or {}
    return {
        "label": data.get("label", path.stem.replace("_partial", "")),
        "status": data.get("status", "partial_quota"),
        "out_path": str(path),
        "wallclock_s": data.get("wallclock_s"),
        "partial_reason": partial.get("reason"),
        "window_summary": partial.get("window_summary"),
        "checkpointed_records": md.get("checkpointed_records"),
        "provenance": _stamp_hf_revision(data.get("provenance")),
    }


def _load_ext_score_overlays(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    """Load ``*_score.json`` files produced by
    :mod:`tools.dev.score_ext_checkpoints`.

    These are offline re-reads of ``_ext`` checkpoint JSONLs and carry
    pooled + per-model outcomes plus caveats. We splice them into the
    partial-bundle entries so ``_summary_live.json`` carries a scored
    outcome for the v2.1.1 power-extension bundles even when the live
    run ended in ``partial_quota``.
    """
    if not checkpoint_dir.exists():
        return {}
    overlays: dict[str, dict[str, Any]] = {}
    for p in sorted(checkpoint_dir.glob("*_score.json")):
        try:
            data = _load_json(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip score file %s: %s", p, exc)
            continue
        label = data.get("label") or p.stem.replace("_score", "")
        overlays[label] = {"path": str(p), "payload": data}
    return overlays


def _splice_ext_overlay(
    entry: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    payload = overlay["payload"]
    pooled = payload.get("pooled_outcome") or {}
    entry = dict(entry)
    entry["score_source"] = overlay["path"]
    entry["outcome"] = {
        "treatment_metric": pooled.get("treatment_metric"),
        "baseline_metric": pooled.get("baseline_metric"),
        "delta_observed": pooled.get("delta_observed"),
        "ci_low": pooled.get("ci_low"),
        "ci_high": pooled.get("ci_high"),
        "p_value": pooled.get("p_value"),
        "cohens_d": pooled.get("cohens_d"),
        "target_met": pooled.get("target_met"),
        "n_paired": pooled.get("n_paired"),
    }
    entry["per_model_outcome"] = payload.get("per_model_outcome")
    entry["per_cell_counts"] = payload.get("per_cell_counts")
    entry["caveats"] = payload.get("caveats")
    return entry


def _count_quota_events(journal_path: Path) -> dict[str, int]:
    """Count QUOTA_EXHAUSTED / rate-limit events in the attractor journal."""
    if not journal_path.exists():
        return {"quota_exhausted": 0, "halt_rate_limit": 0}
    counts = {"quota_exhausted": 0, "halt_rate_limit": 0}
    for line in journal_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        note = (ev.get("note") or "").lower()
        regime = (ev.get("regime") or "").lower()
        if "quota_exhausted" in note or ev.get("event") == "QUOTA_EXHAUSTED":
            counts["quota_exhausted"] += 1
        if regime == "halt" and "rate-limit" in note:
            counts["halt_rate_limit"] += 1
    return counts


def consolidate(
    *,
    results_dir: Path = RESULTS_DIR,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    journal_path: Path = JOURNAL_PATH,
) -> tuple[Path, Path]:
    """Rebuild _summary_live.json and live_run_index.json.

    Returns
    -------
    (summary_path, run_index_path)
    """
    bundles: list[dict[str, Any]] = []

    completed_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for pattern in ("*_live.json", "*_live_*.json"):
        for p in results_dir.glob(pattern):
            if p.name in {"_summary_live.json", "live_run_index.json"}:
                continue
            if p in seen_paths:
                continue
            seen_paths.add(p)
            completed_paths.append(p)
    completed_paths.sort()
    completed_by_label: dict[str, Path] = {}
    for p in completed_paths:
        try:
            data = _load_json(p)
        except Exception as exc:
            logger.warning("skip %s: %s", p, exc)
            continue
        completed_by_label[data.get("label", p.stem)] = p

    for label, p in sorted(completed_by_label.items()):
        bundles.append(_extract_completed_entry(p))

    ext_overlays = _load_ext_score_overlays(checkpoint_dir)

    partial_paths = sorted(checkpoint_dir.glob("*_partial.json")) if checkpoint_dir.exists() else []
    for p in partial_paths:
        entry = _extract_partial_entry(p)
        if entry["label"] in completed_by_label:
            continue
        overlay = ext_overlays.get(entry["label"])
        if overlay is not None:
            entry = _splice_ext_overlay(entry, overlay)
        bundles.append(entry)

    scheduler = CLIBudgetScheduler(
        SchedulerConfig(
            cache_root=str(REPO_ROOT / ".cache" / "llm"),
            ledger_path=str(REPO_ROOT / ".cache" / "cost_ledger.db"),
            journal_path=str(journal_path),
            max_input_tokens_per_window=2_000_000,
            fail_fast_on_quota=True,
        )
    )
    scheduler_status = scheduler.status()
    quota_events = _count_quota_events(journal_path)

    summary_payload = {
        "ts": _utcnow_iso(),
        "git_sha": _git_sha(),
        "source": "live_hf_consolidate",
        "bundles": bundles,
        "scheduler_status": scheduler_status,
    }
    summary_path = results_dir / "_summary_live.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    n_complete = sum(1 for b in bundles if b["status"] == "complete")
    n_partial = sum(1 for b in bundles if b["status"] == "partial_quota")
    run_index = {
        "ts": _utcnow_iso(),
        "git_sha": _git_sha(),
        "source": "live_hf_consolidate",
        "bundle_counts": {
            "complete": n_complete,
            "partial_quota": n_partial,
            "total_tracked": len(bundles),
        },
        "bundles": [
            {
                "label": b["label"],
                "status": b["status"],
                "out_path": b.get("out_path"),
                "outcome_snapshot": b.get("outcome"),
                "checkpointed_records": b.get("checkpointed_records"),
                "partial_reason": b.get("partial_reason"),
            }
            for b in bundles
        ],
        "scheduler_status_current_window": scheduler_status,
        "cumulative_totals": scheduler_status.get("total", {}),
        "quota_events_journal": quota_events,
    }
    run_index_path = results_dir / "live_run_index.json"
    run_index_path.write_text(json.dumps(run_index, indent=2))

    return summary_path, run_index_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    summary_path, index_path = consolidate()
    logger.info("wrote %s", summary_path.relative_to(REPO_ROOT))
    logger.info("wrote %s", index_path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
