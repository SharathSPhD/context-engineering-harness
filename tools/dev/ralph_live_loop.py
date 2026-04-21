"""Ralph-style orchestrator for the live-HF full-battery run.

Runs ``python -m experiments.v2.p6a.run_live_hf --scope full_battery
--continue-on-partial --live`` in a loop until every bundle reports
``status=complete`` in ``experiments/results/p6a/_summary_live.json``.

Between iterations it:
  1. Consolidates partial + completed JSONs via
     ``tools.dev.live_hf_consolidate.consolidate()``.
  2. Emits a structured progress record to
     ``.cache/ralph_live_loop.jsonl``.
  3. Sleeps ``--min-sleep-s`` (default 300s) or honors the scheduler's
     reported next-window ETA, whichever is longer.
  4. Enforces hard stops via ``--max-iterations`` (default 40) and
     ``--max-wall-hours`` (default 96).

Typical CLI usage::

    uv run --active python -m tools.dev.ralph_live_loop

To probe completion without making any API calls::

    uv run --active python -m tools.dev.ralph_live_loop --done-probe

Design notes
============
- The inner runner already traps ``QuotaExhausted`` and exits 0 after
  writing ``{label}_partial.json`` files. That means the outer loop
  treats an exit code of 0 as "iteration finished", not as "done".
- "Done" is determined by the consolidator: every spec in the locked
  scope must have a matching ``status=complete`` entry in the
  ``_summary_live.json`` under ``--out-dir``. Any other status
  (``partial_quota``, unknown, or absent) keeps the loop running.
- The loop is completion-blind at the content level — a
  ``status=complete`` bundle whose live result is an honest null is
  still complete, the loop just moves on. Bundle content is audited
  downstream by the reviewer fleet, not by this loop.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "p6a"
LOOP_JOURNAL = REPO_ROOT / ".cache" / "ralph_live_loop.jsonl"

# Safety cap on a single inner-runner invocation. A live run that
# doesn't hit quota or terminate within this window is almost
# certainly hung (subprocess deadlock, infinite retry, etc.) and we'd
# rather kill-and-resume than let the outer loop wait forever. The
# cap is sized slightly above the Claude 5h window so a well-behaved
# inner run can complete one full window before being killed.
INNER_RUN_TIMEOUT_S = 6 * 60 * 60

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IterationResult:
    iteration: int
    started_at: str
    ended_at: str
    exit_code: int
    completed_labels: list[str] = field(default_factory=list)
    partial_labels: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    scheduler_status: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _expected_labels(scope: str) -> list[str]:
    """Resolve the full ordered label list for a given scope.

    Defers the import so the probe sub-command doesn't need to load
    HuggingFace-related packages.

    The ``power_ext`` scope (v2.1.1) passes its own N values / tiers
    instead of the v2.1 ``LIVE_DEFAULT_*`` constants so the label list
    resolves cleanly against the pre-registered amendment.
    """
    from experiments.v2.p6a.run_live_hf import _select_bundles  # type: ignore
    from experiments.v2.p6a.specs import DEFAULT_MODELS  # type: ignore
    from experiments.v2.p6a.specs_live import (  # type: ignore
        LIVE_DEFAULT_HALLU_N,
        LIVE_DEFAULT_RULER_N,
        LIVE_DEFAULT_SEEDS,
        LIVE_DEFAULT_SWEB_N,
        LIVE_DEFAULT_TIERS,
        LIVE_EXT_HALLU_N,
        LIVE_EXT_RULER_N,
        LIVE_EXT_RULER_TIERS,
    )

    if scope == "power_ext":
        bundles = _select_bundles(
            scope,
            models=tuple(DEFAULT_MODELS),
            seeds=tuple(LIVE_DEFAULT_SEEDS),
            ruler_n=LIVE_EXT_RULER_N,
            hallu_n=LIVE_EXT_HALLU_N,
            sweb_n=LIVE_DEFAULT_SWEB_N,
            tiers=tuple(LIVE_EXT_RULER_TIERS),
        )
    else:
        bundles = _select_bundles(
            scope,
            models=tuple(DEFAULT_MODELS),
            seeds=tuple(LIVE_DEFAULT_SEEDS),
            ruler_n=LIVE_DEFAULT_RULER_N,
            hallu_n=LIVE_DEFAULT_HALLU_N,
            sweb_n=LIVE_DEFAULT_SWEB_N,
            tiers=tuple(LIVE_DEFAULT_TIERS),
        )
    return [b.label for b in bundles]


def _status_from_summary(
    summary_path: Path, expected_labels: list[str]
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """Bucket each expected bundle into complete / partial / missing.

    Any status other than ``"complete"`` (including unknown values and
    ``"partial_quota"``) counts as non-complete and the loop must keep
    trying. An unexpected status string is logged and bucketed as
    ``partial`` so the caller knows the run isn't done.
    """
    if not summary_path.exists():
        return ([], [], list(expected_labels), {})
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return ([], [], list(expected_labels), {})
    status_by_label: dict[str, str] = {}
    for b in data.get("bundles") or []:
        label = b.get("label")
        status = b.get("status")
        if not isinstance(label, str) or not isinstance(status, str):
            continue
        # Later "complete" entries win over any prior status; a prior
        # "complete" is never downgraded by a subsequent partial.
        prior = status_by_label.get(label)
        if prior == "complete":
            continue
        status_by_label[label] = status
    completed: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    for lab in expected_labels:
        st = status_by_label.get(lab)
        if st == "complete":
            completed.append(lab)
        elif st is None:
            missing.append(lab)
        else:
            if st != "partial_quota":
                logger.warning(
                    "unexpected bundle status %r for %s; treating as partial",
                    st,
                    lab,
                )
            partial.append(lab)
    scheduler = dict(data.get("scheduler_status") or {})
    return (completed, partial, missing, scheduler)


def _run_consolidate(results_dir: Path) -> dict[str, Any]:
    from tools.dev.live_hf_consolidate import consolidate  # type: ignore

    # Newer consolidator supports results_dir; pass it when available so
    # the ralph loop respects a non-default --out-dir. Older versions
    # silently ignore the kwarg since we catch and retry without it.
    try:
        summary_path, _ = consolidate(results_dir=results_dir)  # type: ignore[call-arg]
    except TypeError:
        summary_path, _ = consolidate()
    try:
        return json.loads(Path(summary_path).read_text())
    except Exception as exc:
        logger.warning("failed to re-read summary after consolidate: %s", exc)
        return {}


def _journal_record(payload: dict[str, Any]) -> None:
    LOOP_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with LOOP_JOURNAL.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")


def _launch_inner_runner(
    *,
    scope: str,
    out_dir: Path,
    extra_args: list[str],
    timeout_s: int,
) -> int:
    """Invoke the inner runner as a subprocess and return its exit code.

    Using a subprocess (not an in-process call) so the scheduler's
    per-window state is fully reset between iterations — the scheduler
    reads the persistent ledger on init, so this is equivalent to
    resuming from the on-disk state. A ``timeout_s`` safety cap
    prevents a hung inner run from wedging the outer loop.
    """
    cmd = [
        sys.executable,
        "-m",
        "experiments.v2.p6a.run_live_hf",
        "--scope",
        scope,
        "--live",
        "--continue-on-partial",
        "--out-dir",
        str(out_dir),
        *extra_args,
    ]
    logger.info("iteration kickoff (timeout=%ds): %s", timeout_s, " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), timeout=timeout_s)
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        logger.error(
            "inner runner exceeded %ds; killing and marking iteration as timeout",
            timeout_s,
        )
        return 124  # POSIX timeout convention


def _compute_sleep_s(
    *, scheduler_status: dict[str, Any], min_sleep_s: int
) -> int:
    """Pick the sleep interval for the next iteration.

    Uses the scheduler's ``next_window_at`` ISO timestamp to align the
    loop's wakeup with the actual Claude quota-window boundary — that
    removes the "sleep 300s, wake up, still exhausted, repeat 50x"
    burn the performance reviewer flagged.

    Bounds:
      * Lower: ``min_sleep_s`` (default 300s).
      * Upper: 2 hours — a scheduler that reports a wildly-future reset
        time shouldn't be able to stall the loop indefinitely.
    """
    if not isinstance(scheduler_status, dict):
        return max(min_sleep_s, 60)
    next_iso = scheduler_status.get("next_window_at")
    if isinstance(next_iso, str) and next_iso:
        try:
            next_dt = datetime.fromisoformat(next_iso)
        except ValueError:
            next_dt = None
        if next_dt is not None:
            now = datetime.now(timezone.utc)
            if next_dt.tzinfo is None:
                next_dt = next_dt.replace(tzinfo=timezone.utc)
            wait_s = (next_dt - now).total_seconds()
            if wait_s > 0:
                # +30s padding to cover clock skew on the Claude side.
                return max(min_sleep_s, min(int(wait_s) + 30, 7200))
    # No usable hint — fall through to the floor. 60s floor is kept
    # below min_sleep_s so --min-sleep-s always wins.
    return max(min_sleep_s, 60)


def _done_probe(scope: str, *, results_dir: Path) -> int:
    expected = _expected_labels(scope)
    summary_path = results_dir / "_summary_live.json"
    completed, partial, missing, _ = _status_from_summary(summary_path, expected)
    body = {
        "ts": _utcnow_iso(),
        "scope": scope,
        "results_dir": str(results_dir),
        "summary_path": str(summary_path),
        "expected": expected,
        "completed": completed,
        "partial": partial,
        "missing": missing,
    }
    print(json.dumps(body, indent=2))
    return 0 if not partial and not missing else 1


def run_loop(
    *,
    scope: str,
    max_iterations: int,
    max_wall_hours: float,
    min_sleep_s: int,
    inner_timeout_s: int,
    out_dir: Path,
    extra_args: list[str],
    dry_only_consolidate: bool,
) -> int:
    expected = _expected_labels(scope)
    summary_path = out_dir / "_summary_live.json"
    logger.info(
        "ralph loop started: scope=%s, expected %d bundles=%s, summary=%s",
        scope,
        len(expected),
        expected,
        summary_path,
    )
    t_start = time.time()
    iteration = 0

    while True:
        iteration += 1
        iter_start = _utcnow_iso()
        if iteration > max_iterations:
            logger.error("hit max_iterations=%d, stopping", max_iterations)
            return 2
        elapsed_h = (time.time() - t_start) / 3600.0
        if elapsed_h > max_wall_hours:
            logger.error("hit max_wall_hours=%.1f, stopping", max_wall_hours)
            return 3

        if dry_only_consolidate:
            exit_code = 0
        else:
            exit_code = _launch_inner_runner(
                scope=scope,
                out_dir=out_dir,
                extra_args=extra_args,
                timeout_s=inner_timeout_s,
            )

        _run_consolidate(out_dir)
        completed, partial, missing, scheduler_status = _status_from_summary(
            summary_path, expected
        )
        iter_end = _utcnow_iso()

        result = IterationResult(
            iteration=iteration,
            started_at=iter_start,
            ended_at=iter_end,
            exit_code=exit_code,
            completed_labels=completed,
            partial_labels=partial,
            missing_labels=missing,
            scheduler_status=scheduler_status,
        )
        logger.info(
            "iteration %d done: exit=%d, complete=%d/%d, partial=%d, missing=%d",
            iteration,
            exit_code,
            len(completed),
            len(expected),
            len(partial),
            len(missing),
        )
        _journal_record(
            {
                "ts": iter_end,
                "event": "iteration_end",
                "iteration": iteration,
                "exit_code": exit_code,
                "completed": completed,
                "partial": partial,
                "missing": missing,
                "scheduler_status": scheduler_status,
            }
        )

        if not partial and not missing:
            logger.info(
                "all %d bundles complete — exiting successfully after %d iteration(s)",
                len(expected),
                iteration,
            )
            return 0

        sleep_s = _compute_sleep_s(
            scheduler_status=scheduler_status, min_sleep_s=min_sleep_s
        )
        logger.info(
            "sleeping %ds before next iteration (partial=%s, missing=%s)",
            sleep_s,
            partial,
            missing,
        )
        time.sleep(sleep_s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scope",
        default="full_battery",
        help="inner runner scope (default: full_battery)",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=40,
        help="hard cap on loop iterations (default: 40)",
    )
    p.add_argument(
        "--max-wall-hours",
        type=float,
        default=96.0,
        help="hard cap on total wall time (default: 96h)",
    )
    p.add_argument(
        "--min-sleep-s",
        type=int,
        default=300,
        help="minimum seconds to sleep between iterations (default: 300)",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="output directory for inner runner receipts",
    )
    p.add_argument(
        "--inner-timeout-s",
        type=int,
        default=INNER_RUN_TIMEOUT_S,
        help=(
            "max seconds a single inner run may take before the loop kills "
            f"it (default: {INNER_RUN_TIMEOUT_S}s = 6h)"
        ),
    )
    p.add_argument(
        "--done-probe",
        action="store_true",
        help="print current completion status as JSON and exit 0 iff all complete",
    )
    p.add_argument(
        "--dry-only-consolidate",
        action="store_true",
        help="skip the inner runner and only re-consolidate; useful after a crash",
    )
    p.add_argument(
        "--inner-arg",
        action="append",
        default=[],
        help="extra CLI arg passed through to run_live_hf (repeatable)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s ralph :: %(message)s",
    )

    out_dir = Path(args.out_dir).resolve()
    if args.done_probe:
        return _done_probe(args.scope, results_dir=out_dir)

    return run_loop(
        scope=args.scope,
        max_iterations=args.max_iterations,
        max_wall_hours=args.max_wall_hours,
        min_sleep_s=args.min_sleep_s,
        inner_timeout_s=args.inner_timeout_s,
        out_dir=out_dir,
        extra_args=list(args.inner_arg),
        dry_only_consolidate=args.dry_only_consolidate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
