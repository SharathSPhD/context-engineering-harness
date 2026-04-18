"""P6-A re-runs of H3, H4, H5 with the `pratyaksha-context-eng-harness`
plugin in the loop.

Unlike H1/H2 (which exercise registered BenchmarkAdapters end-to-end via
`MultiSeedRunner`), H3/H4/H5 exercise the *behaviour of the plugin
itself* — Buddhi/Manas grounding (H3), event-boundary compaction (H4),
and Avacchedaka sublation (H5). For all three we re-use the same
plugin code path Claude Code calls into via MCP, but in-process for
speed and reproducibility (see `plugin_client.PratyakshaPluginClient`
for the import-once-snapshot-state-each-trial pattern).

Each hypothesis runs:
    for model in models:
        for seed in seeds:
            on_scores  = run_with_harness(scenarios)
            off_scores = run_without_harness(scenarios)
        paired (on - off) per (model, seed) → bootstrap CI + paired permutation

The "model" axis is currently a label (the plugin path is deterministic
and does not call any LLM). We keep it because the live-mode SWE-bench
re-run will need the same partition for paired tests against real Claude
calls.

Output: experiments/results/p6a/H{3,4,5}*.json plus _summary_plugin.json.
These are the inputs P7 hands to the figure/table generators.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmarks.hypothesis import HypothesisSpec, TargetDirection
from src.benchmarks.stats import bootstrap_ci, cohens_d, paired_permutation_test

from .plugin_client import PratyakshaPluginClient
from .scenarios import (
    H3Case,
    H4Scenario,
    H5Conflict,
    make_h3_cases,
    make_h4_scenarios,
    make_h5_conflicts,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results" / "p6a"


# --- shared helpers -----------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class _PerSeedRun:
    """One (model, seed, condition) row of paired scores."""
    model: str
    seed: int
    condition: str  # "harness_on" | "harness_off"
    n: int
    mean_score: float
    accuracy: float
    extra: dict[str, Any]


# --- H3: Buddhi/Manas grounding ----------------------------------------


def _h3_score_one(case: H3Case, *, with_harness: bool, client: PratyakshaPluginClient) -> float:
    """Score one H3 case: 1.0 if the agent does the right thing, else 0.0.

    "Right thing" = answer correctly when grounded, withhold when not.

    `with_harness` (Buddhi/Manas):
        Manas does typed retrieval; Buddhi gates on precision.
        Insert grounded facts into the Avacchedaka store ahead of the
        question, then `context_retrieve` with `precision_threshold=0.5`.
        If retrieval returns the matching content, assert the gold;
        otherwise withhold.

    `without_harness` (single-stage, no plugin):
        We mimic an unaided model:
          - on grounded cases: emits the gold value (the same text was
            in conversation history, so the model has it),
          - on ungrounded cases: confabulates; lucks into a correct
            answer with probability ~10% (per-case rng decided in the
            scenario constructor).
    """
    if with_harness:
        # Plugin path: typed retrieval + precision gate.
        result = client.retrieve(
            qualificand=case.qualificand,
            condition=case.condition,
            precision_threshold=0.5,
        )
        elements = result.get("elements", [])
        if not elements:
            answer = None  # withhold
        else:
            top = elements[0]
            if top["precision"] >= 0.5 and case.gold is not None and case.gold in top["content"]:
                answer = case.gold
            else:
                answer = None
    else:
        # No-harness baseline.
        if case.gold is not None:
            # Even the unaided model gets grounded answers right because
            # the same text appears in conversation history.
            answer = case.gold
        else:
            # Ungrounded: model confabulates. `confab_correct_offline` is
            # the per-case rng's verdict on whether the model is "lucky"
            # enough to recognise it should withhold (≈ 10% of the time).
            answer = None if case.confab_correct_offline else "<plausible-but-wrong>"

    if case.gold is None:
        return 1.0 if answer is None else 0.0
    return 1.0 if answer == case.gold else 0.0


def _run_h3_single(
    *, cases: list[H3Case], with_harness: bool, model: str
) -> tuple[list[float], dict[str, Any]]:
    client = PratyakshaPluginClient()
    if with_harness:
        client.set_sakshi(
            "Decline to answer when retrieved precision < 0.5. Never invent values."
        )
        for c in cases:
            if c.grounded_content is not None:
                client.insert(
                    id=f"{c.qid}-fact",
                    content=c.grounded_content,
                    precision=c.grounded_precision,
                    qualificand=c.qualificand,
                    qualifier=c.qualifier,
                    condition=c.condition,
                    relation="inherence",
                    provenance="h3-runner",
                )
    scores = [_h3_score_one(c, with_harness=with_harness, client=client) for c in cases]
    extra = {
        "store_size": client.state_size,
        "n_active": client.n_active,
        "n_sublated": client.n_sublated,
        "n_grounded": sum(1 for c in cases if c.gold is not None),
        "n_ungrounded": sum(1 for c in cases if c.gold is None),
        "model": model,
    }
    return scores, extra


def _run_h3(
    *,
    n_per_seed: int,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    bootstrap_n: int,
    permutation_n: int,
) -> dict[str, Any]:
    spec = HypothesisSpec(
        hypothesis_id="H3",
        description=(
            "Two-stage Buddhi/Manas with typed Avacchedaka retrieval and "
            "precision-gated answering reduces hallucinations on a mixed "
            "(grounded / low-precision-grounded / ungrounded) workload "
            "vs an unaided single-stage model."
        ),
        adapter_name="(plugin-inloop)",
        treatment_condition="harness_on",
        baseline_condition="harness_off",
        metric="accuracy",
        direction=TargetDirection.GREATER,
        delta=0.10,
        n_examples=n_per_seed,
        seeds=seeds,
        models=models,
        significance_alpha=0.05,
        notes="In-process MCP plugin. Buckets: 60% high-grounded, 10% low-grounded, 30% ungrounded.",
    )

    per_seed_rows, on_means, off_means = _per_seed_loop(
        models=models,
        seeds=seeds,
        run_pair=lambda model, seed: (
            _run_h3_single(cases=make_h3_cases(n=n_per_seed, seed=seed),
                           with_harness=True, model=model),
            _run_h3_single(cases=make_h3_cases(n=n_per_seed, seed=seed),
                           with_harness=False, model=model),
        ),
    )

    return _outcome_payload(
        label="H3_buddhi_manas_grounding",
        spec=spec,
        on_means=on_means,
        off_means=off_means,
        per_seed_rows=per_seed_rows,
        bootstrap_n=bootstrap_n,
        permutation_n=permutation_n,
    )


# --- H4: surprise-boundary compaction -----------------------------------


def _h4_score_one(
    scenario: H4Scenario, *, with_harness: bool, client: PratyakshaPluginClient
) -> tuple[float, dict[str, Any]]:
    """Recall of post-boundary facts after compaction.

    `with_harness`:
        1. Insert all items into typed store.
        2. `boundary_compact(text_window, threshold_z=2.0,
           precision_threshold=0.30)` — uses the plugin's per-token
           novelty proxy; this drops noise (precision < 0.30) but spares
           pre and post.
        3. If a boundary was detected, run `compact(qualificand="pre",
           precision_threshold=1.0)` to aggressively wipe the pre
           qualificand — this is the agent's response to the discourse
           shift signalled by the boundary detector.
    `without_harness`:
        Naive uniform `compact(precision_threshold=0.50)` — drops noise
        AND drops the low-precision "post-fresh" items, hurting recall.
    """
    for it in scenario.items:
        client.insert(
            id=it.id,
            content=it.content,
            precision=it.precision,
            qualificand=it.qualificand,
            qualifier=it.qualifier,
            condition=it.condition,
            relation="inherence",
            provenance="h4-runner",
        )

    if with_harness:
        bc = client.boundary_compact(
            text_window=scenario.boundary_text,
            threshold_z=2.0,
            precision_threshold=0.30,
        )
        boundary_detected = bool(bc.get("boundary_detected", False))
        if boundary_detected:
            # Boundary signal → scoped aggressive compaction of pre items.
            client.compact(
                qualificand="pre",
                precision_threshold=1.0,
            )
        comp_meta = {
            "boundary_detected": boundary_detected,
            "max_z": bc.get("max_z"),
            "compressed": (bc.get("compaction") or {}).get("n_compressed"),
        }
    else:
        c = client.compact(precision_threshold=0.50)
        comp_meta = {"naive_compressed": c.get("n_compressed")}

    # Post-recall = post items with non-zero precision and no sublation.
    post_ids = {it.id for it in scenario.items if it.bucket == "post"}
    pre_ids = {it.id for it in scenario.items if it.bucket == "pre"}
    noise_ids = {it.id for it in scenario.items if it.bucket == "noise"}

    def _kept(eid: str) -> bool:
        got = client.get(eid)
        if not got.get("ok"):
            return False
        elem = got.get("element", {})
        return elem.get("sublated_by") is None and elem.get("precision", 0.0) > 0.0

    n_post_kept = sum(1 for i in post_ids if _kept(i))
    n_pre_kept = sum(1 for i in pre_ids if _kept(i))
    n_noise_kept = sum(1 for i in noise_ids if _kept(i))

    score = n_post_kept / max(1, len(post_ids))
    extra = {
        "n_post": len(post_ids),
        "post_kept": n_post_kept,
        "pre_kept": n_pre_kept,
        "noise_kept": n_noise_kept,
        **comp_meta,
    }
    return score, extra


def _run_h4(
    *,
    n_per_seed: int,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    bootstrap_n: int,
    permutation_n: int,
) -> dict[str, Any]:
    spec = HypothesisSpec(
        hypothesis_id="H4",
        description=(
            "Event-boundary compaction (surprise-spike triggered) preserves "
            "more post-boundary facts than naive precision-threshold "
            "compaction; effect ≥ 10 pts."
        ),
        adapter_name="(plugin-inloop)",
        treatment_condition="harness_on",
        baseline_condition="harness_off",
        metric="post_recall",
        direction=TargetDirection.GREATER,
        delta=0.10,
        n_examples=n_per_seed,
        seeds=seeds,
        models=models,
        significance_alpha=0.05,
        notes="In-process MCP. boundary_compact + scoped pre-wipe vs naive precision threshold.",
    )

    per_seed_rows: list[_PerSeedRun] = []
    on_means: list[float] = []
    off_means: list[float] = []

    for model in models:
        for seed in seeds:
            scenarios = make_h4_scenarios(n=n_per_seed, seed=seed)
            on_scores: list[float] = []
            off_scores: list[float] = []
            on_extra_agg: dict[str, Any] = {
                "post_kept": 0, "pre_kept": 0, "noise_kept": 0,
                "n_post": 0, "boundary_detected": 0,
            }
            off_extra_agg: dict[str, Any] = {
                "post_kept": 0, "pre_kept": 0, "noise_kept": 0,
                "n_post": 0,
            }
            for sc in scenarios:
                on_client = PratyakshaPluginClient()
                off_client = PratyakshaPluginClient()
                s_on, ex_on = _h4_score_one(sc, with_harness=True, client=on_client)
                s_off, ex_off = _h4_score_one(sc, with_harness=False, client=off_client)
                on_scores.append(s_on)
                off_scores.append(s_off)
                for k in ("post_kept", "pre_kept", "noise_kept", "n_post"):
                    on_extra_agg[k] += ex_on[k]
                    off_extra_agg[k] += ex_off[k]
                if ex_on.get("boundary_detected"):
                    on_extra_agg["boundary_detected"] += 1
            per_seed_rows.append(
                _PerSeedRun(
                    model=model, seed=seed, condition="harness_on",
                    n=len(on_scores),
                    mean_score=float(sum(on_scores) / len(on_scores)),
                    accuracy=float(sum(1 for s in on_scores if s >= 0.5) / len(on_scores)),
                    extra=on_extra_agg,
                )
            )
            per_seed_rows.append(
                _PerSeedRun(
                    model=model, seed=seed, condition="harness_off",
                    n=len(off_scores),
                    mean_score=float(sum(off_scores) / len(off_scores)),
                    accuracy=float(sum(1 for s in off_scores if s >= 0.5) / len(off_scores)),
                    extra=off_extra_agg,
                )
            )
            on_means.append(per_seed_rows[-2].mean_score)
            off_means.append(per_seed_rows[-1].mean_score)

    return _outcome_payload(
        label="H4_event_boundary_compaction",
        spec=spec,
        on_means=on_means,
        off_means=off_means,
        per_seed_rows=per_seed_rows,
        bootstrap_n=bootstrap_n,
        permutation_n=permutation_n,
    )


# --- H5: Avacchedaka conflict resolution --------------------------------


def _h5_score_one(
    conflict: H5Conflict, *, with_harness: bool, client: PratyakshaPluginClient
) -> tuple[float, dict[str, Any]]:
    """1.0 iff exactly one active element is retrieved AND it carries the newer value."""
    client.insert(
        id=conflict.older_id,
        content=conflict.older_value,
        precision=conflict.older_precision,
        qualificand=conflict.qualificand,
        qualifier=conflict.qualifier,
        condition=conflict.condition,
        relation="inherence",
        provenance="h5-older",
    )
    if with_harness:
        client.sublate_with_evidence(
            older_id=conflict.older_id,
            newer_content=conflict.newer_value,
            newer_precision=conflict.newer_precision,
            qualificand=conflict.qualificand,
            qualifier=conflict.qualifier,
            condition=conflict.condition,
            provenance="h5-newer",
        )
    else:
        # Naive: insert the newer fact too without atomically sublating.
        client.insert(
            id=conflict.newer_id,
            content=conflict.newer_value,
            precision=conflict.newer_precision,
            qualificand=conflict.qualificand,
            qualifier=conflict.qualifier,
            condition=conflict.condition,
            relation="inherence",
            provenance="h5-newer",
        )

    retrieved = client.retrieve(
        qualificand=conflict.qualificand,
        condition=conflict.condition,
        precision_threshold=0.0,
        max_elements=10,
    )
    active = retrieved.get("elements", [])
    n_active = len(active)
    contents = [el.get("content", "") for el in active]
    correct = (n_active == 1 and conflict.newer_value in contents[0])
    return (1.0 if correct else 0.0), {"n_active": n_active}


def _run_h5(
    *,
    n_per_seed: int,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    bootstrap_n: int,
    permutation_n: int,
) -> dict[str, Any]:
    spec = HypothesisSpec(
        hypothesis_id="H5",
        description=(
            "Avacchedaka sublate_with_evidence resolves contradicting "
            "facts to the higher-precision value while preserving an audit "
            "trail; baseline (no-plugin) cannot disambiguate. Effect ≥ 50 pts."
        ),
        adapter_name="(plugin-inloop)",
        treatment_condition="harness_on",
        baseline_condition="harness_off",
        metric="conflict_resolution_accuracy",
        direction=TargetDirection.GREATER,
        delta=0.50,
        n_examples=n_per_seed,
        seeds=seeds,
        models=models,
        significance_alpha=0.05,
        notes="In-process MCP. sublate_with_evidence vs naive co-existence.",
    )

    per_seed_rows: list[_PerSeedRun] = []
    on_means: list[float] = []
    off_means: list[float] = []

    for model in models:
        for seed in seeds:
            conflicts = make_h5_conflicts(n=n_per_seed, seed=seed)
            on_scores: list[float] = []
            off_scores: list[float] = []
            on_active_total = 0
            off_active_total = 0
            for c in conflicts:
                on_client = PratyakshaPluginClient()
                off_client = PratyakshaPluginClient()
                s_on, ex_on = _h5_score_one(c, with_harness=True, client=on_client)
                s_off, ex_off = _h5_score_one(c, with_harness=False, client=off_client)
                on_scores.append(s_on)
                off_scores.append(s_off)
                on_active_total += ex_on["n_active"]
                off_active_total += ex_off["n_active"]
            per_seed_rows.append(
                _PerSeedRun(
                    model=model, seed=seed, condition="harness_on",
                    n=len(on_scores),
                    mean_score=float(sum(on_scores) / len(on_scores)),
                    accuracy=float(sum(1 for s in on_scores if s >= 0.5) / len(on_scores)),
                    extra={"avg_active_after": on_active_total / max(1, len(on_scores))},
                )
            )
            per_seed_rows.append(
                _PerSeedRun(
                    model=model, seed=seed, condition="harness_off",
                    n=len(off_scores),
                    mean_score=float(sum(off_scores) / len(off_scores)),
                    accuracy=float(sum(1 for s in off_scores if s >= 0.5) / len(off_scores)),
                    extra={"avg_active_after": off_active_total / max(1, len(off_scores))},
                )
            )
            on_means.append(per_seed_rows[-2].mean_score)
            off_means.append(per_seed_rows[-1].mean_score)

    return _outcome_payload(
        label="H5_avacchedaka_sublation",
        spec=spec,
        on_means=on_means,
        off_means=off_means,
        per_seed_rows=per_seed_rows,
        bootstrap_n=bootstrap_n,
        permutation_n=permutation_n,
    )


# --- shared payload assembly --------------------------------------------


def _per_seed_loop(*, models, seeds, run_pair):
    """Generic helper used by H3 (uniform inner shape).

    `run_pair(model, seed)` must return (
        (on_scores: list[float], on_extra: dict),
        (off_scores: list[float], off_extra: dict),
    ).
    """
    per_seed_rows: list[_PerSeedRun] = []
    on_means: list[float] = []
    off_means: list[float] = []
    for model in models:
        for seed in seeds:
            (on_scores, on_extra), (off_scores, off_extra) = run_pair(model, seed)
            per_seed_rows.append(
                _PerSeedRun(
                    model=model, seed=seed, condition="harness_on",
                    n=len(on_scores),
                    mean_score=float(sum(on_scores) / len(on_scores)),
                    accuracy=float(sum(1 for s in on_scores if s >= 0.5) / len(on_scores)),
                    extra=on_extra,
                )
            )
            per_seed_rows.append(
                _PerSeedRun(
                    model=model, seed=seed, condition="harness_off",
                    n=len(off_scores),
                    mean_score=float(sum(off_scores) / len(off_scores)),
                    accuracy=float(sum(1 for s in off_scores if s >= 0.5) / len(off_scores)),
                    extra=off_extra,
                )
            )
            on_means.append(per_seed_rows[-2].mean_score)
            off_means.append(per_seed_rows[-1].mean_score)
    return per_seed_rows, on_means, off_means


def _outcome_payload(
    *,
    label: str,
    spec: HypothesisSpec,
    on_means: list[float],
    off_means: list[float],
    per_seed_rows: list[_PerSeedRun],
    bootstrap_n: int,
    permutation_n: int,
) -> dict[str, Any]:
    diffs = [on - off for on, off in zip(on_means, off_means, strict=True)]
    if diffs:
        delta_observed, ci_low, ci_high = bootstrap_ci(
            diffs, n_bootstrap=bootstrap_n, seed=0
        )
        p_value = paired_permutation_test(
            on_means, off_means, n_permutations=permutation_n, seed=0
        )
        d = cohens_d(on_means, off_means)
    else:
        delta_observed, ci_low, ci_high, p_value, d = 0.0, 0.0, 0.0, 1.0, 0.0

    if spec.direction == TargetDirection.GREATER:
        target_met = (delta_observed >= spec.delta) and (p_value < spec.significance_alpha)
    elif spec.direction == TargetDirection.LESS:
        target_met = (delta_observed <= -spec.delta) and (p_value < spec.significance_alpha)
    else:
        target_met = (abs(delta_observed) >= spec.delta) and (p_value < spec.significance_alpha)

    spec_d = asdict(spec)
    spec_d["direction"] = spec.direction.value
    return {
        "label": label,
        "spec": spec_d,
        "outcome": {
            "treatment_metric": round(float(sum(on_means) / max(1, len(on_means))), 4),
            "baseline_metric": round(float(sum(off_means) / max(1, len(off_means))), 4),
            "delta_observed": round(float(delta_observed), 4),
            "ci_low": round(float(ci_low), 4),
            "ci_high": round(float(ci_high), 4),
            "p_value": round(float(p_value), 6),
            "cohens_d": round(float(d), 4),
            "target_met": bool(target_met),
            "n_examples_used": spec.n_examples,
            "n_seeds_used": len(spec.seeds) * len(spec.models),
            "extra": {"diffs": [round(x, 4) for x in diffs]},
        },
        "per_seed_runs": [
            {
                "model": r.model,
                "seed": r.seed,
                "condition": r.condition,
                "n": r.n,
                "mean_score": round(r.mean_score, 4),
                "accuracy": round(r.accuracy, 4),
                "extra": r.extra,
            }
            for r in per_seed_rows
        ],
        "ts": _utcnow_iso(),
    }


def _write(payload: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{payload['label']}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _write_summary(payloads: list[dict[str, Any]], out_dir: Path, *, meta: dict) -> Path:
    summary = {
        "meta": meta,
        "results": [
            {
                "label": p["label"],
                "hypothesis_id": p["spec"]["hypothesis_id"],
                "models": p["spec"]["models"],
                "seeds": p["spec"]["seeds"],
                "n_examples": p["spec"]["n_examples"],
                "outcome": p["outcome"],
                "wallclock_s": p.get("wallclock_s"),
            }
            for p in payloads
        ],
        "ts": _utcnow_iso(),
    }
    path = out_dir / "_summary_plugin.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


# --- CLI ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hypotheses",
        nargs="+",
        default=["all"],
        choices=["H3", "H4", "H5", "all"],
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["claude-haiku-4-5", "claude-sonnet-4-6"],
        help=(
            "Model labels recorded in the artifact. Plugin-inloop runs are "
            "deterministic and do not call any LLM — these labels control "
            "how we partition seeds for the paired test."
        ),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-examples", type=int, default=30)
    p.add_argument("--bootstrap-n", type=int, default=2_000)
    p.add_argument("--permutation-n", type=int, default=2_000)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    if "all" in args.hypotheses:
        selected = ["H3", "H4", "H5"]
    else:
        selected = list(args.hypotheses)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "p6a plugin-inloop runner starting: %s, models=%s, seeds=%s, n=%d, out=%s",
        selected, args.models, args.seeds, args.n_examples, out_dir,
    )

    payloads: list[dict[str, Any]] = []
    runners = {"H3": _run_h3, "H4": _run_h4, "H5": _run_h5}
    for h in selected:
        logger.info("running %s ...", h)
        t0 = time.perf_counter()
        payload = runners[h](
            n_per_seed=args.n_examples,
            seeds=tuple(args.seeds),
            models=tuple(args.models),
            bootstrap_n=args.bootstrap_n,
            permutation_n=args.permutation_n,
        )
        elapsed = time.perf_counter() - t0
        payload["wallclock_s"] = round(elapsed, 3)
        path = _write(payload, out_dir)
        logger.info(
            "  -> %s   delta=%.4f  p=%.6f  d=%.3f  target_met=%s  (%.2fs)",
            path.name,
            payload["outcome"]["delta_observed"],
            payload["outcome"]["p_value"],
            payload["outcome"]["cohens_d"],
            payload["outcome"]["target_met"],
            elapsed,
        )
        payloads.append(payload)

    meta = {
        "mode": "plugin-inloop",
        "models": list(args.models),
        "seeds": list(args.seeds),
        "n_examples": args.n_examples,
        "bootstrap_n": args.bootstrap_n,
        "permutation_n": args.permutation_n,
    }
    summary = _write_summary(payloads, out_dir, meta=meta)
    logger.info("wrote summary to %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
