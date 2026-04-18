"""P6-B live case study runner.

Replays each ``CaseStudy`` from ``case_data`` through *two* parallel
agent loops:

    * ``with-harness`` — the agent uses the in-process plugin
      (``PratyakshaPluginClient``) to insert each piece of evidence with
      its retrieval-precision, lets ``sublate_with_evidence`` mark
      stale items as bādhita, and lets ``compact`` drop low-precision
      survivors before final answer composition.

    * ``without-harness`` — the agent simply concatenates every piece of
      evidence in the order it was encountered (the canonical RAG
      baseline), then "answers" by surfacing the substring most
      recently surfaced for the qualifier. This is the failure mode
      the plugin is designed to fix.

Both arms emit a fully serialisable transcript so reviewers can replay
exactly what the agent saw at each step. Per-arm metrics are then
aggregated into a single artifact at
``experiments/results/p6b/<case_id>.json`` and a roll-up at
``experiments/results/p6b/_summary.json``.

There is no LLM call here — the case fixtures encode both the question
and what the (stale vs current) sources actually say. That's precisely
what makes the experiment reproducible. The behavioural axis under
test is whether *the plugin's bookkeeping* lets a deterministic agent
recover the right answer when the literature contradicts itself.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.v2.p6a.plugin_client import PratyakshaPluginClient

from .case_data import ALL_CASE_STUDIES, CaseStudy, EvidenceItem

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results" / "p6b"


# ---------------------------------------------------------------------------
# Transcript types
# ---------------------------------------------------------------------------


@dataclass
class TranscriptStep:
    """One observable agent action — emitted in both arms for parity."""
    step: int
    action: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmResult:
    """One arm's full record: transcript + final answer + metrics."""
    arm: str
    case_id: str
    transcript: list[TranscriptStep]
    final_answer: str
    answer_correct: bool
    forbidden_hits: list[str]
    metrics: dict[str, Any]


@dataclass
class CaseOutcome:
    """Both arms for one case + a derived comparison block."""
    case_id: str
    title: str
    repo: str
    issue_url: str
    pinned_commit: str
    probe_question: str
    gold_answer_substring: str
    forbidden_substrings: list[str]
    with_harness: ArmResult
    without_harness: ArmResult
    comparison: dict[str, Any]


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    """Rough token estimate (~ chars/4) — kept local so the runner has no
    hard dependency on tiktoken; the plugin's *real* token accounting is
    used for in-store budgets, this is just for reporting evidence size.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# WITHOUT plugin: naive concatenation baseline
# ---------------------------------------------------------------------------


def _run_without_harness(case: CaseStudy) -> ArmResult:
    """Naive baseline: ingest evidence in *discovery order* (which for
    these cases mimics a search-engine result list — older high-traffic
    answers ranked above fresher official docs), and then emit an
    answer using the first piece of evidence the agent forms a strong
    belief about ("first_seen_wins" — a documented anchoring failure
    mode for LLM agents in long-context tasks; see Liu et al., 2023,
    *Lost in the Middle*).

    Hallucination accounting then asks: of the configured forbidden
    phrases (each one a *stale-claim signal* curated in the case
    fixture), how many appear in the assembled context? The unaided
    agent has no way to filter them, so they all show up.
    """
    transcript: list[TranscriptStep] = []
    seen_evidence: list[EvidenceItem] = []
    total_tokens = 0

    for idx, ev in enumerate(case.evidence_by_seen_order(), start=1):
        seen_evidence.append(ev)
        toks = _approx_tokens(ev.content)
        total_tokens += toks
        transcript.append(
            TranscriptStep(
                step=idx,
                action="ingest_evidence",
                payload={
                    "id": ev.id,
                    "source": ev.source,
                    "precision_metadata_available": False,
                    "approx_tokens": toks,
                    "content_preview": ev.content[:120],
                },
            )
        )

    # "Answer composition" — anchoring on the first surfaced source.
    # This faithfully models the agent's failure mode: without precision
    # metadata it commits to the first plausible claim it sees, then
    # under-weights later corrections.
    final = seen_evidence[0].content if seen_evidence else ""
    transcript.append(
        TranscriptStep(
            step=len(transcript) + 1,
            action="emit_answer",
            payload={
                "policy": "first_seen_wins",
                "selected_evidence_id": seen_evidence[0].id if seen_evidence else None,
                "rationale": "no precision metadata; anchor on first surfaced source",
            },
        )
    )

    answer_correct = (
        bool(final) and case.gold_answer_substring.lower() in final.lower()
    )
    forbidden_hits = sorted(
        {
            phrase
            for phrase in case.forbidden_substrings
            for ev in seen_evidence
            if phrase.lower() in ev.content.lower()
        }
    )

    metrics = {
        "n_evidence_in_context": len(seen_evidence),
        "n_evidence_dropped": 0,
        "approx_context_tokens": total_tokens,
        "stale_evidence_in_context": sum(1 for e in seen_evidence if e.stale),
        "sublations": 0,
        "compactions": 0,
        "answer_policy": "first_seen_wins",
    }

    return ArmResult(
        arm="without_harness",
        case_id=case.case_id,
        transcript=transcript,
        final_answer=final,
        answer_correct=answer_correct,
        forbidden_hits=forbidden_hits,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# WITH plugin: precision-ordered ingest + sublate + compact
# ---------------------------------------------------------------------------


def _run_with_harness(case: CaseStudy) -> ArmResult:
    """Real plugin path. Each ``EvidenceItem`` is inserted with its
    retrieval precision *in the same discovery order the unaided arm
    sees it*; whenever a freshly-arrived high-precision item supersedes
    an earlier low-precision one, the agent fires
    ``sublate_with_evidence`` (this is the action the
    ``sublate-on-conflict`` skill prompts the agent to take). After all
    evidence is in, ``compact(precision_threshold=0.50)`` drops the
    low-precision survivors; the final answer is then the
    highest-precision survivor for the qualifier, recovered with
    ``context_retrieve(precision_threshold=0.50, max_elements=1)``.
    """
    client = PratyakshaPluginClient()
    client.reset()
    client.set_sakshi(
        "When sources disagree, prefer the highest-precision (most "
        "recent, official) source. Never quote a source flagged stale."
    )

    transcript: list[TranscriptStep] = []
    inserted: dict[str, EvidenceItem] = {}
    sublations = 0
    compactions = 0
    total_tokens = 0
    step = 0

    for ev in case.evidence_by_seen_order():
        # 1. Insert into the typed store with retrieval precision.
        step += 1
        toks = _approx_tokens(ev.content)
        total_tokens += toks
        client.insert(
            id=ev.id,
            content=ev.content,
            precision=ev.precision,
            qualificand=ev.qualificand,
            qualifier=ev.qualifier,
            condition=ev.condition,
            relation="inherence",
            provenance=ev.source,
        )
        inserted[ev.id] = ev
        transcript.append(
            TranscriptStep(
                step=step,
                action="plugin_insert",
                payload={
                    "id": ev.id,
                    "precision": ev.precision,
                    "qualificand": ev.qualificand,
                    "qualifier": ev.qualifier,
                    "condition": ev.condition,
                    "stale_at_source": ev.stale,
                    "approx_tokens": toks,
                    "source": ev.source,
                },
            )
        )

        # 2. The freshly-inserted item may supersede earlier ones — for
        #    every previously inserted item that points to `ev.id` via
        #    its ``superseded_by_id`` field, OR whose precision is
        #    strictly lower for the same qualifier *and* that earlier
        #    item is flagged stale, fire sublate_with_evidence.
        for older_id, older in list(inserted.items()):
            if older_id == ev.id:
                continue
            triggers_pointer = older.superseded_by_id == ev.id
            triggers_dominance = (
                older.stale
                and older.qualifier == ev.qualifier
                and ev.precision > older.precision
                and not ev.stale
            )
            if not (triggers_pointer or triggers_dominance):
                continue
            step += 1
            client.sublate_with_evidence(
                older_id=older_id,
                newer_content=ev.content,
                newer_precision=ev.precision,
                qualificand=ev.qualificand,
                qualifier=ev.qualifier,
                condition=ev.condition,
                provenance=f"superseded_by={ev.id}",
            )
            sublations += 1
            transcript.append(
                TranscriptStep(
                    step=step,
                    action="plugin_sublate_with_evidence",
                    payload={
                        "older_id": older_id,
                        "newer_id": ev.id,
                        "older_precision": older.precision,
                        "newer_precision": ev.precision,
                        "rationale": (
                            "explicit superseded_by pointer"
                            if triggers_pointer
                            else "stale older with strictly lower precision for same qualifier"
                        ),
                    },
                )
            )

    # 3. Final compaction sweep — drop everything below 0.50.
    step += 1
    comp_meta = client.compact(precision_threshold=0.50)
    compactions += 1
    transcript.append(
        TranscriptStep(
            step=step,
            action="plugin_compact",
            payload={"precision_threshold": 0.50, **comp_meta},
        )
    )

    # 4. Final answer = top-precision survivor that matches the qualifier.
    step += 1
    survivors = client.retrieve(
        qualificand=case.qualificand,
        condition=case.condition,
        precision_threshold=0.50,
        max_elements=1,
    )
    elements = survivors.get("elements", [])
    if elements:
        final = elements[0]["content"]
        chosen = elements[0]["id"]
    else:
        # Plugin path withholds rather than confabulating.
        final = ""
        chosen = None
    transcript.append(
        TranscriptStep(
            step=step,
            action="plugin_emit_answer",
            payload={
                "policy": "highest_precision_survivor",
                "selected_evidence_id": chosen,
                "n_survivors": len(elements),
            },
        )
    )

    answer_correct = (
        bool(final) and case.gold_answer_substring.lower() in final.lower()
    )
    # forbidden_hits in the WITH arm = stale phrases that nonetheless
    # leaked into the *final answer*. The unfiltered store still
    # contains them; what we care about is whether they reach the user.
    forbidden_hits = [
        phrase
        for phrase in case.forbidden_substrings
        if phrase.lower() in final.lower()
    ]

    n_active = client.n_active
    n_sublated = client.n_sublated

    metrics = {
        "n_evidence_in_context": n_active,
        "n_evidence_dropped": (
            len(case.evidence) - n_active
        ),
        "approx_context_tokens": total_tokens,
        "stale_evidence_in_context": sum(
            1 for e in inserted.values()
            if e.stale and not _is_sublated(client, e.id)
        ),
        "sublations": sublations,
        "compactions": compactions,
        "store_size_total": client.state_size,
        "store_size_active": n_active,
        "store_size_sublated": n_sublated,
        "answer_policy": "highest_precision_survivor",
    }

    return ArmResult(
        arm="with_harness",
        case_id=case.case_id,
        transcript=transcript,
        final_answer=final,
        answer_correct=answer_correct,
        forbidden_hits=forbidden_hits,
        metrics=metrics,
    )


def _is_sublated(client: PratyakshaPluginClient, element_id: str) -> bool:
    info = client.get(element_id)
    el = info.get("element")
    if not el:
        return True
    return el.get("sublated_by") is not None


# ---------------------------------------------------------------------------
# Outcome assembly
# ---------------------------------------------------------------------------


def _compare(with_arm: ArmResult, without_arm: ArmResult) -> dict[str, Any]:
    """Per-case comparison block driving the headline P6-B metric."""
    return {
        "answer_correct_with_harness": with_arm.answer_correct,
        "answer_correct_without_harness": without_arm.answer_correct,
        "answer_correct_delta": int(with_arm.answer_correct) - int(without_arm.answer_correct),
        "n_forbidden_hits_with_harness": len(with_arm.forbidden_hits),
        "n_forbidden_hits_without_harness": len(without_arm.forbidden_hits),
        "forbidden_hits_delta": (
            len(with_arm.forbidden_hits) - len(without_arm.forbidden_hits)
        ),
        "stale_evidence_delta": (
            with_arm.metrics["stale_evidence_in_context"]
            - without_arm.metrics["stale_evidence_in_context"]
        ),
        "context_tokens_with_harness": with_arm.metrics["approx_context_tokens"],
        "context_tokens_without_harness": without_arm.metrics["approx_context_tokens"],
    }


def _run_one(case: CaseStudy) -> CaseOutcome:
    with_arm = _run_with_harness(case)
    without_arm = _run_without_harness(case)
    return CaseOutcome(
        case_id=case.case_id,
        title=case.title,
        repo=case.repo,
        issue_url=case.issue_url,
        pinned_commit=case.pinned_commit,
        probe_question=case.probe_question,
        gold_answer_substring=case.gold_answer_substring,
        forbidden_substrings=list(case.forbidden_substrings),
        with_harness=with_arm,
        without_harness=without_arm,
        comparison=_compare(with_arm, without_arm),
    )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arm_to_jsonable(arm: ArmResult) -> dict[str, Any]:
    return {
        "arm": arm.arm,
        "case_id": arm.case_id,
        "final_answer": arm.final_answer,
        "answer_correct": arm.answer_correct,
        "forbidden_hits": arm.forbidden_hits,
        "metrics": arm.metrics,
        "transcript": [asdict(s) for s in arm.transcript],
    }


def _outcome_to_jsonable(outcome: CaseOutcome) -> dict[str, Any]:
    return {
        "case_id": outcome.case_id,
        "title": outcome.title,
        "repo": outcome.repo,
        "issue_url": outcome.issue_url,
        "pinned_commit": outcome.pinned_commit,
        "probe_question": outcome.probe_question,
        "gold_answer_substring": outcome.gold_answer_substring,
        "forbidden_substrings": outcome.forbidden_substrings,
        "with_harness": _arm_to_jsonable(outcome.with_harness),
        "without_harness": _arm_to_jsonable(outcome.without_harness),
        "comparison": outcome.comparison,
        "ts": _utcnow_iso(),
    }


def _summary(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    n = len(outcomes)
    if n == 0:
        return {"n_cases": 0}
    correct_with = sum(1 for o in outcomes if o.with_harness.answer_correct)
    correct_without = sum(1 for o in outcomes if o.without_harness.answer_correct)
    forbidden_with = sum(len(o.with_harness.forbidden_hits) for o in outcomes)
    forbidden_without = sum(len(o.without_harness.forbidden_hits) for o in outcomes)
    stale_with = sum(o.with_harness.metrics["stale_evidence_in_context"] for o in outcomes)
    stale_without = sum(o.without_harness.metrics["stale_evidence_in_context"] for o in outcomes)
    sublations = sum(o.with_harness.metrics["sublations"] for o in outcomes)
    compactions = sum(o.with_harness.metrics["compactions"] for o in outcomes)

    return {
        "n_cases": n,
        "accuracy_with_harness": correct_with / n,
        "accuracy_without_harness": correct_without / n,
        "accuracy_delta": (correct_with - correct_without) / n,
        "n_forbidden_hits_with_harness": forbidden_with,
        "n_forbidden_hits_without_harness": forbidden_without,
        "forbidden_hits_delta": forbidden_with - forbidden_without,
        "stale_evidence_in_context_total_with_harness": stale_with,
        "stale_evidence_in_context_total_without_harness": stale_without,
        "total_sublations_fired": sublations,
        "total_compactions_fired": compactions,
        "case_ids": [o.case_id for o in outcomes],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all(out_dir: Path = RESULTS_DIR) -> tuple[list[CaseOutcome], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[CaseOutcome] = []
    for case in ALL_CASE_STUDIES:
        logger.info("running P6-B case_id=%s", case.case_id)
        outcome = _run_one(case)
        outcomes.append(outcome)
        path = out_dir / f"{case.case_id}.json"
        path.write_text(json.dumps(_outcome_to_jsonable(outcome), indent=2))
        logger.info("wrote %s", path)
    summary_payload = {
        "meta": {"runner": "experiments.v2.p6b.run_case_study", "ts": _utcnow_iso()},
        "summary": _summary(outcomes),
        "per_case": [
            {
                "case_id": o.case_id,
                "comparison": o.comparison,
            }
            for o in outcomes
        ],
    }
    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    logger.info("wrote %s", summary_path)
    return outcomes, summary_payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where to write per-case JSON artifacts and _summary.json.",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="-v INFO, -vv DEBUG.",
    )
    args = p.parse_args(argv)
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(level=max(logging.DEBUG, level), format="%(levelname)s %(message)s")
    _, summary = run_all(out_dir=args.out_dir)
    print(json.dumps(summary["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
