"""Tests for the P6-B live case-study runner.

These tests *must* run end-to-end through the in-process plugin so any
regression in plugin behaviour (sublation, compaction, retrieval gate)
shows up here. They do not call any network or LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.v2.p6b.case_data import (
    ALL_CASE_STUDIES,
    CASE_DJANGO,
    CASE_PANDAS,
    CASE_REQUESTS,
    CaseStudy,
    EvidenceItem,
)
from experiments.v2.p6b.run_case_study import (
    _run_one,
    _run_with_harness,
    _run_without_harness,
    _summary,
    run_all,
)


# ---------------------------------------------------------------------------
# Case-data invariants
# ---------------------------------------------------------------------------


def test_three_case_studies_are_registered():
    assert len(ALL_CASE_STUDIES) == 3
    assert {c.case_id for c in ALL_CASE_STUDIES} == {
        "django_request_body",
        "requests_retry_adapter",
        "pandas_iterrows_dtype",
    }


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_each_case_has_at_least_one_stale_and_one_fresh(case: CaseStudy):
    stale = [e for e in case.evidence if e.stale]
    fresh = [e for e in case.evidence if not e.stale]
    assert stale, f"{case.case_id} must have ≥1 stale source to test sublation"
    assert fresh, f"{case.case_id} must have ≥1 fresh source to test the gold answer"


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_gold_substring_present_in_at_least_one_fresh_source(case: CaseStudy):
    gold = case.gold_answer_substring.lower()
    fresh_sources = [e for e in case.evidence if not e.stale]
    assert any(gold in e.content.lower() for e in fresh_sources), (
        f"gold substring {case.gold_answer_substring!r} must be present in some "
        f"non-stale evidence for case {case.case_id}"
    )


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_at_least_one_forbidden_substring_present_in_a_stale_source(case: CaseStudy):
    forbidden_norm = [p.lower() for p in case.forbidden_substrings]
    stale_sources = [e for e in case.evidence if e.stale]
    appears = any(
        p in e.content.lower()
        for p in forbidden_norm
        for e in stale_sources
    )
    assert appears, (
        f"at least one forbidden phrase must appear in a stale source for "
        f"{case.case_id} — otherwise the without-harness arm cannot exhibit "
        f"the failure mode the case study targets"
    )


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_freshness_ordering_is_monotone(case: CaseStudy):
    by_freshness = case.evidence_by_freshness()
    precisions = [e.precision for e in by_freshness]
    assert precisions == sorted(precisions, reverse=True)


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_stale_precision_is_strictly_lower_than_freshest(case: CaseStudy):
    fresh_top = max(e.precision for e in case.evidence if not e.stale)
    for stale in (e for e in case.evidence if e.stale):
        assert stale.precision < fresh_top, (
            f"{case.case_id}: stale {stale.id} has precision {stale.precision} "
            f"≥ freshest {fresh_top}; sublation would not fire"
        )


# ---------------------------------------------------------------------------
# Without-harness arm: must exhibit the documented failure mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_without_harness_anchors_on_first_evidence(case: CaseStudy):
    arm = _run_without_harness(case)
    first = case.evidence_by_seen_order()[0]
    assert arm.final_answer == first.content
    assert arm.metrics["answer_policy"] == "first_seen_wins"
    assert arm.metrics["sublations"] == 0
    assert arm.metrics["compactions"] == 0


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_without_harness_has_at_least_one_forbidden_hit(case: CaseStudy):
    arm = _run_without_harness(case)
    assert len(arm.forbidden_hits) >= 1, (
        f"{case.case_id}: the unaided arm must surface a stale-claim signal; "
        f"otherwise the case is not actually exhibiting the failure mode"
    )


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_without_harness_keeps_every_stale_source_in_context(case: CaseStudy):
    arm = _run_without_harness(case)
    n_stale = sum(1 for e in case.evidence if e.stale)
    assert arm.metrics["stale_evidence_in_context"] == n_stale


# ---------------------------------------------------------------------------
# With-harness arm: sublations fire, no stale survives compaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_with_harness_fires_at_least_one_sublation(case: CaseStudy):
    arm = _run_with_harness(case)
    assert arm.metrics["sublations"] >= 1, (
        f"{case.case_id}: at least one fresh source should supersede a stale one"
    )


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_with_harness_drops_all_stale_from_active_store(case: CaseStudy):
    arm = _run_with_harness(case)
    assert arm.metrics["stale_evidence_in_context"] == 0, (
        f"{case.case_id}: every stale source must be sublated or compacted out"
    )


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_with_harness_picks_a_fresh_source_for_final_answer(case: CaseStudy):
    arm = _run_with_harness(case)
    # The selected answer must equal one of the fresh evidence contents.
    fresh_contents = {e.content for e in case.evidence if not e.stale}
    assert arm.final_answer in fresh_contents


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_with_harness_answer_contains_gold_substring(case: CaseStudy):
    arm = _run_with_harness(case)
    assert case.gold_answer_substring.lower() in arm.final_answer.lower()


# ---------------------------------------------------------------------------
# Comparison: with-harness must dominate without-harness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_per_case_outcome_with_dominates_without(case: CaseStudy):
    outcome = _run_one(case)
    assert outcome.with_harness.answer_correct, (
        f"{case.case_id}: with-harness arm must answer correctly"
    )
    assert not outcome.without_harness.answer_correct, (
        f"{case.case_id}: without-harness arm must exhibit the failure mode"
    )
    assert (
        len(outcome.with_harness.forbidden_hits)
        <= len(outcome.without_harness.forbidden_hits)
    ), f"{case.case_id}: harness must not increase forbidden-claim hits"
    # Sublations are a *positive* metric — at least one must have fired.
    assert outcome.with_harness.metrics["sublations"] >= 1


# ---------------------------------------------------------------------------
# Summary aggregator
# ---------------------------------------------------------------------------


def test_summary_handles_empty_input():
    assert _summary([]) == {"n_cases": 0}


def test_summary_aggregates_three_cases_correctly():
    outcomes = [_run_one(c) for c in ALL_CASE_STUDIES]
    s = _summary(outcomes)
    assert s["n_cases"] == 3
    assert s["accuracy_with_harness"] == 1.0
    assert s["accuracy_without_harness"] == 0.0
    assert s["accuracy_delta"] == 1.0
    assert s["forbidden_hits_delta"] <= 0
    assert s["total_sublations_fired"] >= 3  # at least one per case
    assert s["total_compactions_fired"] == 3
    assert s["case_ids"] == [
        "django_request_body",
        "requests_retry_adapter",
        "pandas_iterrows_dtype",
    ]


# ---------------------------------------------------------------------------
# Driver: writes JSON artifacts that downstream P7 will read
# ---------------------------------------------------------------------------


def test_run_all_writes_per_case_and_summary_artifacts(tmp_path: Path):
    outcomes, summary = run_all(out_dir=tmp_path)
    assert len(outcomes) == 3
    for o in outcomes:
        path = tmp_path / f"{o.case_id}.json"
        assert path.exists()
        payload = json.loads(path.read_text())
        # Both arms recorded.
        assert payload["with_harness"]["arm"] == "with_harness"
        assert payload["without_harness"]["arm"] == "without_harness"
        # Transcripts present and non-empty.
        assert payload["with_harness"]["transcript"]
        assert payload["without_harness"]["transcript"]
        # Comparison block present.
        assert "comparison" in payload
        assert payload["comparison"]["answer_correct_with_harness"] is True
        assert payload["comparison"]["answer_correct_without_harness"] is False

    summary_path = tmp_path / "_summary.json"
    assert summary_path.exists()
    sp = json.loads(summary_path.read_text())
    assert sp["summary"]["n_cases"] == 3
    assert sp["summary"]["accuracy_delta"] == 1.0
    assert "per_case" in sp


# ---------------------------------------------------------------------------
# Plugin determinism: rerunning a case produces identical metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASE_STUDIES, ids=lambda c: c.case_id)
def test_with_harness_run_is_deterministic(case: CaseStudy):
    a = _run_with_harness(case)
    b = _run_with_harness(case)
    assert a.final_answer == b.final_answer
    assert a.answer_correct == b.answer_correct
    assert a.forbidden_hits == b.forbidden_hits
    assert a.metrics == b.metrics
