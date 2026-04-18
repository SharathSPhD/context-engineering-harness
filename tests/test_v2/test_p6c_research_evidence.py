"""Tests for the P6-C synthetic research-trail generator."""
from __future__ import annotations

import pytest

from experiments.v2.p6c.research_evidence import (
    ResearchSnippet,
    generate_research_trail,
)


_INST = dict(
    instance_id="synth_swe_00_0001_abc",
    repo="synthorg/utils",
    file_path="synth_utils/strings.py",
    issue_summary="snake_to_camel returns wrong case for first segment",
)


def test_trail_is_deterministic_per_seed():
    a = generate_research_trail(seed=7, **_INST)
    b = generate_research_trail(seed=7, **_INST)
    assert [s.id for s in a] == [s.id for s in b]
    assert [s.precision for s in a] == [s.precision for s in b]
    assert [s.content for s in a] == [s.content for s in b]


def test_trail_diverges_across_seeds():
    a = generate_research_trail(seed=0, **_INST)
    b = generate_research_trail(seed=99, **_INST)
    assert (
        [s.precision for s in a] != [s.precision for s in b]
        or [s.id for s in a] != [s.id for s in b]
    )


def test_default_trail_has_two_stale_and_two_fresh():
    trail = generate_research_trail(seed=0, **_INST)
    assert len(trail) == 4
    assert sum(1 for s in trail if s.stale) == 2
    assert sum(1 for s in trail if not s.stale) == 2


def test_every_stale_has_strictly_lower_precision_than_every_fresh():
    trail = generate_research_trail(seed=42, **_INST)
    fresh_min = min(s.precision for s in trail if not s.stale)
    stale_max = max(s.precision for s in trail if s.stale)
    assert stale_max < fresh_min, (
        f"sublation can only fire if every stale.precision < every fresh.precision; "
        f"got stale_max={stale_max}, fresh_min={fresh_min}"
    )


def test_every_stale_points_at_a_fresh_id_via_superseded_by():
    trail = generate_research_trail(seed=3, **_INST)
    fresh_ids = {s.id for s in trail if not s.stale}
    for stale in (s for s in trail if s.stale):
        assert stale.superseded_by_id in fresh_ids


def test_stale_snippets_reference_a_wrong_path():
    trail = generate_research_trail(seed=11, **_INST)
    for stale in (s for s in trail if s.stale):
        assert "_legacy_" in stale.content, (
            "the stale snippet's wrong-path signal must be present so the "
            "patch simulator's anchoring failure is exercised"
        )
        assert _INST["file_path"] not in stale.content, (
            "stale snippet must NOT mention the correct file path verbatim"
        )


def test_fresh_snippets_reference_the_correct_path():
    trail = generate_research_trail(seed=22, **_INST)
    for fresh in (s for s in trail if not s.stale):
        assert _INST["file_path"] in fresh.content


def test_qualificand_qualifier_condition_are_uniform():
    trail = generate_research_trail(seed=5, **_INST)
    qualificands = {s.qualificand for s in trail}
    qualifiers = {s.qualifier for s in trail}
    conditions = {s.condition for s in trail}
    assert qualificands == {f"swe::{_INST['instance_id']}"}
    assert qualifiers == {"patch_target_file"}
    assert conditions == {f"repo={_INST['repo']}"}


def test_custom_n_stale_and_n_fresh():
    trail = generate_research_trail(seed=0, n_stale=4, n_fresh=1, **_INST)
    assert sum(1 for s in trail if s.stale) == 4
    assert sum(1 for s in trail if not s.stale) == 1
