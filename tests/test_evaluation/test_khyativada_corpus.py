"""Tests for the deterministic Khyātivāda example corpus generator."""
from __future__ import annotations

import pytest

from src.evaluation.khyativada_corpus import (
    CorpusRow,
    class_distribution,
    generate_corpus,
)

EXPECTED_LABELS = {
    "anyathakhyati",
    "atmakhyati",
    "anirvacaniyakhyati",
    "asatkhyati",
    "viparitakhyati",
    "akhyati",
    "none",
}


def test_generate_corpus_default_size_is_3000():
    rows = generate_corpus()
    assert len(rows) == 3000


def test_generate_corpus_is_deterministic_per_seed():
    a = generate_corpus(n=500, seed=42)
    b = generate_corpus(n=500, seed=42)
    assert [r.id for r in a] == [r.id for r in b]
    assert [r.claim for r in a] == [r.claim for r in b]
    assert [r.ground_truth for r in a] == [r.ground_truth for r in b]


def test_generate_corpus_seed_changes_results():
    a = generate_corpus(n=500, seed=1)
    b = generate_corpus(n=500, seed=2)
    assert [r.claim for r in a] != [r.claim for r in b]


def test_corpus_covers_all_seven_labels():
    rows = generate_corpus(n=700)
    dist = class_distribution(rows)
    assert set(dist.keys()) == EXPECTED_LABELS
    for label in EXPECTED_LABELS:
        assert dist[label] > 0


def test_corpus_distribution_is_balanced():
    rows = generate_corpus(n=700)
    dist = class_distribution(rows)
    counts = list(dist.values())
    assert max(counts) - min(counts) <= 1


def test_corpus_remainder_distributed_to_first_classes():
    rows = generate_corpus(n=10)
    dist = class_distribution(rows)
    counts = list(dist.values())
    assert sum(counts) == 10
    assert max(counts) - min(counts) <= 1


def test_corpus_rejects_too_small_n():
    with pytest.raises(ValueError):
        generate_corpus(n=3)


def test_each_row_carries_one_gold_label_in_taxonomy():
    rows = generate_corpus(n=350)
    for r in rows:
        assert r.gold_label in EXPECTED_LABELS
        assert r.id and r.template_id and r.claim and r.ground_truth


def test_template_id_groups_rows_by_construction_pattern():
    rows = generate_corpus(n=700, seed=0)
    templates = {r.template_id for r in rows if r.gold_label == "anyathakhyati"}
    assert len(templates) >= 2  # python/react/db swap families


def test_corpus_row_as_dict_round_trip():
    rows = generate_corpus(n=10)
    d = rows[0].as_dict()
    assert {"id", "claim", "context", "ground_truth", "gold_label", "template_id"} <= set(d.keys())
    assert d["gold_label"] in EXPECTED_LABELS


def test_class_distribution_handles_empty_input():
    assert class_distribution([]) == {}


def test_anyathakhyati_rows_mention_two_versions():
    rows = generate_corpus(n=350, seed=0)
    matches = [r for r in rows if r.gold_label == "anyathakhyati"]
    assert matches
    sample = matches[0]
    import re

    digits = re.findall(r"\d+(?:\.\d+)?", sample.claim + sample.ground_truth)
    assert len(digits) >= 2


def test_asatkhyati_rows_assert_nonexistence():
    rows = generate_corpus(n=350, seed=0)
    matches = [r for r in rows if r.gold_label == "asatkhyati"]
    assert matches
    assert any("does not exist" in r.ground_truth.lower() or "no such" in r.ground_truth.lower() or "has no " in r.ground_truth.lower() for r in matches)


def test_none_rows_have_consistent_claim_vs_ground_truth():
    rows = generate_corpus(n=350)
    matches = [r for r in rows if r.gold_label == "none"]
    assert matches
    for r in matches:
        first_word = r.claim.split()[0].lower()
        assert first_word in r.ground_truth.lower() or any(
            tok.lower() in r.ground_truth.lower() for tok in r.claim.split()[:3]
        )


def test_corpus_ids_are_unique():
    rows = generate_corpus(n=3000, seed=0)
    ids = [r.id for r in rows]
    assert len(set(ids)) == len(ids)


def test_corpus_row_is_immutable():
    rows = generate_corpus(n=10)
    with pytest.raises(Exception):
        rows[0].claim = "x"  # type: ignore[misc]
