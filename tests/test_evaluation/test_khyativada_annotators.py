"""Tests for the rule-based P4 Khyātivāda annotator."""
from __future__ import annotations

from src.evaluation.khyativada_annotators import HeuristicAnnotator, HeuristicLabel
from src.evaluation.khyativada_corpus import generate_corpus


def test_inversion_cue_yields_viparitakhyati():
    ann = HeuristicAnnotator()
    out = ann.label("x", "POST is idempotent; PUT is not.", "The opposite holds: PUT is idempotent, POST is not.")
    assert out.label == "viparitakhyati"
    assert out.rule == "inversion_cue"


def test_nonexistence_cue_yields_asatkhyati():
    ann = HeuristicAnnotator()
    out = ann.label(
        "x",
        "Use the `--strict-mode` flag of `git commit`.",
        "`git commit` has no `--strict-mode` flag.",
    )
    assert out.label == "asatkhyati"


def test_novel_protocol_yields_anirvacaniyakhyati():
    ann = HeuristicAnnotator()
    out = ann.label(
        "x",
        "The HTTP/4 protocol uses Reed-Solomon framing for header recovery.",
        "There is no HTTP/4 protocol; current standards stop at HTTP/3.",
    )
    assert out.label == "anirvacaniyakhyati"


def test_combination_cue_yields_akhyati():
    ann = HeuristicAnnotator()
    out = ann.label(
        "x",
        "Einstein won the 1921 Nobel Prize for the theory of relativity.",
        "Einstein did win the 1921 Nobel Prize, but not for relativity — it was for the photoelectric effect.",
    )
    assert out.label == "akhyati"


def test_version_swap_yields_anyathakhyati():
    ann = HeuristicAnnotator()
    out = ann.label(
        "x",
        "Python 3.10 introduced the free-threaded build.",
        "The free-threaded build actually shipped in Python 3.13, not 3.10.",
    )
    assert out.label == "anyathakhyati"


def test_matching_claim_yields_none():
    ann = HeuristicAnnotator()
    out = ann.label(
        "x",
        "Python 3.13 introduced an experimental free-threaded build.",
        "Python 3.13 introduced an experimental free-threaded build via PEP 703.",
    )
    assert out.label == "none"


def test_default_yields_atmakhyati():
    ann = HeuristicAnnotator()
    out = ann.label(
        "x",
        "The default port for Redis is 8080.",
        "There is no documented default port in the provided source for this service.",
    )
    assert out.label == "atmakhyati"


def test_label_many_returns_one_per_row():
    rows = generate_corpus(n=70)
    ann = HeuristicAnnotator()
    preds = ann.label_many(rows)
    assert len(preds) == len(rows)
    assert {p.item_id for p in preds} == {r.id for r in rows}
    assert all(isinstance(p, HeuristicLabel) for p in preds)


def test_heuristic_accuracy_above_floor_on_corpus():
    """The annotator should reach > 75% accuracy on the deterministic corpus.

    This is the lower bound of "competent rule-based annotator" — anything
    less means the templates and rules drifted apart.
    """
    rows = generate_corpus(n=700, seed=0)
    ann = HeuristicAnnotator()
    preds = ann.label_many(rows)
    correct = sum(1 for p, r in zip(preds, rows, strict=True) if p.label == r.gold_label)
    accuracy = correct / len(rows)
    assert accuracy >= 0.75, f"Heuristic accuracy {accuracy:.3f} below 0.75"


def test_label_predictions_are_in_taxonomy():
    rows = generate_corpus(n=350, seed=1)
    ann = HeuristicAnnotator()
    valid = {
        "anyathakhyati",
        "atmakhyati",
        "anirvacaniyakhyati",
        "asatkhyati",
        "viparitakhyati",
        "akhyati",
        "none",
    }
    assert {p.label for p in ann.label_many(rows)} <= valid


def test_heuristic_is_deterministic():
    rows = generate_corpus(n=100, seed=0)
    ann = HeuristicAnnotator()
    a = ann.label_many(rows)
    b = ann.label_many(rows)
    assert [p.label for p in a] == [p.label for p in b]
    assert [p.rule for p in a] == [p.rule for p in b]


def test_confidence_in_unit_interval():
    rows = generate_corpus(n=100, seed=0)
    ann = HeuristicAnnotator()
    preds = ann.label_many(rows)
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)
