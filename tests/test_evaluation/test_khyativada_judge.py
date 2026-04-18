"""Tests for the deterministic LLM-as-judge surrogate."""
from __future__ import annotations

import pytest

from src.evaluation.khyativada_corpus import generate_corpus
from src.evaluation.khyativada_judge import (
    JudgePrediction,
    simulate_judge,
)


def test_simulate_judge_returns_prediction_per_row():
    rows = generate_corpus(n=70)
    preds = simulate_judge(rows, accuracy=0.8, seed=1)
    assert len(preds) == len(rows)
    assert all(isinstance(p, JudgePrediction) for p in preds)
    assert all(p.source == "simulated_judge" for p in preds)
    assert {p.item_id for p in preds} == {r.id for r in rows}


def test_simulate_judge_is_deterministic_per_seed():
    rows = generate_corpus(n=200, seed=0)
    a = simulate_judge(rows, accuracy=0.8, seed=11)
    b = simulate_judge(rows, accuracy=0.8, seed=11)
    assert [p.label for p in a] == [p.label for p in b]
    assert [p.confidence for p in a] == [p.confidence for p in b]


def test_simulate_judge_seed_changes_outcomes():
    rows = generate_corpus(n=200, seed=0)
    a = simulate_judge(rows, accuracy=0.8, seed=11)
    b = simulate_judge(rows, accuracy=0.8, seed=12)
    assert [p.label for p in a] != [p.label for p in b]


def test_accuracy_controls_realised_agreement_rate():
    rows = generate_corpus(n=2000, seed=0)
    high = simulate_judge(rows, accuracy=0.9, seed=1)
    low = simulate_judge(rows, accuracy=0.5, seed=1)

    high_correct = sum(1 for p, r in zip(high, rows) if p.label == r.gold_label)
    low_correct = sum(1 for p, r in zip(low, rows) if p.label == r.gold_label)
    assert high_correct > low_correct
    assert 0.84 < high_correct / len(rows) < 0.94
    assert 0.46 < low_correct / len(rows) < 0.55


def test_invalid_accuracy_raises():
    rows = generate_corpus(n=10)
    with pytest.raises(ValueError):
        simulate_judge(rows, accuracy=-0.1)
    with pytest.raises(ValueError):
        simulate_judge(rows, accuracy=1.5)


def test_simulated_wrong_labels_stay_in_taxonomy():
    rows = generate_corpus(n=400, seed=0)
    preds = simulate_judge(rows, accuracy=0.7, seed=2)
    valid = {
        "anyathakhyati",
        "atmakhyati",
        "anirvacaniyakhyati",
        "asatkhyati",
        "viparitakhyati",
        "akhyati",
        "none",
    }
    assert {p.label for p in preds} <= valid


def test_simulated_judge_confidence_is_in_unit_interval():
    rows = generate_corpus(n=300, seed=0)
    preds = simulate_judge(rows, accuracy=0.75, seed=3)
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)


def test_simulated_judge_emits_higher_confidence_when_correct():
    rows = generate_corpus(n=500, seed=0)
    preds = simulate_judge(rows, accuracy=0.9, seed=5)
    correct = [p.confidence for p, r in zip(preds, rows) if p.label == r.gold_label]
    wrong = [p.confidence for p, r in zip(preds, rows) if p.label != r.gold_label]
    assert correct
    assert wrong
    assert sum(correct) / len(correct) > sum(wrong) / len(wrong)
