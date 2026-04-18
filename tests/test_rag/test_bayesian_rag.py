"""Tests for src.rag.bayesian_rag.BayesianBetaRAG."""

from __future__ import annotations

import math

import pytest

from src.calibration.metrics import brier_score, expected_calibration_error
from src.rag.bayesian_rag import BayesianBetaRAG
from src.rag.conflicting_qa import ConflictingSourceQA
from src.rag.precision_rag import PrecisionWeightedRAG


def _two_sources(p_correct: float, p_wrong: float) -> list[dict]:
    return [
        {"content": "src-correct", "precision": p_correct, "answer": "correct", "is_correct": True},
        {"content": "src-wrong", "precision": p_wrong, "answer": "wrong", "is_correct": False},
    ]


class TestPosteriorBasics:
    def test_empty_sources_returns_empty_posterior(self) -> None:
        rag = BayesianBetaRAG()
        assert rag.posteriors([]) == {}

    def test_predict_picks_higher_precision_source(self) -> None:
        rag = BayesianBetaRAG()
        best, prob = rag.predict(_two_sources(0.95, 0.30))
        assert best == "correct"
        assert prob > 0.5

    def test_predict_returns_calibrated_probability(self) -> None:
        rag = BayesianBetaRAG()
        _, prob = rag.predict(_two_sources(0.95, 0.30))
        assert 0.0 < prob < 1.0

    def test_predict_returns_none_on_empty(self) -> None:
        best, prob = BayesianBetaRAG().predict([])
        assert best is None
        assert prob == 0.0

    def test_higher_evidence_strength_sharpens_posterior(self) -> None:
        srcs = _two_sources(0.95, 0.30)
        weak = BayesianBetaRAG(evidence_strength=1.0)
        strong = BayesianBetaRAG(evidence_strength=8.0)
        _, p_weak = weak.predict(srcs)
        _, p_strong = strong.predict(srcs)
        assert p_strong > p_weak


class TestConflictDetection:
    def test_no_conflict_when_one_source_dominates(self) -> None:
        rag = BayesianBetaRAG()
        assert rag.detect_conflict(_two_sources(0.99, 0.05)) is False

    def test_conflict_when_precisions_close(self) -> None:
        rag = BayesianBetaRAG()
        assert rag.detect_conflict(_two_sources(0.55, 0.45)) is True

    def test_no_conflict_when_only_one_candidate(self) -> None:
        rag = BayesianBetaRAG()
        srcs = [
            {"content": "a", "precision": 0.6, "answer": "X", "is_correct": True},
            {"content": "b", "precision": 0.6, "answer": "X", "is_correct": True},
        ]
        assert rag.detect_conflict(srcs) is False

    def test_conflict_margin_is_configurable(self) -> None:
        srcs = _two_sources(0.6, 0.4)
        loose = BayesianBetaRAG(conflict_margin=0.5)
        strict = BayesianBetaRAG(conflict_margin=0.05)
        assert loose.detect_conflict(srcs) is True
        assert strict.detect_conflict(srcs) is False


class TestPromptSurface:
    def test_select_sources_sorted_by_precision(self) -> None:
        rag = BayesianBetaRAG()
        srcs = [
            {"content": "a", "precision": 0.5, "answer": "a"},
            {"content": "b", "precision": 0.9, "answer": "b"},
            {"content": "c", "precision": 0.7, "answer": "c"},
        ]
        ranked = rag.select_sources(srcs)
        assert [s["precision"] for s in ranked] == [0.9, 0.7, 0.5]

    def test_build_prompt_includes_calibrated_conflict_note(self) -> None:
        rag = BayesianBetaRAG()
        prompt = rag.build_prompt("Q?", _two_sources(0.6, 0.5))
        assert "conflict" in prompt.lower()
        assert "probability" in prompt.lower()

    def test_build_prompt_omits_note_when_no_conflict(self) -> None:
        rag = BayesianBetaRAG()
        prompt = rag.build_prompt("Q?", _two_sources(0.99, 0.05))
        assert "conflict" not in prompt.lower()


class TestCalibrationOnSyntheticPanel:
    def _panel(self, n: int = 60, seed: int = 7) -> tuple[list[float], list[int], list[float], list[int]]:
        import random

        rng = random.Random(seed)
        bayes = BayesianBetaRAG(evidence_strength=4.0)
        legacy = PrecisionWeightedRAG()

        bayes_probs: list[float] = []
        bayes_outcomes: list[int] = []
        legacy_probs: list[float] = []
        legacy_outcomes: list[int] = []

        for _ in range(n):
            p_correct = rng.uniform(0.55, 0.99)
            p_wrong = rng.uniform(0.05, 0.50)
            ex = ConflictingSourceQA.build_example("Q", "correct", "wrong", p_correct, p_wrong)
            best, prob = bayes.predict(ex.sources)
            bayes_probs.append(prob)
            bayes_outcomes.append(int(best == "correct"))

            top = legacy.select_sources(ex.sources, top_k=1)[0]
            legacy_probs.append(float(top["precision"]))
            legacy_outcomes.append(int(top["answer"] == "correct"))

        return bayes_probs, bayes_outcomes, legacy_probs, legacy_outcomes

    def test_bayesian_brier_finite_and_below_naive(self) -> None:
        bp, bo, _, _ = self._panel()
        bs = brier_score(bp, bo)
        assert math.isfinite(bs)
        assert bs < 0.25

    def test_ece_returns_finite_value(self) -> None:
        bp, bo, _, _ = self._panel()
        ece = expected_calibration_error(bp, bo, n_bins=5)
        assert 0.0 <= ece <= 1.0
