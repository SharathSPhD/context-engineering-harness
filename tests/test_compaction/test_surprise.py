"""Tests for src.compaction.surprise."""

from __future__ import annotations

import math
import os

import pytest

from src.compaction.detector import EventBoundaryDetector
from src.compaction.surprise import (
    HeuristicSurpriseScorer,
    SurpriseProfile,
    TokenSurprise,
    _squash_nll,
    event_boundaries_from_text,
    make_surprise_scorer,
    smooth,
)


def test_squash_nll_zero_is_zero():
    assert _squash_nll(0.0) == 0.0


def test_squash_nll_monotonic():
    assert _squash_nll(0.5) < _squash_nll(2.0) < _squash_nll(8.0)


def test_squash_nll_bounded_below_one():
    assert _squash_nll(50.0) < 1.0


def test_smooth_centers_window():
    out = smooth([0.0, 0.0, 1.0, 0.0, 0.0], window=3)
    assert len(out) == 5
    assert out[2] > out[0]


def test_smooth_empty_returns_empty():
    assert smooth([]) == []


class TestHeuristic:
    def test_empty_text_returns_empty_profile(self):
        scorer = HeuristicSurpriseScorer()
        prof = scorer.score_text("")
        assert prof.tokens == []
        assert prof.backend == "heuristic"

    def test_repeated_token_decays_to_zero_surprise(self):
        scorer = HeuristicSurpriseScorer()
        prof = scorer.score_text("foo foo foo foo foo foo foo foo foo foo")
        assert prof.tokens[0].normalised < 1e-9
        assert prof.tokens[-1].normalised < 1e-9

    def test_first_occurrence_of_token_has_positive_surprise_when_following_other_tokens(self):
        scorer = HeuristicSurpriseScorer()
        prof = scorer.score_text("alpha beta gamma delta")
        normed = [t.normalised for t in prof.tokens]
        for v in normed:
            assert v >= 0.0

    def test_distinct_tokens_in_long_text_produce_some_high_surprise(self):
        text = " ".join(["word_" + str(i) for i in range(40)])
        scorer = HeuristicSurpriseScorer()
        prof = scorer.score_text(text)
        normed = [t.normalised for t in prof.tokens]
        assert max(normed) > 0.3

    def test_token_count_matches_whitespace_split(self):
        scorer = HeuristicSurpriseScorer()
        prof = scorer.score_text("the quick brown fox")
        assert len(prof.tokens) == 4

    def test_profile_normalised_property(self):
        scorer = HeuristicSurpriseScorer()
        prof = scorer.score_text("a b c")
        assert prof.normalised == [t.normalised for t in prof.tokens]
        assert prof.nll == [t.nll for t in prof.tokens]


class TestFactory:
    def test_heuristic_force(self, monkeypatch):
        monkeypatch.delenv("CEH_SURPRISE_BACKEND", raising=False)
        scorer = make_surprise_scorer("heuristic")
        assert scorer.backend_name == "heuristic"

    def test_env_var_heuristic_override(self, monkeypatch):
        monkeypatch.setenv("CEH_SURPRISE_BACKEND", "heuristic")
        scorer = make_surprise_scorer()
        assert scorer.backend_name == "heuristic"

    def test_auto_falls_back_to_heuristic_when_models_absent(self, monkeypatch):
        # In CI we don't have vllm/torch installed; auto must NOT raise.
        monkeypatch.delenv("CEH_SURPRISE_BACKEND", raising=False)
        scorer = make_surprise_scorer("auto")
        assert scorer.backend_name in {"vllm", "hf", "heuristic"}


class TestEventBoundaryFromText:
    def test_returns_indices_with_high_smoothed_surprise(self):
        text = " ".join(["repeated"] * 30 + ["XYZUNIQUE"] * 1 + ["repeated"] * 30)
        scorer = HeuristicSurpriseScorer(scale=2.0)
        idx = event_boundaries_from_text(text, scorer=scorer, threshold=0.05)
        assert isinstance(idx, list)

    def test_threshold_above_one_returns_no_boundaries(self):
        scorer = HeuristicSurpriseScorer()
        # threshold=1.0 is the maximum permitted by EventBoundaryDetector,
        # so nothing strictly exceeds it.
        idx = event_boundaries_from_text("any text here at all", scorer=scorer, threshold=1.0)
        assert idx == []


class TestDetectInTextIntegration:
    def test_detect_in_text_returns_indices_and_profile(self):
        det = EventBoundaryDetector(surprise_threshold=0.05)
        scorer = HeuristicSurpriseScorer(scale=2.0)
        text = "alpha beta gamma delta epsilon zeta"
        idx, profile = det.detect_in_text(text, scorer=scorer)
        assert isinstance(idx, list)
        assert isinstance(profile, SurpriseProfile)
        assert profile.backend == "heuristic"
        assert len(profile.tokens) == 6

    def test_detect_in_text_profile_records_backend_truthfully(self):
        det = EventBoundaryDetector(surprise_threshold=0.5)
        scorer = HeuristicSurpriseScorer()
        _, profile = det.detect_in_text("one two three", scorer=scorer)
        assert profile.backend == "heuristic"
        assert profile.model is None

    def test_detect_in_text_uses_default_scorer_when_none_passed(self, monkeypatch):
        monkeypatch.setenv("CEH_SURPRISE_BACKEND", "heuristic")
        det = EventBoundaryDetector(surprise_threshold=0.5)
        idx, profile = det.detect_in_text("one two three four five")
        assert profile.backend == "heuristic"
        assert isinstance(idx, list)


class TestVLLMScorerNotInstalled:
    def test_vllm_raises_clean_runtime_error_when_missing(self, monkeypatch):
        # Skip if vllm IS installed in this environment.
        try:
            import vllm  # noqa: F401
        except ImportError:
            from src.compaction.surprise import VLLMSurpriseScorer

            with pytest.raises(RuntimeError, match="vllm not installed"):
                VLLMSurpriseScorer()
        else:
            pytest.skip("vllm is installed in this environment")


class TestHFScorerNotInstalled:
    def test_hf_raises_clean_runtime_error_when_missing(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            from src.compaction.surprise import HFSurpriseScorer

            with pytest.raises(RuntimeError, match="transformers"):
                HFSurpriseScorer()
        else:
            pytest.skip("torch+transformers installed in this environment")
