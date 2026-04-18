"""Tests for src.calibration.metrics."""

from __future__ import annotations

import math

import pytest

from src.calibration.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram_bins,
)


class TestBrier:
    def test_perfect_prediction(self) -> None:
        assert brier_score([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]) == 0.0

    def test_worst_prediction(self) -> None:
        assert brier_score([0.0, 1.0, 0.0, 1.0], [1, 0, 1, 0]) == 1.0

    def test_uninformed_50pct(self) -> None:
        bs = brier_score([0.5] * 4, [1, 0, 1, 0])
        assert math.isclose(bs, 0.25)

    def test_calibrated_better_than_overconfident(self) -> None:
        cal = brier_score([0.7, 0.7, 0.7, 0.7], [1, 1, 1, 0])
        over = brier_score([0.99, 0.99, 0.99, 0.99], [1, 1, 1, 0])
        assert cal < over

    def test_validation_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            brier_score([0.5], [0, 1])

    def test_validation_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            brier_score([1.5], [1])

    def test_validation_bad_outcome(self) -> None:
        with pytest.raises(ValueError):
            brier_score([0.5], [2])  # type: ignore[list-item]


class TestECE:
    def test_perfect_calibration_zero_ece(self) -> None:
        probs = [0.95] * 100
        outcomes = [1] * 95 + [0] * 5
        assert expected_calibration_error(probs, outcomes, n_bins=10) < 0.01

    def test_overconfident_classifier_has_high_ece(self) -> None:
        probs = [0.99] * 100
        outcomes = [1] * 70 + [0] * 30
        ece = expected_calibration_error(probs, outcomes, n_bins=10)
        assert ece > 0.25

    def test_uniform_predictions_at_50(self) -> None:
        probs = [0.5] * 100
        outcomes = [1] * 50 + [0] * 50
        assert expected_calibration_error(probs, outcomes, n_bins=10) < 1e-9

    def test_n_bins_validation(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [1], n_bins=0)


class TestMCE:
    def test_mce_geq_ece(self) -> None:
        probs = [0.1, 0.4, 0.6, 0.9, 0.95, 0.99]
        outcomes = [0, 1, 0, 1, 0, 1]
        ece = expected_calibration_error(probs, outcomes, n_bins=10)
        mce = maximum_calibration_error(probs, outcomes, n_bins=10)
        assert mce >= ece - 1e-12

    def test_mce_zero_when_perfectly_calibrated(self) -> None:
        probs = [0.0] * 50 + [1.0] * 50
        outcomes = [0] * 50 + [1] * 50
        assert maximum_calibration_error(probs, outcomes, n_bins=10) == 0.0


class TestReliabilityBins:
    def test_returns_n_bins(self) -> None:
        bins = reliability_diagram_bins([0.5], [1], n_bins=10)
        assert len(bins) == 10

    def test_total_count_equals_input(self) -> None:
        probs = [0.05, 0.15, 0.55, 0.95, 0.95]
        outcomes = [0, 1, 1, 1, 0]
        bins = reliability_diagram_bins(probs, outcomes, n_bins=10)
        assert sum(b.count for b in bins) == len(probs)

    def test_empty_bins_have_zero_stats(self) -> None:
        bins = reliability_diagram_bins([0.5, 0.55], [1, 0], n_bins=10)
        empties = [b for b in bins if b.count == 0]
        for b in empties:
            assert b.mean_confidence == 0.0
            assert b.accuracy == 0.0
            assert b.gap == 0.0

    def test_bin_for_p_one_lands_in_last_bin(self) -> None:
        bins = reliability_diagram_bins([1.0], [1], n_bins=10)
        assert bins[-1].count == 1
