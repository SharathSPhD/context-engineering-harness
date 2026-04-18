"""Tests for ``src.evaluation.agreement`` — Cohen's κ and friends."""
from __future__ import annotations

import pytest

from src.evaluation.agreement import (
    AgreementReport,
    agreement_report,
    cohens_kappa,
    confusion_matrix,
    per_class_kappa,
    percent_agreement,
)


def test_perfect_agreement_yields_kappa_one_and_band_almost_perfect():
    a = ["x", "y", "z", "x", "y"]
    b = ["x", "y", "z", "x", "y"]
    assert percent_agreement(a, b) == 1.0
    assert cohens_kappa(a, b) == 1.0
    rep = agreement_report(a, b)
    assert rep.landis_koch_band() == "almost perfect"


def test_perfect_disagreement_yields_negative_kappa():
    a = ["x", "y", "x", "y"]
    b = ["y", "x", "y", "x"]
    k = cohens_kappa(a, b)
    assert k < 0


def test_chance_level_kappa_is_near_zero():
    """If both raters use a uniform prior independently, κ ≈ 0."""
    import random

    rng = random.Random(7)
    n = 4000
    a = [rng.choice(["a", "b", "c"]) for _ in range(n)]
    b = [rng.choice(["a", "b", "c"]) for _ in range(n)]
    k = cohens_kappa(a, b)
    assert -0.05 < k < 0.05


def test_kappa_matches_hand_computed_value():
    """2x2 sanity check.

    Counts: yes-yes=25, yes-no=13, no-yes=5, no-no=7 (n=50).
    p_o   = (25 + 7) / 50 = 0.64.
    p_e   = (38/50)(30/50) + (12/50)(20/50) = 0.456 + 0.096 = 0.552.
    κ     = (0.64 - 0.552) / (1 - 0.552) = 0.088 / 0.448 ≈ 0.1964.
    """
    a = ["yes"] * 25 + ["yes"] * 13 + ["no"] * 5 + ["no"] * 7
    b = ["yes"] * 25 + ["no"] * 13 + ["yes"] * 5 + ["no"] * 7
    expected_kappa = 0.0880 / 0.4480
    k = cohens_kappa(a, b)
    assert abs(k - expected_kappa) < 0.001


def test_per_class_kappa_scopes_to_each_label():
    a = ["x", "y", "x", "y", "x"]
    b = ["x", "x", "x", "y", "x"]
    pck = per_class_kappa(a, b)
    assert "x" in pck
    assert "y" in pck
    assert pck["x"] >= 0.0


def test_confusion_matrix_counts_pairs():
    a = ["x", "y", "x", "y"]
    b = ["x", "x", "y", "y"]
    m = confusion_matrix(a, b)
    assert m["x"]["x"] == 1
    assert m["x"]["y"] == 1
    assert m["y"]["x"] == 1
    assert m["y"]["y"] == 1


def test_confusion_matrix_includes_passed_label_columns_even_if_unused():
    a = ["x", "x"]
    b = ["x", "x"]
    m = confusion_matrix(a, b, labels=["x", "y", "z"])
    assert m["x"]["y"] == 0
    assert m["y"]["x"] == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_agreement_report_bundles_metrics():
    a = ["x", "y", "z", "x", "y", "z"]
    b = ["x", "y", "z", "x", "y", "z"]
    rep = agreement_report(a, b)
    assert isinstance(rep, AgreementReport)
    assert rep.n == 6
    assert rep.percent_agreement == 1.0
    assert rep.kappa == 1.0
    assert set(rep.labels) == {"x", "y", "z"}


def test_landis_koch_bands_partition_kappa_range():
    cases = [
        (-0.5, "poor (worse than chance)"),
        (-0.01, "poor (worse than chance)"),
        (0.10, "slight"),
        (0.30, "fair"),
        (0.50, "moderate"),
        (0.70, "substantial"),
        (0.90, "almost perfect"),
    ]

    def _band_for(k: float) -> str:
        rep = AgreementReport(
            n=10,
            labels=("a", "b"),
            percent_agreement=0.0,
            kappa=k,
            per_class_kappa={},
            confusion={},
            marginal_a={},
            marginal_b={},
        )
        return rep.landis_koch_band()

    for k, expected in cases:
        assert _band_for(k) == expected


def test_agreement_report_as_dict_is_json_safe():
    import json

    a = ["x", "y", "z", "x", "y", "z"]
    b = ["x", "y", "x", "x", "y", "z"]
    rep = agreement_report(a, b)
    d = rep.as_dict()
    json.dumps(d)
    assert d["n"] == 6
    assert "kappa" in d
    assert "kappa_band" in d


def test_kappa_one_when_both_raters_pick_only_one_matching_label():
    a = ["x"] * 10
    b = ["x"] * 10
    assert cohens_kappa(a, b) == 1.0
