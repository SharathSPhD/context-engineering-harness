"""Inter-annotator agreement metrics.

Pure-Python implementations (no scipy) of:

* Cohen's κ for two raters on the same set of nominal items.
* Per-class κ (binary one-vs-rest κ for each label).
* Percent agreement (the upper baseline κ corrects for).
* Confusion matrix in dict form, with marginal sums attached.

These metrics back the P4 annotation experiment that asks: "do two
independent annotators (heuristic + LLM-as-judge) agree on the
6-class Khyātivāda taxonomy strongly enough to support our paper's
claim that the taxonomy is operational?" The bar in the plan is
κ ≥ 0.6.

References:
* Cohen, J. (1960). A coefficient of agreement for nominal scales.
* Landis & Koch (1977). The measurement of observer agreement for
  categorical data — interpretive bands for κ.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _check_lengths(a: Sequence[str], b: Sequence[str]) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("at least one paired observation is required")


@dataclass(frozen=True)
class AgreementReport:
    """Bundled agreement metrics for two annotators."""

    n: int
    labels: tuple[str, ...]
    percent_agreement: float
    kappa: float
    per_class_kappa: dict[str, float]
    confusion: dict[str, dict[str, int]]
    marginal_a: dict[str, int]
    marginal_b: dict[str, int]

    def landis_koch_band(self) -> str:
        """Return the qualitative band per Landis & Koch (1977)."""
        k = self.kappa
        if k < 0.0:
            return "poor (worse than chance)"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "labels": list(self.labels),
            "percent_agreement": self.percent_agreement,
            "kappa": self.kappa,
            "kappa_band": self.landis_koch_band(),
            "per_class_kappa": dict(self.per_class_kappa),
            "confusion": {k: dict(v) for k, v in self.confusion.items()},
            "marginal_a": dict(self.marginal_a),
            "marginal_b": dict(self.marginal_b),
        }


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    """Fraction of items where the two annotators picked the same label."""
    _check_lengths(a, b)
    same = sum(1 for ai, bi in zip(a, b, strict=True) if ai == bi)
    return same / len(a)


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's κ for two raters on the same items.

    Returns 1.0 for perfect agreement, 0.0 for chance-level agreement,
    and negative values for systematic disagreement.
    """
    _check_lengths(a, b)
    n = len(a)
    p_o = percent_agreement(a, b)

    labels = sorted({*a, *b})
    p_e = 0.0
    for label in labels:
        pa = sum(1 for x in a if x == label) / n
        pb = sum(1 for x in b if x == label) / n
        p_e += pa * pb

    if p_e == 1.0:  # both raters always pick the same single label
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def confusion_matrix(
    a: Sequence[str],
    b: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Return ``{rater_a_label: {rater_b_label: count}}`` for the paired rows."""
    _check_lengths(a, b)
    if labels is None:
        labels = sorted({*a, *b})
    matrix: dict[str, dict[str, int]] = {row: {col: 0 for col in labels} for row in labels}
    for ai, bi in zip(a, b, strict=True):
        if ai not in matrix:
            matrix[ai] = {col: 0 for col in labels}
        if bi not in matrix[ai]:
            matrix[ai][bi] = 0
        matrix[ai][bi] += 1
    return matrix


def per_class_kappa(
    a: Sequence[str],
    b: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute one-vs-rest κ per label so the report can localize disagreements."""
    _check_lengths(a, b)
    if labels is None:
        labels = sorted({*a, *b})
    out: dict[str, float] = {}
    for label in labels:
        bin_a = [str(int(x == label)) for x in a]
        bin_b = [str(int(x == label)) for x in b]
        if len(set(bin_a)) == 1 and len(set(bin_b)) == 1 and bin_a[0] == bin_b[0]:
            out[label] = 1.0
        else:
            out[label] = cohens_kappa(bin_a, bin_b)
    return out


def agreement_report(a: Sequence[str], b: Sequence[str]) -> AgreementReport:
    """Bundle :class:`AgreementReport` with overall κ, per-class κ, and confusion matrix."""
    _check_lengths(a, b)
    labels = tuple(sorted({*a, *b}))
    return AgreementReport(
        n=len(a),
        labels=labels,
        percent_agreement=percent_agreement(a, b),
        kappa=cohens_kappa(a, b),
        per_class_kappa=per_class_kappa(a, b, labels=labels),
        confusion=confusion_matrix(a, b, labels=labels),
        marginal_a={label: sum(1 for x in a if x == label) for label in labels},
        marginal_b={label: sum(1 for x in b if x == label) for label in labels},
    )


__all__ = [
    "AgreementReport",
    "agreement_report",
    "cohens_kappa",
    "confusion_matrix",
    "per_class_kappa",
    "percent_agreement",
]
