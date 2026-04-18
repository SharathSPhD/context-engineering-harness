"""Calibration metrics for binary classifiers.

All functions take ``probs`` (predicted P(y=1)) and ``outcomes`` (0/1
ground truth) as plain Python sequences. They have zero numpy / sklearn
dependency so they are safe to import in any context (CI, plugin
runtime, hooks).

Implementations follow the standard definitions:

* **Brier score**: mean squared error between predicted probability
  and binary outcome — `Brier 1950
  <https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml>`_.
* **ECE / MCE**: equal-width binning over the probability axis —
  `Naeini, Cooper, Hauskrecht (AAAI 2015)
  <https://ojs.aaai.org/index.php/AAAI/article/view/9602>`_.

All functions are pure and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class BinStat:
    """Per-bin statistics used to draw reliability diagrams."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.accuracy)


def _validate(probs: Sequence[float], outcomes: Sequence[int]) -> None:
    if len(probs) != len(outcomes):
        raise ValueError(
            f"probs and outcomes must have equal length, got {len(probs)} vs {len(outcomes)}"
        )
    if not probs:
        raise ValueError("probs must be non-empty")
    for p in probs:
        if p < 0.0 or p > 1.0:
            raise ValueError(f"probability {p} outside [0, 1]")
    for o in outcomes:
        if o not in (0, 1, True, False):
            raise ValueError(f"outcome {o!r} must be 0 or 1")


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    _validate(probs, outcomes)
    n = len(probs)
    return sum((p - int(o)) ** 2 for p, o in zip(probs, outcomes)) / n


def reliability_diagram_bins(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_bins: int = 10,
) -> list[BinStat]:
    """Return per-bin statistics for an equal-width reliability diagram.

    Bins are right-closed except for the first bin, which is left-closed
    at 0.0. Empty bins are returned with zero count and zero stats so
    callers can render a complete diagram.
    """
    _validate(probs, outcomes)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = [i / n_bins for i in range(n_bins + 1)]
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]

    for p, o in zip(probs, outcomes):
        if p == 0.0:
            idx = 0
        else:
            idx = min(n_bins - 1, max(0, int((p - 1e-12) * n_bins)))
        buckets[idx].append((float(p), int(o)))

    stats: list[BinStat] = []
    for i, items in enumerate(buckets):
        lo, hi = edges[i], edges[i + 1]
        if not items:
            stats.append(BinStat(lo, hi, 0, 0.0, 0.0))
            continue
        m = len(items)
        mean_conf = sum(p for p, _ in items) / m
        acc = sum(o for _, o in items) / m
        stats.append(BinStat(lo, hi, m, mean_conf, acc))
    return stats


def expected_calibration_error(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    """ECE: weighted average gap between confidence and accuracy."""
    bins = reliability_diagram_bins(probs, outcomes, n_bins=n_bins)
    n = len(probs)
    return sum(b.count * b.gap for b in bins) / n


def maximum_calibration_error(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    """MCE: largest gap between confidence and accuracy across non-empty bins."""
    bins = reliability_diagram_bins(probs, outcomes, n_bins=n_bins)
    nonempty = [b for b in bins if b.count > 0]
    if not nonempty:
        return 0.0
    return max(b.gap for b in nonempty)
