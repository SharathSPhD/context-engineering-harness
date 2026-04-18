"""HypothesisSpec — declarative description of one hypothesis (H1-H7).

A hypothesis says: "Under condition X (treatment), metric M is greater
(or smaller, or equal-within-epsilon) than under condition Y (baseline),
with effect ≥ delta on adapter A."

The runner reads the spec, executes both conditions, and produces a
`HypothesisOutcome` with point estimates, bootstrap CI, paired permutation
p-value, paired Cohen's d, and a binary `target_met` verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TargetDirection(str, Enum):
    GREATER = "greater"  # treatment > baseline + delta
    LESS = "less"        # treatment < baseline - delta
    EQUIV = "equiv"      # |treatment - baseline| <= delta (TOST-style equivalence)


@dataclass
class HypothesisSpec:
    """Declarative hypothesis declaration consumed by MultiSeedRunner."""
    hypothesis_id: str           # e.g. "H1"
    description: str             # one-sentence claim
    adapter_name: str            # adapter to use (registry key)
    treatment_condition: str     # label for the harness-on condition
    baseline_condition: str      # label for the baseline / harness-off condition
    metric: str = "accuracy"     # "accuracy" | "score" | adapter-defined
    direction: TargetDirection = TargetDirection.GREATER
    delta: float = 0.0           # minimum-effect-of-interest
    n_examples: int | None = None  # None ⇒ adapter default
    seeds: tuple[int, ...] = (0, 1, 2)
    models: tuple[str, ...] = ("claude-sonnet-4-6",)
    significance_alpha: float = 0.05
    notes: str = ""


@dataclass
class HypothesisOutcome:
    """Post-run verdict for one HypothesisSpec."""
    spec: HypothesisSpec
    treatment_metric: float
    baseline_metric: float
    delta_observed: float
    ci_low: float
    ci_high: float
    p_value: float
    cohens_d: float
    target_met: bool
    n_examples_used: int
    n_seeds_used: int
    extra: dict = field(default_factory=dict)
