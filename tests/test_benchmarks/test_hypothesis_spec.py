"""HypothesisSpec — declarative shape, defaults, and TargetDirection enum."""
from __future__ import annotations

from src.benchmarks.hypothesis import HypothesisOutcome, HypothesisSpec, TargetDirection


def test_target_direction_enum_values():
    assert TargetDirection.GREATER.value == "greater"
    assert TargetDirection.LESS.value == "less"
    assert TargetDirection.EQUIV.value == "equiv"


def test_hypothesis_spec_defaults():
    spec = HypothesisSpec(
        hypothesis_id="H1",
        description="example",
        adapter_name="ruler",
        treatment_condition="harness_on",
        baseline_condition="harness_off",
    )
    assert spec.metric == "accuracy"
    assert spec.direction == TargetDirection.GREATER
    assert spec.delta == 0.0
    assert spec.seeds == (0, 1, 2)
    assert spec.significance_alpha == 0.05


def test_hypothesis_outcome_extra_defaults_to_empty_dict():
    spec = HypothesisSpec(
        hypothesis_id="H1",
        description="x",
        adapter_name="a",
        treatment_condition="t",
        baseline_condition="b",
    )
    out = HypothesisOutcome(
        spec=spec,
        treatment_metric=0.9,
        baseline_metric=0.5,
        delta_observed=0.4,
        ci_low=0.3,
        ci_high=0.5,
        p_value=0.001,
        cohens_d=1.5,
        target_met=True,
        n_examples_used=100,
        n_seeds_used=3,
    )
    assert out.extra == {}
    assert out.spec is spec
