"""Tests for the H3/H4/H5 scenario generators.

Generators must be (a) deterministic per seed, (b) produce the
documented bucket distribution, (c) emit non-degenerate samples that
exercise every code path the runner depends on.
"""
from __future__ import annotations

from experiments.v2.p6a.scenarios import (
    make_h3_cases,
    make_h4_scenarios,
    make_h5_conflicts,
)


def test_h3_is_deterministic_per_seed() -> None:
    a = make_h3_cases(n=20, seed=7)
    b = make_h3_cases(n=20, seed=7)
    assert [c.qid for c in a] == [c.qid for c in b]
    assert [c.gold for c in a] == [c.gold for c in b]
    assert [c.grounded_precision for c in a] == [c.grounded_precision for c in b]


def test_h3_seeds_diverge() -> None:
    a = make_h3_cases(n=30, seed=1)
    b = make_h3_cases(n=30, seed=2)
    assert [c.gold for c in a] != [c.gold for c in b]


def test_h3_buckets_present() -> None:
    cases = make_h3_cases(n=300, seed=42)
    high = [c for c in cases if c.gold is not None and c.grounded_precision >= 0.5]
    low = [c for c in cases if c.gold is not None and c.grounded_precision < 0.5 and c.grounded_content is not None]
    none = [c for c in cases if c.gold is None]
    assert high, "expected high-precision grounded cases"
    assert low, "expected low-precision grounded cases"
    assert none, "expected ungrounded cases"
    assert (len(high) + len(low) + len(none)) == 300


def test_h4_is_deterministic_per_seed() -> None:
    a = make_h4_scenarios(n=10, seed=3)
    b = make_h4_scenarios(n=10, seed=3)
    assert [s.sid for s in a] == [s.sid for s in b]
    assert [s.n_post for s in a] == [s.n_post for s in b]


def test_h4_buckets_distinct_qualificands() -> None:
    scenarios = make_h4_scenarios(n=5, seed=11)
    for sc in scenarios:
        quals = {it.qualificand for it in sc.items}
        assert {"pre", "post", "noise"}.issubset(quals)
        post = [it for it in sc.items if it.bucket == "post"]
        assert post, "post bucket non-empty"
        # split: at least one validated and one fresh post item
        assert any(it.qualifier == "validated" for it in post)
        assert any(it.qualifier == "fresh" for it in post)


def test_h4_boundary_text_contains_pivot() -> None:
    scenarios = make_h4_scenarios(n=2, seed=0)
    assert "BREAKING CHANGE" in scenarios[0].boundary_text
    assert "default settings" in scenarios[0].boundary_text


def test_h5_is_deterministic_per_seed() -> None:
    a = make_h5_conflicts(n=8, seed=5)
    b = make_h5_conflicts(n=8, seed=5)
    assert [c.cid for c in a] == [c.cid for c in b]
    assert [c.older_value for c in a] == [c.older_value for c in b]
    assert [c.newer_value for c in a] == [c.newer_value for c in b]


def test_h5_precision_ordering_strict() -> None:
    """Newer must always strictly exceed older — sublate_with_evidence
    rejects equal-precision cases as not-justified."""
    for seed in range(5):
        for c in make_h5_conflicts(n=20, seed=seed):
            assert c.newer_precision > c.older_precision
