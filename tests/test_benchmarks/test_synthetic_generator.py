"""_synthetic.py — deterministic, tokenizer-exact long-context generator."""
from __future__ import annotations

import pytest

from src.benchmarks.adapters.longctx._synthetic import (
    build_haystack,
    generate_examples,
    make_needle,
)
from src.utils.tokenizer import count_tokens


def test_make_needle_is_deterministic():
    a = make_needle(seed=42, idx=0)
    b = make_needle(seed=42, idx=0)
    c = make_needle(seed=42, idx=1)
    assert a == b
    assert a != c
    assert a.value in a.sentence


def test_build_haystack_contains_every_needle_exactly_once():
    needles = [make_needle(seed=7, idx=i) for i in range(3)]
    haystack = build_haystack(target_tokens=2_000, needles=needles, seed=7)
    for n in needles:
        assert haystack.count(n.value) >= 1


def test_build_haystack_respects_token_budget():
    needles = [make_needle(seed=1, idx=0)]
    target = 1_500
    haystack = build_haystack(target_tokens=target, needles=needles, seed=1)
    actual = count_tokens(haystack)
    assert target * 0.7 <= actual <= target * 1.3


def test_generate_examples_is_deterministic_per_seed():
    a = generate_examples(n=3, seed=11, target_tokens=600)
    b = generate_examples(n=3, seed=11, target_tokens=600)
    assert [e.id for e in a] == [e.id for e in b]
    assert [e.haystack for e in a] == [e.haystack for e in b]


def test_generate_examples_unique_across_seeds():
    a = generate_examples(n=3, seed=11, target_tokens=600)
    b = generate_examples(n=3, seed=12, target_tokens=600)
    assert {e.haystack for e in a}.isdisjoint({e.haystack for e in b})


def test_generate_examples_rejects_non_positive_n():
    with pytest.raises(ValueError):
        generate_examples(n=0, seed=0, target_tokens=512)


def test_build_haystack_rejects_zero_budget():
    with pytest.raises(ValueError):
        build_haystack(target_tokens=0, needles=[make_needle(0, 0)], seed=0)
