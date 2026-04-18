"""Tests for ``_real_corpus`` — real long-context distractor module.

These tests do not hit the network. They:

* Verify the module is opt-in (raises ``RealCorpusUnavailable`` when not enabled).
* Use a stub :class:`RealCorpus` to exercise the haystack assembler.
* Confirm caching round-trips through the cache directory.
* Confirm the assembler honours token budgets and inserts every needle.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from src.benchmarks.adapters.longctx._real_corpus import (
    HFArxivCorpus,
    HFWikipediaCorpus,
    MixedCorpus,
    RealCorpus,
    RealCorpusUnavailable,
    _read_cache,
    _write_cache,
    build_real_haystack,
    make_real_corpus,
)
from src.benchmarks.adapters.longctx._synthetic import make_needle
from src.utils.tokenizer import count_tokens


class _StubCorpus(RealCorpus):
    """Predictable corpus for assembler tests."""

    name = "stub"

    def __init__(self, passages: list[str]) -> None:
        self._passages = passages

    def passages(self, *, target_total_tokens: int, seed: int) -> Iterator[str]:
        emitted = 0
        for p in self._passages:
            if emitted >= target_total_tokens:
                return
            yield p
            emitted += count_tokens(p)


@pytest.fixture
def cache_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    yield tmp_path


def test_opt_in_required_for_wikipedia(monkeypatch):
    monkeypatch.delenv("CEH_REAL_LONGCTX", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/ceh-test-noenv")
    corpus = HFWikipediaCorpus()
    with pytest.raises(RealCorpusUnavailable, match="opt-in"):
        corpus._load_chunks()


def test_opt_in_disabled_when_hf_disabled(monkeypatch):
    monkeypatch.setenv("CEH_REAL_LONGCTX", "1")
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    corpus = HFWikipediaCorpus()
    with pytest.raises(RealCorpusUnavailable, match="CEH_DISABLE_HF"):
        corpus._load_chunks()


def test_make_real_corpus_aggregates_failures(monkeypatch):
    monkeypatch.delenv("CEH_REAL_LONGCTX", raising=False)
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    with pytest.raises(RealCorpusUnavailable):
        make_real_corpus(("wikipedia", "arxiv"))


def test_make_real_corpus_rejects_unknown_source(monkeypatch):
    monkeypatch.delenv("CEH_REAL_LONGCTX", raising=False)
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    with pytest.raises(RealCorpusUnavailable, match="unknown source"):
        make_real_corpus(("not-a-corpus",))


def test_cache_round_trip(cache_tmp):
    chunks = ["one para", "two para", "three para"]
    _write_cache("unit-test", chunks)
    assert _read_cache("unit-test") == chunks


def test_cache_miss_returns_none(cache_tmp):
    assert _read_cache("does-not-exist") is None


def test_wikipedia_uses_cache_when_present(cache_tmp, monkeypatch):
    """Even with opt-in disabled, a hot cache short-circuits network."""
    monkeypatch.delenv("CEH_REAL_LONGCTX", raising=False)
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    corpus = HFWikipediaCorpus()
    _write_cache(corpus._cache_key(), ["paragraph one", "paragraph two"])
    chunks = corpus._load_chunks()
    assert chunks == ["paragraph one", "paragraph two"]


def test_arxiv_uses_cache_when_present(cache_tmp, monkeypatch):
    monkeypatch.delenv("CEH_REAL_LONGCTX", raising=False)
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    corpus = HFArxivCorpus()
    _write_cache(corpus._cache_key(), ["abstract one", "abstract two"])
    chunks = corpus._load_chunks()
    assert chunks == ["abstract one", "abstract two"]


def test_mixed_corpus_requires_at_least_one_source():
    with pytest.raises(ValueError):
        MixedCorpus(corpora=())


def test_mixed_corpus_round_robins_passages():
    a = _StubCorpus(["a1", "a2", "a3"])
    b = _StubCorpus(["b1", "b2", "b3"])
    mixed = MixedCorpus(corpora=(a, b))
    out = list(mixed.passages(target_total_tokens=100, seed=0))
    assert "a1" in out and "b1" in out


def test_build_real_haystack_inserts_every_needle():
    passages = [f"distractor passage number {i} with various filler text" for i in range(60)]
    corpus = _StubCorpus(passages)
    needles = [make_needle(seed=42, idx=i) for i in range(3)]
    hay = build_real_haystack(
        target_tokens=1_000,
        needles=needles,
        seed=42,
        corpus=corpus,
    )
    for n in needles:
        assert n.value in hay


def test_build_real_haystack_respects_token_budget_loosely():
    passages = ["short passage " + ("filler " * 20) for _ in range(50)]
    corpus = _StubCorpus(passages)
    needles = [make_needle(seed=1, idx=0)]
    target = 1_500
    hay = build_real_haystack(
        target_tokens=target,
        needles=needles,
        seed=1,
        corpus=corpus,
    )
    actual = count_tokens(hay)
    # Real-corpus assembler is allowed to overshoot by at most one passage.
    assert target * 0.5 <= actual <= target * 1.5


def test_build_real_haystack_handles_no_needles():
    passages = ["passage one", "passage two"]
    corpus = _StubCorpus(passages)
    out = build_real_haystack(target_tokens=200, needles=[], seed=0, corpus=corpus)
    assert "passage one" in out


def test_build_real_haystack_rejects_zero_budget():
    corpus = _StubCorpus(["x"])
    with pytest.raises(ValueError):
        build_real_haystack(target_tokens=0, needles=[], seed=0, corpus=corpus)


def test_build_real_haystack_raises_when_corpus_empty():
    corpus = _StubCorpus([])
    with pytest.raises(RealCorpusUnavailable):
        build_real_haystack(
            target_tokens=500,
            needles=[make_needle(0, 0)],
            seed=0,
            corpus=corpus,
        )


def test_build_real_haystack_cycles_when_few_passages():
    """If there are fewer passages than needles, the assembler still places every needle."""
    corpus = _StubCorpus(["one", "two"])
    needles = [make_needle(seed=9, idx=i) for i in range(5)]
    out = build_real_haystack(target_tokens=100, needles=needles, seed=9, corpus=corpus)
    for n in needles:
        assert n.value in out


def test_build_real_haystack_is_deterministic_per_seed():
    passages = [f"passage {i} " + ("noise " * 5) for i in range(40)]
    corpus = _StubCorpus(passages)
    needles = [make_needle(seed=11, idx=i) for i in range(2)]
    a = build_real_haystack(target_tokens=600, needles=needles, seed=11, corpus=_StubCorpus(passages))
    b = build_real_haystack(target_tokens=600, needles=needles, seed=11, corpus=_StubCorpus(passages))
    assert a == b
