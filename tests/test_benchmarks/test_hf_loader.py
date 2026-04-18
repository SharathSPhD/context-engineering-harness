"""HuggingFace loader behavior + load_real fallback semantics.

These tests intentionally do NOT exercise a network call — they verify that:
  1. The loader raises HFUnavailable when CEH_DISABLE_HF=1.
  2. Each adapter's load_real path falls back to synthetic on HFUnavailable
     instead of crashing or silently lying about provenance.
"""
from __future__ import annotations

import logging

import pytest

from src.benchmarks.adapters.longctx._hf_loader import HFUnavailable, load_hf_examples
from src.benchmarks.adapters.longctx.helmet import HelmetRagAdapter, HelmetRecallAdapter
from src.benchmarks.adapters.longctx.nocha import NochaJointAccuracyAdapter
from src.benchmarks.adapters.longctx.ruler import RulerNIAHAdapter


def test_loader_respects_ceh_disable_hf(monkeypatch):
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    with pytest.raises(HFUnavailable):
        load_hf_examples(dataset_id="any/thing", split="test")


def test_ruler_load_real_falls_back_to_synthetic_when_disabled(monkeypatch, caplog):
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    adapter = RulerNIAHAdapter(target_tokens=2_000, default_n=4, load_real=True)
    with caplog.at_level(logging.WARNING):
        examples = adapter.load_examples(seed=0)
    assert len(examples) == 4
    assert all(ex.metadata.get("source") == "synthetic" for ex in examples)
    assert any("RULER real loader unavailable" in rec.message for rec in caplog.records)


def test_helmet_rag_load_real_defers_to_p3_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    adapter = HelmetRagAdapter(target_tokens=2_000, default_n=3, load_real=True)
    with caplog.at_level(logging.WARNING):
        examples = adapter.load_examples(seed=0)
    assert len(examples) == 3
    assert any(
        "HELMET RAG real loader unavailable" in rec.message for rec in caplog.records
    )


def test_helmet_recall_load_real_defers_to_p3_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    adapter = HelmetRecallAdapter(target_tokens=2_000, default_n=2, load_real=True)
    with caplog.at_level(logging.WARNING):
        examples = adapter.load_examples(seed=0)
    assert len(examples) == 2
    assert any(
        "HELMET Recall real loader unavailable" in rec.message for rec in caplog.records
    )


def test_nocha_load_real_defers_to_p3_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("CEH_DISABLE_HF", "1")
    adapter = NochaJointAccuracyAdapter(target_tokens=3_000, default_n=2, load_real=True)
    with caplog.at_level(logging.WARNING):
        examples = adapter.load_examples(seed=0)
    assert len(examples) == 2
    assert any("NoCha real loader unavailable" in rec.message for rec in caplog.records)


def test_ruler_hf_config_for_budget_picks_nearest_published_tier():
    adapter = RulerNIAHAdapter(target_tokens=10_000, load_real=True)
    assert adapter._hf_config_for_budget() == "8192"
    adapter2 = RulerNIAHAdapter(target_tokens=70_000, load_real=True)
    assert adapter2._hf_config_for_budget() == "65536"
    adapter3 = RulerNIAHAdapter(target_tokens=131_072, load_real=True)
    assert adapter3._hf_config_for_budget() == "131072"
