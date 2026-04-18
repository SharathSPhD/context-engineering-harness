"""End-to-end tests for the P4 annotation runner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.h6_khyativada_classifier import run_annotation


def test_run_writes_all_artifacts(tmp_path: Path):
    out = tmp_path / "run-out"
    summary = run_annotation.run(
        n=140, seed=0, judge_seed=1, judge_accuracy=0.85, out=out,
    )
    assert (out / "corpus.jsonl").exists()
    assert (out / "annotator_a.jsonl").exists()
    assert (out / "annotator_b.jsonl").exists()
    assert (out / "agreement_report.json").exists()
    assert (out / "summary.md").exists()

    rep = json.loads((out / "agreement_report.json").read_text())
    assert rep["config"]["n"] == 140
    assert "kappa" in rep["agreement"]
    assert summary["agreement"]["kappa"] == rep["agreement"]["kappa"]


def test_run_meets_kappa_floor_on_default_corpus(tmp_path: Path):
    """Sanity: with a competent simulated judge, Cohen's κ exceeds the 0.6 plan floor."""
    out = tmp_path / "kappa-floor"
    summary = run_annotation.run(
        n=3000, seed=0, judge_seed=1, judge_accuracy=0.85, out=out,
    )
    kappa = summary["agreement"]["kappa"]
    assert kappa >= 0.6, f"Expected κ ≥ 0.6, got {kappa}"


def test_run_returns_distribution_summary(tmp_path: Path):
    out = tmp_path / "dist-out"
    summary = run_annotation.run(
        n=70, seed=2, judge_seed=4, judge_accuracy=0.75, out=out,
    )
    dist = summary["class_distribution"]
    assert sum(dist.values()) == 70
    assert set(dist.keys()) >= {"anyathakhyati", "asatkhyati", "akhyati", "none"}


def test_summary_md_contains_landis_koch_band(tmp_path: Path):
    out = tmp_path / "md-out"
    summary = run_annotation.run(
        n=70, seed=0, judge_seed=0, judge_accuracy=0.85, out=out,
    )
    body = (out / "summary.md").read_text()
    assert "Landis & Koch band" in body
    assert summary["agreement"]["kappa_band"] in body


def test_main_exits_nonzero_when_kappa_below_floor(tmp_path: Path, monkeypatch):
    out = tmp_path / "below-floor"
    rc = run_annotation.main(
        [
            "--n",
            "70",
            "--seed",
            "0",
            "--judge-accuracy",
            "0.20",  # very weak judge ⇒ κ will be far below 0.6
            "--out",
            str(out),
            "--target-kappa",
            "0.6",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 1


def test_main_exits_zero_when_kappa_above_floor(tmp_path: Path):
    out = tmp_path / "above-floor"
    rc = run_annotation.main(
        [
            "--n",
            "210",
            "--seed",
            "0",
            "--judge-accuracy",
            "0.95",
            "--out",
            str(out),
            "--target-kappa",
            "0.6",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0


def test_jsonl_output_is_well_formed(tmp_path: Path):
    out = tmp_path / "wellformed"
    run_annotation.run(n=70, seed=0, judge_seed=0, judge_accuracy=0.8, out=out)
    for fname in ("corpus.jsonl", "annotator_a.jsonl", "annotator_b.jsonl"):
        for line in (out / fname).read_text().splitlines():
            assert json.loads(line)


def test_annotator_b_jsonl_records_simulated_source(tmp_path: Path):
    out = tmp_path / "judge-source"
    run_annotation.run(n=70, seed=0, judge_seed=0, judge_accuracy=0.8, out=out)
    for line in (out / "annotator_b.jsonl").read_text().splitlines():
        rec = json.loads(line)
        assert rec["source"] == "simulated_judge"


def test_annotator_a_jsonl_records_heuristic_source_and_rule(tmp_path: Path):
    out = tmp_path / "heur-source"
    run_annotation.run(n=70, seed=0, judge_seed=0, judge_accuracy=0.8, out=out)
    for line in (out / "annotator_a.jsonl").read_text().splitlines():
        rec = json.loads(line)
        assert rec["source"] == "heuristic"
        assert "rule" in rec
        assert rec["label"] in {
            "anyathakhyati",
            "atmakhyati",
            "anirvacaniyakhyati",
            "asatkhyati",
            "viparitakhyati",
            "akhyati",
            "none",
        }


def test_run_is_deterministic_per_seed(tmp_path: Path):
    a = run_annotation.run(n=210, seed=7, judge_seed=11, judge_accuracy=0.85, out=tmp_path / "a")
    b = run_annotation.run(n=210, seed=7, judge_seed=11, judge_accuracy=0.85, out=tmp_path / "b")
    assert a["agreement"]["kappa"] == b["agreement"]["kappa"]
    assert a["agreement"]["per_class_kappa"] == b["agreement"]["per_class_kappa"]
