"""End-to-end tests for the P6-A multi-seed runner script.

Confirms that the public CLI surface — `python -m experiments.v2.p6a.run`
under the `mock` mode — produces the expected JSON artifacts and that
the rolled-up HypothesisOutcome carries a positive lift for both H1 and
H2 (since the mock simulator was built to honour the harness contract).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.v2.p6a.run import main as run_main


@pytest.fixture
def tmp_results(tmp_path: Path) -> Path:
    return tmp_path / "p6a"


def test_smoke_run_writes_per_spec_files(tmp_results: Path) -> None:
    rc = run_main(
        [
            "--mode", "mock",
            "--hypotheses", "H1",
            "--n-examples", "8",
            "--bootstrap-n", "200",
            "--permutation-n", "200",
            "--out-dir", str(tmp_results),
            "--seeds", "0", "1",
            "--models", "claude-sonnet-4-6",
        ]
    )
    assert rc == 0
    files = sorted(p.name for p in tmp_results.glob("*.json"))
    assert "_summary.json" in files
    assert any(name.startswith("H1_ruler_") for name in files)


def test_smoke_run_h1_target_met_in_mock(tmp_results: Path) -> None:
    rc = run_main(
        [
            "--mode", "mock",
            "--hypotheses", "H1",
            "--n-examples", "12",
            "--bootstrap-n", "200",
            "--permutation-n", "200",
            "--out-dir", str(tmp_results),
            "--seeds", "0", "1", "2",
            "--models", "claude-sonnet-4-6",
        ]
    )
    assert rc == 0
    summary = json.loads((tmp_results / "_summary.json").read_text())
    h1 = [r for r in summary["results"] if r["hypothesis_id"] == "H1"]
    assert h1, "no H1 results emitted"
    for row in h1:
        out = row["outcome"]
        assert out["delta_observed"] > 0.0, row
        assert out["target_met"] is True, row


def test_smoke_run_h2_target_met_in_mock(tmp_results: Path) -> None:
    rc = run_main(
        [
            "--mode", "mock",
            "--hypotheses", "H2",
            "--n-examples", "10",
            "--bootstrap-n", "200",
            "--permutation-n", "200",
            "--out-dir", str(tmp_results),
            "--seeds", "0", "1",
            "--models", "claude-sonnet-4-6",
        ]
    )
    assert rc == 0
    summary = json.loads((tmp_results / "_summary.json").read_text())
    h2 = [r for r in summary["results"] if r["hypothesis_id"] == "H2"]
    assert h2, "no H2 results emitted"
    for row in h2:
        out = row["outcome"]
        assert out["delta_observed"] > 0.0, row


def test_summary_carries_meta_and_per_run_breakdowns(tmp_results: Path) -> None:
    rc = run_main(
        [
            "--mode", "mock",
            "--hypotheses", "H1",
            "--n-examples", "5",
            "--bootstrap-n", "100",
            "--permutation-n", "100",
            "--out-dir", str(tmp_results),
            "--seeds", "0",
            "--models", "claude-sonnet-4-6", "claude-haiku-4-5",
        ]
    )
    assert rc == 0
    summary = json.loads((tmp_results / "_summary.json").read_text())
    meta = summary["meta"]
    assert meta["mode"] == "mock"
    assert meta["models"] == ["claude-sonnet-4-6", "claude-haiku-4-5"]
    assert meta["seeds"] == [0]
    # Each per-spec file contains both treatment_runs and baseline_runs:
    for path in tmp_results.glob("H1_ruler_*.json"):
        payload = json.loads(path.read_text())
        assert payload["treatment_runs"], path
        assert payload["baseline_runs"], path
        # Two models × one seed = two runs per condition:
        assert len(payload["treatment_runs"]) == 2
        assert len(payload["baseline_runs"]) == 2
