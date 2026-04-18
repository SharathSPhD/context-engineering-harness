"""End-to-end tests for the H3/H4/H5 plugin-in-loop runner.

These run the full pipeline against the in-process plugin client and
assert the directional, structural, and statistical contracts every
P6-A artifact must satisfy:

    - JSON file shape contains spec + outcome + per_seed_runs
    - delta_observed > 0   (harness wins by design)
    - target_met == True   (effect ≥ spec.delta and p < alpha)
    - Cohen's d > 0
    - the per-seed paired structure has 2 rows per (model, seed)

Tests intentionally use small n_examples / few seeds to stay fast.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.v2.p6a.run_plugin_inloop import main


@pytest.fixture()
def tmp_out(tmp_path: Path) -> Path:
    out = tmp_path / "p6a"
    out.mkdir()
    return out


def _run(hypothesis: str, out_dir: Path, **kwargs) -> dict:
    n = kwargs.pop("n_examples", 20)
    rc = main([
        "--hypotheses", hypothesis,
        "--models", "claude-haiku-4-5", "claude-sonnet-4-6",
        "--seeds", "0", "1", "2",
        "--n-examples", str(n),
        "--bootstrap-n", "200",
        "--permutation-n", "200",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    files = sorted(
        p for p in out_dir.iterdir()
        if p.suffix == ".json" and not p.name.startswith("_")
    )
    payloads = [json.loads(f.read_text()) for f in files]
    by_label = {p["label"]: p for p in payloads}
    return by_label


def _assert_payload_shape(payload: dict, *, hypothesis_id: str) -> None:
    assert payload["spec"]["hypothesis_id"] == hypothesis_id
    out = payload["outcome"]
    for k in (
        "treatment_metric", "baseline_metric", "delta_observed",
        "ci_low", "ci_high", "p_value", "cohens_d",
        "target_met", "n_examples_used", "n_seeds_used",
    ):
        assert k in out, f"missing outcome key {k!r}"
    rows = payload["per_seed_runs"]
    assert rows
    on_rows = [r for r in rows if r["condition"] == "harness_on"]
    off_rows = [r for r in rows if r["condition"] == "harness_off"]
    assert len(on_rows) == len(off_rows)
    assert all(0.0 <= r["mean_score"] <= 1.0 for r in rows)


def test_h3_runner_emits_positive_delta_and_meets_target(tmp_out: Path) -> None:
    payloads = _run("H3", tmp_out, n_examples=40)
    payload = payloads["H3_buddhi_manas_grounding"]
    _assert_payload_shape(payload, hypothesis_id="H3")
    out = payload["outcome"]
    assert out["delta_observed"] > 0.0, f"expected positive delta, got {out['delta_observed']}"
    assert out["cohens_d"] > 0.0
    # H3 target is +0.10 — the bucketed scenario design easily clears it.
    assert out["target_met"] is True


def test_h4_runner_emits_positive_delta_and_meets_target(tmp_out: Path) -> None:
    # H4 needs a few extra examples per seed to dampen integer noise from
    # the per-scenario random post-bucket size.
    payloads = _run("H4", tmp_out, n_examples=40)
    payload = payloads["H4_event_boundary_compaction"]
    _assert_payload_shape(payload, hypothesis_id="H4")
    out = payload["outcome"]
    assert out["delta_observed"] > 0.0
    assert out["cohens_d"] > 0.0
    assert out["target_met"] is True


def test_h5_runner_emits_unanimous_perfect_resolution(tmp_out: Path) -> None:
    payloads = _run("H5", tmp_out, n_examples=20)
    payload = payloads["H5_avacchedaka_sublation"]
    _assert_payload_shape(payload, hypothesis_id="H5")
    out = payload["outcome"]
    # Plugin: every conflict is sublated cleanly → 1.0
    assert out["treatment_metric"] == pytest.approx(1.0)
    # Baseline: no path resolves the conflict → 0.0
    assert out["baseline_metric"] == pytest.approx(0.0)
    assert out["delta_observed"] == pytest.approx(1.0)
    assert out["target_met"] is True


def test_summary_file_aggregates_all_three(tmp_out: Path) -> None:
    rc = main([
        "--hypotheses", "all",
        "--models", "claude-haiku-4-5",
        "--seeds", "0", "1", "2",
        "--n-examples", "20",
        "--bootstrap-n", "200",
        "--permutation-n", "200",
        "--out-dir", str(tmp_out),
    ])
    assert rc == 0
    summary_path = tmp_out / "_summary_plugin.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    ids = sorted(r["hypothesis_id"] for r in summary["results"])
    assert ids == ["H3", "H4", "H5"]
    assert summary["meta"]["mode"] == "plugin-inloop"
