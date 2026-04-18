"""End-to-end tests for the P6-C SWE-bench-Verified A/B runner.

The runner is fully deterministic offline (no LLM, no docker). These
tests run a small N so the suite stays fast, but they re-run through
the *same* code path as the production artifact so any regression in
the plugin's sublation/compaction or in the per-instance permutation
test will fail here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.v2.p6c.research_evidence import generate_research_trail
from experiments.v2.p6c.run_swebench_ab import (
    PatchSimulator,
    _build_research_block_with_harness,
    _build_research_block_without_harness,
    run_ab,
)
from experiments.v2.p6a.plugin_client import PratyakshaPluginClient


# ---------------------------------------------------------------------------
# Per-arm research-block builders
# ---------------------------------------------------------------------------


def _trail():
    return generate_research_trail(
        instance_id="synth_swe_00_0001_abc",
        repo="synthorg/utils",
        file_path="synth_utils/strings.py",
        issue_summary="snake_to_camel returns wrong case for first segment",
        seed=0,
    )


def test_with_harness_block_drops_every_stale_snippet():
    client = PratyakshaPluginClient()
    block, telemetry = _build_research_block_with_harness(
        snippets=_trail(), client=client, precision_gate=0.50
    )
    assert "_legacy_" not in block, (
        "post-compact block must not surface any stale wrong-path snippet"
    )
    assert telemetry["n_sublated"] >= 1
    assert telemetry["n_active_after_compact"] >= 1
    assert telemetry["n_active_after_compact"] <= telemetry["n_snippets_total"]


def test_with_harness_block_keeps_fresh_snippets():
    client = PratyakshaPluginClient()
    block, _ = _build_research_block_with_harness(
        snippets=_trail(), client=client, precision_gate=0.50
    )
    assert "synth_utils/strings.py" in block


def test_without_harness_block_preserves_stale_signal_when_below_budget():
    block, telemetry = _build_research_block_without_harness(
        snippets=_trail(), max_research_tokens=10_000
    )
    assert "_legacy_" in block
    assert telemetry["n_sublated"] == 0
    assert telemetry["n_compacted"] == 0


def test_without_harness_block_truncates_under_tight_budget():
    block, telemetry = _build_research_block_without_harness(
        snippets=_trail(), max_research_tokens=20  # tiny
    )
    assert telemetry["research_block_tokens"] <= 20


# ---------------------------------------------------------------------------
# Patch simulator anchoring
# ---------------------------------------------------------------------------


def test_simulator_anchors_on_first_backticked_path():
    sim = PatchSimulator()
    out = sim(
        prompt=(
            "## Research notes\n"
            "- (0.92) The relevant module is `pkg/foo.py` per changelog\n"
            "- (0.40) Old SO answer says edit `pkg/legacy.py`\n"
            "## Task\nFix it."
        ),
        model="claude-haiku-4-5",
        max_tokens=512,
        system="",
        seed=0,
    )
    assert out.metadata["anchored_path"] == "pkg/foo.py"
    assert "diff --git a/pkg/foo.py b/pkg/foo.py" in out.text


def test_simulator_anchors_on_stale_when_only_stale_present():
    sim = PatchSimulator()
    out = sim(
        prompt="## Research\n- Old SO post says edit `pkg/legacy.py`\n",
        model="claude-haiku-4-5",
        max_tokens=512,
        system="",
        seed=0,
    )
    assert out.metadata["anchored_path"] == "pkg/legacy.py"


# ---------------------------------------------------------------------------
# End-to-end smoke run
# ---------------------------------------------------------------------------


def test_run_ab_writes_artifact_and_summary(tmp_path: Path):
    headline = run_ab(
        n_examples=20,
        seeds=(0, 1, 2),
        models=("claude-haiku-4-5", "claude-sonnet-4-6"),
        research_budget_tokens=512,
        precision_gate=0.50,
        bootstrap_n=200,
        permutation_n=200,
        out_dir=tmp_path,
        load_real=False,
        use_docker=False,
    )

    assert (tmp_path / "swebench_ab.json").exists()
    assert (tmp_path / "_summary.json").exists()

    payload = json.loads((tmp_path / "swebench_ab.json").read_text())
    assert payload["label"] == "P6-C_swebench_verified_AB"
    assert payload["spec"]["n_examples_per_seed"] == 20
    assert len(payload["spec"]["seeds"]) == 3
    assert len(payload["spec"]["models"]) == 2

    # 20 instances × 3 seeds × 2 models = 120 paired instances per arm.
    assert payload["outcome_per_instance"]["n_pairs"] == 120
    assert payload["outcome_per_seed_mean"]["n_pairs"] == 6
    # 6 (model, seed, with_harness) + 6 baseline = 12 per_seed rows.
    assert len(payload["per_seed"]) == 12


def test_run_ab_treatment_dominates_baseline(tmp_path: Path):
    """The plugin should produce a strictly higher mean heuristic score
    AND a strictly higher target-file anchor rate, because it filters
    the stale wrong-path snippet out of the prompt.
    """
    headline = run_ab(
        n_examples=30,
        seeds=(0, 1, 2),
        models=("claude-sonnet-4-6",),
        research_budget_tokens=512,
        precision_gate=0.50,
        bootstrap_n=500,
        permutation_n=500,
        out_dir=tmp_path,
        load_real=False,
        use_docker=False,
    )
    assert headline["treatment_metric_mean"] > headline["baseline_metric_mean"]
    assert (
        headline["treatment_target_path_hit_rate"]
        > headline["baseline_target_path_hit_rate"]
    )
    assert headline["outcome_per_instance"]["delta_observed"] > 0.0
    assert headline["total_sublations_fired"] >= 1


def test_run_ab_target_met_at_default_size(tmp_path: Path):
    """At the published P6-C size the headline target_met flag must be True
    on both the per-instance and the per-(model, seed) tests.
    """
    headline = run_ab(
        n_examples=120,
        seeds=(0, 1, 2),
        models=("claude-haiku-4-5", "claude-sonnet-4-6"),
        research_budget_tokens=512,
        precision_gate=0.50,
        bootstrap_n=2000,
        permutation_n=2000,
        out_dir=tmp_path,
        load_real=False,
        use_docker=False,
    )
    assert headline["outcome_per_instance"]["p_value"] < 0.05
    # 6 paired means → 2^6 = 64 perms → smallest p ≈ 1/32 = 0.03125.
    assert headline["outcome_per_seed_mean"]["p_value"] <= 0.05
    assert headline["target_met"] is True


def test_per_instance_permutation_uses_per_instance_pairs(tmp_path: Path):
    """The two-test design (per-instance AND per-seed) must report
    different n_pairs so reviewers can see we are not double-counting.
    """
    headline = run_ab(
        n_examples=15,
        seeds=(0, 1),
        models=("claude-haiku-4-5",),
        research_budget_tokens=512,
        precision_gate=0.50,
        bootstrap_n=200,
        permutation_n=200,
        out_dir=tmp_path,
        load_real=False,
        use_docker=False,
    )
    assert headline["outcome_per_instance"]["n_pairs"] == 30
    assert headline["outcome_per_seed_mean"]["n_pairs"] == 2
