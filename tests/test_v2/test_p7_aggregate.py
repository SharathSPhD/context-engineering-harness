"""Tests for the P7 aggregator: every figure, every table, the
omnibus combination, and the partial-input fallback.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from experiments.v2.p7 import aggregate as agg


# ---------------------------------------------------------------------------
# Stouffer machinery
# ---------------------------------------------------------------------------


def test_norm_inv_cdf_round_trips():
    for q in [0.001, 0.025, 0.5, 0.975, 0.999]:
        z = agg._norm_inv_cdf(q)
        recov = 1.0 - agg._norm_sf(z)
        assert recov == pytest.approx(q, rel=1e-3, abs=1e-3)


def test_stouffer_p_collapses_when_one_study_dominates():
    out = agg.stouffer_combine([1e-12], [1.0])
    assert out["n_studies"] == 1
    assert out["p_combined_two_sided"] < 1e-6


def test_stouffer_p_softens_with_neutral_evidence():
    out_strong = agg.stouffer_combine([1e-6, 1e-6, 1e-6], [1, 1, 1])
    out_mixed = agg.stouffer_combine([1e-6, 0.5, 0.5], [1, 1, 1])
    assert out_strong["z"] > out_mixed["z"]
    assert out_strong["p_combined_two_sided"] < out_mixed["p_combined_two_sided"]


def test_stouffer_handles_zero_and_one_clamping():
    out = agg.stouffer_combine([0.0, 1.0], [1.0, 1.0])
    assert math.isfinite(out["z"])
    assert math.isfinite(out["p_combined_two_sided"])


# ---------------------------------------------------------------------------
# End-to-end aggregation against the real artifact tree
# ---------------------------------------------------------------------------


def test_aggregate_emits_every_figure_and_table(tmp_path: Path):
    out = agg.aggregate(out_dir=tmp_path)

    expected_figs = [name for name, _ in agg._FIGURES]
    expected_tabs = [name for name, _ in agg._TABLES]
    assert set(out["figures"].keys()) == set(expected_figs)
    assert set(out["tables"].keys()) == set(expected_tabs)

    n_emitted = sum(1 for v in out["figures"].values() if v is not None)
    assert n_emitted == len(expected_figs), (
        f"all 13 figures should emit on the live artifact tree, got {n_emitted}"
    )
    for name in expected_tabs:
        md_rel = out["tables"][name]["md"]
        csv_rel = out["tables"][name]["csv"]
        assert (tmp_path / md_rel).exists()
        assert (tmp_path / csv_rel).exists()
        assert (tmp_path / md_rel).read_text().count("|") >= 4

    for name, rel in out["figures"].items():
        assert rel is not None, f"{name} did not emit"
        path = tmp_path / rel
        assert path.exists() and path.suffix == ".png"
        assert path.stat().st_size > 1000, f"{name} png is suspiciously small"


def test_aggregate_omnibus_passes_significance(tmp_path: Path):
    out = agg.aggregate(out_dir=tmp_path)
    assert out["n_total_studies"] >= 7
    assert out["n_significant_p_lt_0p05"] >= 7
    om = out["omnibus_stouffer"]
    assert om["z"] > 5.0  # 10+ studies all p<0.05 → very large Stouffer Z
    assert om["p_combined_two_sided"] < 1e-10


def test_aggregate_index_and_summary_are_well_formed(tmp_path: Path):
    out = agg.aggregate(out_dir=tmp_path)
    idx_path = tmp_path / "_index.json"
    summary_path = tmp_path / "_summary.md"
    assert idx_path.exists() and summary_path.exists()
    payload = json.loads(idx_path.read_text())
    assert payload["label"] == "P7_analysis_index"
    assert "ts" in payload
    body = summary_path.read_text()
    assert "Stouffer combined Z" in body
    for name, _ in agg._FIGURES:
        assert f"**{name}**" in body
    for name, _ in agg._TABLES:
        assert f"**{name}**" in body


# ---------------------------------------------------------------------------
# Partial-input resilience: missing every artifact must not crash.
# ---------------------------------------------------------------------------


def test_aggregate_with_missing_artifacts_emits_skeleton(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(agg, "P6A", tmp_path / "missing_p6a")
    monkeypatch.setattr(agg, "P6B", tmp_path / "missing_p6b")
    monkeypatch.setattr(agg, "P6C", tmp_path / "missing_p6c")
    monkeypatch.setattr(agg, "H6_AGREEMENT", tmp_path / "missing_h6.json")

    out = agg.aggregate(out_dir=tmp_path / "p7out")
    assert out["n_artifacts_loaded"] == 0
    assert out["omnibus_stouffer"]["n_studies"] == 0
    # Every table still emits (even if empty), every figure returns None.
    assert all(v is None for v in out["figures"].values())
    for name, _ in agg._TABLES:
        assert (tmp_path / "p7out" / out["tables"][name]["md"]).exists()


# ---------------------------------------------------------------------------
# Per-table content checks
# ---------------------------------------------------------------------------


def test_T4_table_carries_expected_swebench_metrics(tmp_path: Path):
    agg.aggregate(out_dir=tmp_path)
    md = (tmp_path / "tables" / "T4_p6c_swebench_ab_headline.md").read_text()
    for needle in [
        "treatment_metric_mean",
        "baseline_metric_mean",
        "per_instance_delta",
        "per_instance_p_value",
        "per_instance_cohens_d",
        "treatment_target_path_hit_rate",
        "total_sublations_fired",
        "target_met",
    ]:
        assert needle in md


def test_T6_table_contains_overall_kappa_and_per_class(tmp_path: Path):
    agg.aggregate(out_dir=tmp_path)
    md = (tmp_path / "tables" / "T6_khyativada_iaa.md").read_text()
    assert "overall_kappa" in md
    assert "kappa_band" in md
    for cls in ("anyathakhyati", "viparitakhyati", "atmakhyati", "asatkhyati"):
        assert f"kappa[{cls}]" in md


def test_T7_omnibus_lists_per_study_rows(tmp_path: Path):
    agg.aggregate(out_dir=tmp_path)
    md = (tmp_path / "tables" / "T7_omnibus_stouffer.md").read_text()
    assert "combined_z" in md
    assert "combined_p_two_sided" in md
    assert "per-study" in md
    assert md.count("p=") >= 7  # at least one row per surviving study


# ---------------------------------------------------------------------------
# Determinism: two consecutive runs produce byte-identical _index.json
# (excluding the timestamp field, which is allowed to change).
# ---------------------------------------------------------------------------


def test_aggregate_is_deterministic_modulo_timestamp(tmp_path: Path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    payload_a = agg.aggregate(out_dir=a_dir)
    payload_b = agg.aggregate(out_dir=b_dir)
    payload_a.pop("ts", None)
    payload_b.pop("ts", None)
    assert json.dumps(payload_a, sort_keys=True) == json.dumps(payload_b, sort_keys=True)
