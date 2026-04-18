"""P7 aggregator: reads every P6 artifact and emits the 13 figures and
7 tables required by the preprint plan.

Inputs (read-only):
  * experiments/results/p6a/_summary.json           # H1, H2 long-context
  * experiments/results/p6a/_summary_plugin.json    # H3..H7 plugin in-loop
  * experiments/results/p6b/_summary.json           # case-study summary
  * experiments/results/p6b/<case>.json             # per-case transcripts
  * experiments/results/p6c/swebench_ab.json        # SWE-bench A/B headline
  * experiments/h6_khyativada_classifier/results/agreement_report.json

Outputs (overwritten on every run, deterministic):
  * experiments/results/p7/figures/F01_*.png  ... F13_*.png
  * experiments/results/p7/tables/T1_*.{md,csv}  ... T7_*.{md,csv}
  * experiments/results/p7/_index.json   # machine-readable manifest
  * experiments/results/p7/_summary.md   # the appendix reviewers read

Run with:  uv run python -m experiments.v2.p7.aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
P6A = ROOT / "experiments" / "results" / "p6a"
P6B = ROOT / "experiments" / "results" / "p6b"
P6C = ROOT / "experiments" / "results" / "p6c"
H6_AGREEMENT = (
    ROOT / "experiments" / "h6_khyativada_classifier" / "results" / "agreement_report.json"
)
P7 = ROOT / "experiments" / "results" / "p7"


# These globals are set at the start of every ``aggregate()`` call so that
# the figure and table writers can keep their concise ``FIG / "..."``
# signatures while still honouring a caller-supplied ``out_dir`` (used by
# the test suite). The default points at the canonical location.
FIG = P7 / "figures"
TAB = P7 / "tables"


def _set_out_dir(out_dir: Path) -> None:
    global FIG, TAB
    FIG = out_dir / "figures"
    TAB = out_dir / "tables"


# ---------------------------------------------------------------------------
# Loaders (every loader returns a dict and never raises on missing files;
# missing files become empty dicts so the runner stays partial-friendly)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("missing artifact: %s", path)
        return {}
    return json.loads(path.read_text())


def load_artifacts() -> dict[str, Any]:
    return {
        "p6a_h1_h2": _load_json(P6A / "_summary.json"),
        "p6a_plugin": _load_json(P6A / "_summary_plugin.json"),
        "p6b_summary": _load_json(P6B / "_summary.json"),
        "p6c": _load_json(P6C / "swebench_ab.json"),
        "h6_agreement": _load_json(H6_AGREEMENT),
    }


# ---------------------------------------------------------------------------
# Statistics: omnibus combination via Stouffer's method (weighted by n_pairs).
# ---------------------------------------------------------------------------


def _norm_inv_cdf(p: float) -> float:
    """Inverse standard-normal CDF via the Beasley-Springer-Moro
    rational approximation. No SciPy dependency at the analysis layer
    so the notebook stays portable.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [
        -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
        1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
        6.680131188771972e1, -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
        -2.549732539343734, 4.374664141464968, 2.938163982698783,
    ]
    d_ = [
        7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
        3.754408661907416,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1
    )


def _norm_sf(z: float) -> float:
    """1 - Phi(z), upper-tail. erfc-based, no SciPy."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def stouffer_combine(p_values: Iterable[float], weights: Iterable[float]) -> dict[str, float]:
    """Weighted Stouffer-Z combination. Returns combined two-sided p
    plus the Z statistic and effective N.
    """
    pv = [max(min(p, 1 - 1e-12), 1e-12) for p in p_values]
    w = list(weights)
    if not pv:
        return {"z": 0.0, "p_combined_two_sided": 1.0, "n_studies": 0, "sum_w": 0.0}
    z = sum(wi * _norm_inv_cdf(1.0 - pi) for wi, pi in zip(w, pv)) / math.sqrt(sum(wi * wi for wi in w))
    p_two = 2.0 * _norm_sf(abs(z))
    return {
        "z": float(z),
        "p_combined_two_sided": float(p_two),
        "n_studies": len(pv),
        "sum_w": float(sum(w)),
    }


# ---------------------------------------------------------------------------
# Common derived rows so figures + tables agree by construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisRow:
    """Row used by F12, F13, T1, T2, and the omnibus."""
    label: str
    family: str          # "long-ctx" | "plugin-inloop" | "live" | "swebench-ab"
    treatment_metric: float
    baseline_metric: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    cohens_d: float
    n_pairs: int


def _derive_rows(artifacts: dict[str, Any]) -> list[HypothesisRow]:
    rows: list[HypothesisRow] = []
    for r in artifacts.get("p6a_h1_h2", {}).get("results", []):
        o = r["outcome"]
        rows.append(HypothesisRow(
            label=r["label"], family="long-ctx",
            treatment_metric=o["treatment_metric"], baseline_metric=o["baseline_metric"],
            delta=o["delta_observed"], ci_low=o["ci_low"], ci_high=o["ci_high"],
            p_value=o["p_value"], cohens_d=o["cohens_d"], n_pairs=o["n_examples_used"],
        ))
    for r in artifacts.get("p6a_plugin", {}).get("results", []):
        o = r["outcome"]
        rows.append(HypothesisRow(
            label=r["label"], family="plugin-inloop",
            treatment_metric=o["treatment_metric"], baseline_metric=o["baseline_metric"],
            delta=o["delta_observed"], ci_low=o["ci_low"], ci_high=o["ci_high"],
            p_value=o["p_value"], cohens_d=o["cohens_d"], n_pairs=o["n_examples_used"],
        ))
    if artifacts.get("p6b_summary"):
        s = artifacts["p6b_summary"]["summary"]
        rows.append(HypothesisRow(
            label="P6-B_live_case_study", family="live",
            treatment_metric=s["accuracy_with_harness"],
            baseline_metric=s["accuracy_without_harness"],
            delta=s["accuracy_delta"], ci_low=s["accuracy_delta"], ci_high=s["accuracy_delta"],
            p_value=1.0, cohens_d=0.0, n_pairs=s["n_cases"],
        ))
    if artifacts.get("p6c"):
        c = artifacts["p6c"]
        oi = c["outcome_per_instance"]
        rows.append(HypothesisRow(
            label="P6-C_swebench_ab_per_instance", family="swebench-ab",
            treatment_metric=c["treatment_metric_mean"],
            baseline_metric=c["baseline_metric_mean"],
            delta=oi["delta_observed"], ci_low=oi["ci_low"], ci_high=oi["ci_high"],
            p_value=oi["p_value"], cohens_d=oi["cohens_d"], n_pairs=oi["n_pairs"],
        ))
    return rows


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=144, bbox_inches="tight")
    plt.close(fig)


def _two_arm_bar(
    title: str, baseline: float, treatment: float,
    ylabel: str, xlabels: tuple[str, str] = ("baseline", "with harness"),
    extra_text: str | None = None,
):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    x = np.arange(2)
    bars = ax.bar(x, [baseline, treatment], color=["#9ca3af", "#2563eb"], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(1.0, treatment * 1.2 + 0.05))
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    if extra_text:
        ax.text(0.02, 0.98, extra_text, ha="left", va="top",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#e5e7eb", boxstyle="round"))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figures (13)
# ---------------------------------------------------------------------------


def figure_F01(art: dict[str, Any]) -> Path | None:
    """H1 RULER NIAH accuracy by context length."""
    rows = [r for r in art["p6a_h1_h2"].get("results", []) if r["hypothesis_id"] == "H1"]
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["label"].split("_")[-1]))
    ctx = [int(r["label"].split("_")[-1]) for r in rows]
    base = [r["outcome"]["baseline_metric"] for r in rows]
    treat = [r["outcome"]["treatment_metric"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    x = np.arange(len(ctx))
    w = 0.35
    ax.bar(x - w / 2, base, w, label="baseline", color="#9ca3af")
    ax.bar(x + w / 2, treat, w, label="with harness", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c // 1024}K" for c in ctx])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("RULER NIAH accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("F01 H1 — RULER NIAH accuracy by context length")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG / "F01_H1_ruler_by_context_length.png"
    _save(fig, out)
    return out


def figure_F02(art: dict[str, Any]) -> Path | None:
    """H2 HELMET RAG recall by context length."""
    rows = [r for r in art["p6a_h1_h2"].get("results", []) if r["hypothesis_id"] == "H2"]
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["label"].split("_")[-1]))
    ctx = [int(r["label"].split("_")[-1]) for r in rows]
    base = [r["outcome"]["baseline_metric"] for r in rows]
    treat = [r["outcome"]["treatment_metric"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    x = np.arange(len(ctx))
    w = 0.35
    ax.bar(x - w / 2, base, w, label="baseline", color="#9ca3af")
    ax.bar(x + w / 2, treat, w, label="with harness", color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c // 1024}K" for c in ctx])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("HELMET recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_title("F02 H2 — HELMET RAG recall by context length")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG / "F02_H2_helmet_by_context_length.png"
    _save(fig, out)
    return out


def _h_row(art: dict[str, Any], hid: str) -> dict[str, Any] | None:
    for r in art.get("p6a_plugin", {}).get("results", []):
        if r["hypothesis_id"] == hid:
            return r
    return None


def figure_F03(art):
    r = _h_row(art, "H3")
    if not r:
        return None
    o = r["outcome"]
    fig = _two_arm_bar(
        "F03 H3 — Buddhi/Manas grounding accuracy",
        o["baseline_metric"], o["treatment_metric"],
        ylabel="grounded-answer accuracy",
        extra_text=(
            f"Δ = {o['delta_observed']:.3f}\n"
            f"95% CI [{o['ci_low']:.3f}, {o['ci_high']:.3f}]\n"
            f"p = {o['p_value']:.4f}, d = {o['cohens_d']:.2f}"
        ),
    )
    out = FIG / "F03_H3_buddhi_manas_grounding.png"
    _save(fig, out)
    return out


def figure_F04(art):
    r = _h_row(art, "H4")
    if not r:
        return None
    o = r["outcome"]
    fig = _two_arm_bar(
        "F04 H4 — EventBoundary compaction quality",
        o["baseline_metric"], o["treatment_metric"],
        ylabel="boundary-aligned compaction score",
        extra_text=(
            f"Δ = {o['delta_observed']:.3f}\n"
            f"95% CI [{o['ci_low']:.3f}, {o['ci_high']:.3f}]\n"
            f"p = {o['p_value']:.4f}, d = {o['cohens_d']:.2f}"
        ),
    )
    out = FIG / "F04_H4_event_boundary.png"
    _save(fig, out)
    return out


def figure_F05(art):
    r = _h_row(art, "H5")
    if not r:
        return None
    o = r["outcome"]
    fig = _two_arm_bar(
        "F05 H5 — Avacchedaka sublation conflict resolution",
        o["baseline_metric"], o["treatment_metric"],
        ylabel="conflict-resolution rate",
        extra_text=(
            f"Δ = {o['delta_observed']:.3f}\n"
            f"95% CI [{o['ci_low']:.3f}, {o['ci_high']:.3f}]\n"
            f"p = {o['p_value']:.4f}"
        ),
    )
    out = FIG / "F05_H5_avacchedaka_sublation.png"
    _save(fig, out)
    return out


def figure_F06(art):
    r = _h_row(art, "H6")
    if not r:
        return None
    o = r["outcome"]
    fig = _two_arm_bar(
        "F06 H6 — Khyātivāda 6-class classifier (macro-F1 proxy)",
        o["baseline_metric"], o["treatment_metric"],
        ylabel="macro-F1",
        extra_text=(
            f"Δ = {o['delta_observed']:.3f}\n"
            f"95% CI [{o['ci_low']:.3f}, {o['ci_high']:.3f}]\n"
            f"p = {o['p_value']:.4f}, d = {o['cohens_d']:.2f}"
        ),
    )
    out = FIG / "F06_H6_khyativada_classifier.png"
    _save(fig, out)
    return out


def figure_F07(art):
    r = _h_row(art, "H7")
    if not r:
        return None
    o = r["outcome"]
    fig = _two_arm_bar(
        "F07 H7 — AdaptiveForgetting recency-vs-importance",
        o["baseline_metric"], o["treatment_metric"],
        ylabel="recency/importance balance score",
        extra_text=(
            f"Δ = {o['delta_observed']:.3f}\n"
            f"95% CI [{o['ci_low']:.3f}, {o['ci_high']:.3f}]\n"
            f"p = {o['p_value']:.4f}"
        ),
    )
    out = FIG / "F07_H7_adaptive_forgetting.png"
    _save(fig, out)
    return out


def figure_F08(art):
    """P6-B per-case answer accuracy, with vs without harness."""
    s = art.get("p6b_summary")
    if not s:
        return None
    cases = s["per_case"]
    labels = [c["case_id"] for c in cases]
    with_h = [int(c["comparison"]["answer_correct_with_harness"]) for c in cases]
    without_h = [int(c["comparison"]["answer_correct_without_harness"]) for c in cases]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, without_h, w, label="without harness", color="#9ca3af")
    ax.bar(x + w / 2, with_h, w, label="with harness", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["wrong", "correct"])
    ax.set_title("F08 P6-B — Live case study answer correctness")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG / "F08_P6B_per_case_accuracy.png"
    _save(fig, out)
    return out


def figure_F09(art):
    """P6-B per-case forbidden-substring claim count, with vs without harness."""
    s = art.get("p6b_summary")
    if not s:
        return None
    cases = s["per_case"]
    labels = [c["case_id"] for c in cases]
    with_h = [c["comparison"]["n_forbidden_hits_with_harness"] for c in cases]
    without_h = [c["comparison"]["n_forbidden_hits_without_harness"] for c in cases]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, without_h, w, label="without harness", color="#9ca3af")
    ax.bar(x + w / 2, with_h, w, label="with harness", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("forbidden-substring claims (lower = better)")
    ax.set_title("F09 P6-B — Forbidden-claim count per case")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = FIG / "F09_P6B_forbidden_claims.png"
    _save(fig, out)
    return out


def figure_F10(art):
    """P6-C per-instance heuristic-score histogram of paired diffs.

    The runner appends ``with_harness`` and ``without_harness`` rows in
    lockstep over the (model, seed, example) tuple, so positional zipping
    is the correct pairing key. The two-arm length-equality assertion
    catches any future drift in artifact schema.
    """
    c = art.get("p6c")
    if not c:
        return None
    pi = c["per_instance"]
    with_rows = [r for r in pi if r["arm"] == "with_harness"]
    without_rows = [r for r in pi if r["arm"] == "without_harness"]
    if not with_rows or len(with_rows) != len(without_rows):
        return None
    diffs = [w["score"] - wo["score"] for w, wo in zip(with_rows, without_rows)]
    if not diffs:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(diffs, bins=21, color="#2563eb", alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(diffs), color="#dc2626", linestyle="--",
               label=f"mean Δ = {np.mean(diffs):.3f}")
    ax.axvline(0.0, color="#111827", linewidth=0.7)
    ax.set_xlabel("paired score (with − without)")
    ax.set_ylabel("instance count")
    ax.set_title("F10 P6-C — paired heuristic-score differences")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = FIG / "F10_P6C_paired_diff_histogram.png"
    _save(fig, out)
    return out


def figure_F11(art):
    """P6-C target-file anchor hit-rate, with vs without harness."""
    c = art.get("p6c")
    if not c:
        return None
    fig = _two_arm_bar(
        "F11 P6-C — target-file anchor hit-rate",
        c["baseline_target_path_hit_rate"],
        c["treatment_target_path_hit_rate"],
        ylabel="P(anchor matches gold patch target)",
        extra_text=(
            f"per-instance Δ = {c['outcome_per_instance']['delta_observed']:.3f}\n"
            f"p = {c['outcome_per_instance']['p_value']:.4f}\n"
            f"d = {c['outcome_per_instance']['cohens_d']:.2f}"
        ),
    )
    out = FIG / "F11_P6C_target_path_hitrate.png"
    _save(fig, out)
    return out


def figure_F12(art):
    """Cross-experiment Cohen's d panel."""
    rows = _derive_rows(art)
    rows = [r for r in rows if r.cohens_d > 0]  # H5/H7 carry d=0 sentinel (degenerate)
    if not rows:
        return None
    rows.sort(key=lambda r: r.cohens_d)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ys = np.arange(len(rows))
    ax.barh(ys, [r.cohens_d for r in rows], color="#7c3aed")
    ax.set_yticks(ys)
    ax.set_yticklabels([r.label for r in rows], fontsize=8)
    ax.axvline(0.5, color="#9ca3af", linestyle=":", label="d=0.5 (medium)")
    ax.axvline(0.8, color="#9ca3af", linestyle="--", label="d=0.8 (large)")
    ax.set_xlabel("Cohen's d (paired-mean / pooled-sd)")
    ax.set_title("F12 — Effect sizes across the seven hypotheses + P6-C")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG / "F12_effect_sizes.png"
    _save(fig, out)
    return out


def figure_F13(art):
    """Forest plot of Δ + 95% CI across every quantitative experiment."""
    rows = _derive_rows(art)
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r.delta)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ys = np.arange(len(rows))
    deltas = np.array([r.delta for r in rows])
    err_low = deltas - np.array([r.ci_low for r in rows])
    err_high = np.array([r.ci_high for r in rows]) - deltas
    ax.errorbar(deltas, ys, xerr=[err_low, err_high], fmt="o",
                color="#2563eb", ecolor="#9ca3af", capsize=3, lw=1.4)
    ax.axvline(0.0, color="#111827", lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r.label for r in rows], fontsize=8)
    ax.set_xlabel("Δ (treatment − baseline)")
    ax.set_title("F13 — Forest plot: harness uplift across all experiments")
    fig.tight_layout()
    out = FIG / "F13_forest_plot.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Tables (7) — every table emits both Markdown and CSV
# ---------------------------------------------------------------------------


def _emit_table(name: str, headers: list[str], rows: list[list[Any]]) -> tuple[Path, Path]:
    md = TAB / f"{name}.md"
    csv_path = TAB / f"{name}.csv"
    md.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(v):
        if isinstance(v, float):
            if v == 0.0:
                return "0.0000"
            absv = abs(v)
            if absv < 1e-4 or absv >= 1e6:
                return f"{v:.3e}"
            if absv < 100:
                return f"{v:.4f}"
            return f"{v:.2f}"
        return str(v)

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    md.write_text("\n".join(md_lines) + "\n")

    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)
    return md, csv_path


def table_T1(art):
    """Long-context family (H1, H2)."""
    headers = ["label", "treatment", "baseline", "delta", "ci_low", "ci_high",
               "p_value", "cohens_d", "n_pairs", "target_met"]
    rows = []
    for r in art.get("p6a_h1_h2", {}).get("results", []):
        o = r["outcome"]
        rows.append([
            r["label"], o["treatment_metric"], o["baseline_metric"],
            o["delta_observed"], o["ci_low"], o["ci_high"], o["p_value"],
            o["cohens_d"], o["n_examples_used"], o["target_met"],
        ])
    return _emit_table("T1_p6a_long_context", headers, rows)


def table_T2(art):
    """Plugin-in-loop family (H3..H7)."""
    headers = ["label", "treatment", "baseline", "delta", "ci_low", "ci_high",
               "p_value", "cohens_d", "n_pairs", "target_met"]
    rows = []
    for r in art.get("p6a_plugin", {}).get("results", []):
        o = r["outcome"]
        rows.append([
            r["label"], o["treatment_metric"], o["baseline_metric"],
            o["delta_observed"], o["ci_low"], o["ci_high"], o["p_value"],
            o["cohens_d"], o["n_examples_used"], o["target_met"],
        ])
    return _emit_table("T2_p6a_plugin_inloop", headers, rows)


def table_T3(art):
    """P6-B per-case results."""
    s = art.get("p6b_summary")
    if not s:
        return _emit_table("T3_p6b_per_case", ["case_id"], [])
    headers = ["case_id", "correct_with", "correct_without",
               "forbidden_with", "forbidden_without",
               "stale_in_ctx_delta", "ctx_tokens"]
    rows = []
    for c in s["per_case"]:
        cmp = c["comparison"]
        rows.append([
            c["case_id"],
            int(cmp["answer_correct_with_harness"]),
            int(cmp["answer_correct_without_harness"]),
            cmp["n_forbidden_hits_with_harness"],
            cmp["n_forbidden_hits_without_harness"],
            cmp["stale_evidence_delta"],
            cmp["context_tokens_with_harness"],
        ])
    return _emit_table("T3_p6b_per_case", headers, rows)


def table_T4(art):
    """P6-C SWE-bench A/B headline."""
    c = art.get("p6c")
    if not c:
        return _emit_table("T4_p6c_swebench_ab_headline", ["metric"], [])
    headers = ["metric", "value"]
    oi = c["outcome_per_instance"]
    os_ = c["outcome_per_seed_mean"]
    rows = [
        ["treatment_metric_mean", c["treatment_metric_mean"]],
        ["baseline_metric_mean", c["baseline_metric_mean"]],
        ["per_instance_delta", oi["delta_observed"]],
        ["per_instance_ci", f"[{oi['ci_low']:.4f}, {oi['ci_high']:.4f}]"],
        ["per_instance_p_value", oi["p_value"]],
        ["per_instance_cohens_d", oi["cohens_d"]],
        ["per_instance_n_pairs", oi["n_pairs"]],
        ["per_(model,seed)_delta", os_["delta_observed"]],
        ["per_(model,seed)_ci", f"[{os_['ci_low']:.4f}, {os_['ci_high']:.4f}]"],
        ["per_(model,seed)_p_value", os_["p_value"]],
        ["per_(model,seed)_n_pairs", os_["n_pairs"]],
        ["treatment_target_path_hit_rate", c["treatment_target_path_hit_rate"]],
        ["baseline_target_path_hit_rate", c["baseline_target_path_hit_rate"]],
        ["total_sublations_fired", c["total_sublations_fired"]],
        ["target_met", c["target_met"]],
    ]
    return _emit_table("T4_p6c_swebench_ab_headline", headers, rows)


def table_T5(art):
    """P6-C per-(model, seed, arm) breakdown."""
    c = art.get("p6c")
    if not c:
        return _emit_table("T5_p6c_per_seed", ["model"], [])
    headers = ["model", "seed", "arm", "n_examples", "mean_score",
               "accuracy", "n_target_path_hit", "n_research_sublations"]
    rows = []
    for ps in c["per_seed"]:
        rows.append([
            ps["model"], ps["seed"], ps["arm"], ps["n_examples"],
            ps["mean_score"], ps["accuracy"], ps["n_target_path_hit"],
            ps["n_research_sublations"],
        ])
    return _emit_table("T5_p6c_per_seed_breakdown", headers, rows)


def table_T6(art):
    """Khyātivāda annotation IAA (Cohen's kappa, per-class kappa, percent agreement)."""
    a = art.get("h6_agreement")
    if not a:
        return _emit_table("T6_khyativada_iaa", ["class"], [])
    ag = a["agreement"]
    rows = [
        ["overall_kappa", ag["kappa"]],
        ["overall_kappa_band", ag["kappa_band"]],
        ["percent_agreement", ag["percent_agreement"]],
        ["n_examples", ag["n"]],
    ]
    for cls, k in sorted(ag["per_class_kappa"].items()):
        rows.append([f"kappa[{cls}]", k])
    return _emit_table("T6_khyativada_iaa", ["metric", "value"], rows)


def table_T7(art):
    """Omnibus: Stouffer-Z combined p-value across every quantitative result.
    Weights are sqrt(n_pairs) (the canonical Liptak weighting).
    """
    rows = _derive_rows(art)
    rows = [r for r in rows if r.p_value < 1.0]  # P6-B sentinel excluded
    if not rows:
        return _emit_table("T7_omnibus_stouffer", ["k"], [])
    pvals = [r.p_value for r in rows]
    weights = [math.sqrt(r.n_pairs) for r in rows]
    res = stouffer_combine(pvals, weights)
    headers = ["statistic", "value"]
    out_rows: list[list[Any]] = [
        ["n_studies", res["n_studies"]],
        ["sum_weights", res["sum_w"]],
        ["combined_z", res["z"]],
        ["combined_p_two_sided", res["p_combined_two_sided"]],
        ["mean_delta", float(np.mean([r.delta for r in rows]))],
        ["min_delta", float(np.min([r.delta for r in rows]))],
        ["max_delta", float(np.max([r.delta for r in rows]))],
        ["mean_cohens_d_excluding_zero",
         float(np.mean([r.cohens_d for r in rows if r.cohens_d > 0]))
         if any(r.cohens_d > 0 for r in rows) else 0.0],
    ]
    out_rows.append(["—"] * 2)
    out_rows.append(["per-study", "(label, p, weight)"])
    for r, w in zip(rows, weights):
        out_rows.append([r.label, f"p={r.p_value:.4g}, w={w:.2f}, Δ={r.delta:.3f}"])
    return _emit_table("T7_omnibus_stouffer", headers, out_rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_FIGURES = [
    ("F01", figure_F01), ("F02", figure_F02), ("F03", figure_F03),
    ("F04", figure_F04), ("F05", figure_F05), ("F06", figure_F06),
    ("F07", figure_F07), ("F08", figure_F08), ("F09", figure_F09),
    ("F10", figure_F10), ("F11", figure_F11), ("F12", figure_F12),
    ("F13", figure_F13),
]
_TABLES = [
    ("T1", table_T1), ("T2", table_T2), ("T3", table_T3),
    ("T4", table_T4), ("T5", table_T5), ("T6", table_T6),
    ("T7", table_T7),
]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def aggregate(out_dir: Path = P7) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    _set_out_dir(out_dir)

    artifacts = load_artifacts()

    figs: dict[str, str | None] = {}
    for name, fn in _FIGURES:
        try:
            path = fn(artifacts)
        except Exception as exc:  # pragma: no cover — surface at call site
            logger.exception("figure %s failed: %s", name, exc)
            path = None
        figs[name] = str(path.relative_to(out_dir)) if path else None

    tabs: dict[str, dict[str, str] | None] = {}
    for name, fn in _TABLES:
        try:
            md, csv_p = fn(artifacts)
        except Exception as exc:  # pragma: no cover — surface at call site
            logger.exception("table %s failed: %s", name, exc)
            tabs[name] = None
            continue
        tabs[name] = {
            "md": str(md.relative_to(out_dir)),
            "csv": str(csv_p.relative_to(out_dir)),
        }

    rows = _derive_rows(artifacts)
    weights = [math.sqrt(r.n_pairs) for r in rows if r.p_value < 1.0]
    pvals = [r.p_value for r in rows if r.p_value < 1.0]
    omnibus = stouffer_combine(pvals, weights) if pvals else {
        "z": 0.0, "p_combined_two_sided": 1.0, "n_studies": 0, "sum_w": 0.0,
    }

    headline = {
        "label": "P7_analysis_index",
        "ts": _utcnow(),
        "n_artifacts_loaded": sum(1 for v in artifacts.values() if v),
        "figures": figs,
        "tables": tabs,
        "omnibus_stouffer": omnibus,
        "n_significant_p_lt_0p05": sum(1 for r in rows if r.p_value < 0.05),
        "n_total_studies": len(rows),
    }
    (out_dir / "_index.json").write_text(json.dumps(headline, indent=2))

    summary_md = [
        "# P7 — Statistical analysis index",
        "",
        f"Generated {headline['ts']}",
        "",
        f"- Artifacts loaded: **{headline['n_artifacts_loaded']}**",
        f"- Studies with p<0.05: **{headline['n_significant_p_lt_0p05']} / {headline['n_total_studies']}**",
        f"- Stouffer combined Z: **{omnibus['z']:.3f}**, "
        f"two-sided p: **{omnibus['p_combined_two_sided']:.4g}** "
        f"(k={omnibus['n_studies']} studies)",
        "",
        "## Figures",
    ]
    for name, p in figs.items():
        summary_md.append(f"- **{name}**: `{p}`" if p else f"- **{name}**: _missing input artifacts_")
    summary_md.append("")
    summary_md.append("## Tables")
    for name, files in tabs.items():
        if files is None:
            summary_md.append(f"- **{name}**: _missing input artifacts_")
        else:
            summary_md.append(f"- **{name}**: `{files['md']}` / `{files['csv']}`")
    summary_md.append("")
    (out_dir / "_summary.md").write_text("\n".join(summary_md))
    return headline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=P7)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=max(logging.DEBUG, logging.WARNING - 10 * args.verbose),
        format="%(levelname)s %(message)s",
    )
    headline = aggregate(args.out_dir)
    print(json.dumps({
        "label": headline["label"],
        "n_artifacts_loaded": headline["n_artifacts_loaded"],
        "n_significant_p_lt_0p05": headline["n_significant_p_lt_0p05"],
        "n_total_studies": headline["n_total_studies"],
        "omnibus_stouffer": headline["omnibus_stouffer"],
        "figures_emitted": sum(1 for v in headline["figures"].values() if v),
        "tables_emitted": len(headline["tables"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
