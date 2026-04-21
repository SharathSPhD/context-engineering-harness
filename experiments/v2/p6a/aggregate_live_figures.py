"""P6A live-HF figure aggregator: reads _summary_live.json + swe_bench_outcomes.json
and emits F14 (live-HF forest plot) and F15 (RULER 16K ext per-model panel) into
paper/figures/.

Inputs (read-only):
  * experiments/results/p6a/_summary_live.json       # all 7 live-HF bundles
                                                     # (4 v2.1 core4 + 3 v2.1.1 ext)
  * experiments/results/p6a/swe_bench_outcomes.json  # v2.1 SWE-bench haiku paired slice

Outputs (overwritten on every run, deterministic):
  * paper/figures/F14_live_hf_forest.png
  * paper/figures/F15_ruler16k_ext_per_model.png

Run with:  uv run python -m experiments.v2.p6a.aggregate_live_figures
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
P6A = ROOT / "experiments" / "results" / "p6a"
FIG = ROOT / "paper" / "figures"

# Style: match F12/F13 in experiments/v2/p7/aggregate.py.
COLOR_GREY = "#9ca3af"      # baseline / null
COLOR_BLUE = "#2563eb"      # default marker
COLOR_GREEN = "#16a34a"     # preregistered combined gate met (d_z >= 0.5 AND p <= 0.05)
COLOR_AMBER = "#d97706"     # directional / CI-excludes-zero, gate shy
COLOR_PURPLE = "#7c3aed"    # v2.1.1 band highlight
COLOR_AXIS = "#111827"
COLOR_LIGHT = "#e5e7eb"
COLOR_HOLLOW_EDGE = "#6b7280"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required artefact missing: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# CI helpers
# ---------------------------------------------------------------------------


def wald_ci_from_d(delta: float, dz: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Symmetric Wald CI for paired mean difference, reconstructed from
    Cohen's dz (= mean_diff / sd_diff) and the paired-observation count.

    Used only for rows that lack a bootstrap CI in _summary_live.json (i.e. the
    v2.1 core4 bundles, which were scored by run_live_hf.py with
    bootstrap-n=2000 but the CI was not persisted into the summary). The
    v2.1.1 _ext rows carry bootstrap CIs (bootstrap-n=10000) from
    score_ext_checkpoints.py and are used verbatim.
    """
    if not dz or not math.isfinite(dz) or n <= 1:
        return (delta, delta)
    sd_diff = abs(delta) / abs(dz)
    se = sd_diff / math.sqrt(n)
    z = 1.959963984540054  # two-sided 95%
    return (delta - z * se, delta + z * se)


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveRow:
    label: str          # short display label on the y-axis
    long_label: str     # full bundle id
    band: str           # "v2.1 core4" or "v2.1.1 ext"
    delta: float
    ci_low: float
    ci_high: float
    p_value: float | None
    cohens_d: float | None
    n_paired: int
    gate_status: str    # "met", "shy", "null", "unexercised"
    ci_source: str      # "bootstrap" or "wald_from_d" or "degenerate"


# Display order (top → bottom on forest plot): v2.1 core4 first, then v2.1.1 ext.
DISPLAY_ORDER = [
    ("H1_ruler_8192_live", "RULER 8K"),
    ("H1_ruler_16384_live", "RULER 16K"),
    ("H_TQA_live_v2", "TruthfulQA"),
    ("H_SWEB_live_n15", "SWE-bench (haiku)"),
    ("H1_ruler_16384_live_ext", "RULER 16K (ext, n=103)"),
    ("H_TQA_live_v2_ext", "TruthfulQA (ext)"),
    ("H_SWEB_live_ext", "SWE-bench (ext)"),
]

BAND_MAP = {
    "H1_ruler_8192_live": "v2.1 core4",
    "H1_ruler_16384_live": "v2.1 core4",
    "H_TQA_live_v2": "v2.1 core4",
    "H_SWEB_live_n15": "v2.1 core4",
    "H1_ruler_16384_live_ext": "v2.1.1 ext",
    "H_TQA_live_v2_ext": "v2.1.1 ext",
    "H_SWEB_live_ext": "v2.1.1 ext",
}


def _gate_status(
    label: str,
    delta: float | None,
    p_value: float | None,
    cohens_d: float | None,
    n: int,
    has_new_rows: bool,
) -> str:
    """Preregistered combined gate: Cohen's d_z >= 0.5 AND paired-permutation
    p <= 0.05. Both must fire for a row to be "met" (green).

    This matches the paper's Appendix-G phrasing and the paper's Table T8
    status column ("complete, target met" vs. "partial (CLI-blocked)" vs.
    "complete, null"). A row that would have cleared a Delta-and-p-only gate
    but misses the d_z gate (e.g. SWE-bench haiku at d_z = 0.488) is amber
    ("shy"), not green -- so the figure never reads as stronger than the
    paper's own T8 verdict.

    Status keys:
      * "met"         -> d_z >= 0.5 and p <= 0.05 (combined preregistered gate)
      * "shy"         -> Delta > 0 and (d_z < 0.5 or p > 0.05): directional,
                         at least one gate shy
      * "null"        -> Delta <= 0 or clearly non-directional
      * "unexercised" -> no new paired rows billed (SWE-bench ext)
    """
    if label == "H_SWEB_live_ext":
        return "unexercised"
    if delta is None or p_value is None:
        return "unexercised"
    d = cohens_d if cohens_d is not None else 0.0
    if d >= 0.5 and p_value <= 0.05:
        return "met"
    if delta > 0:
        return "shy"
    return "null"


def extract_rows() -> list[LiveRow]:
    summary = _load_json(P6A / "_summary_live.json")
    bundles_by_label = {b["label"]: b for b in summary.get("bundles", [])}
    sweb_outcomes = _load_json(P6A / "swe_bench_outcomes.json")
    rows: list[LiveRow] = []
    for label, short in DISPLAY_ORDER:
        if label == "H_SWEB_live_n15":
            hk = sweb_outcomes["per_model"]["claude-haiku-4-5"]["full_errors_as_zero"]
            delta = float(hk["delta_observed"])
            n = 30
            dz = float(hk["cohens_dz"])
            p = float(hk["paired_permutation_p"])
            ci_low, ci_high = wald_ci_from_d(delta, dz, n)
            rows.append(
                LiveRow(
                    label=short, long_label=label, band=BAND_MAP[label],
                    delta=delta, ci_low=ci_low, ci_high=ci_high,
                    p_value=p, cohens_d=dz, n_paired=n,
                    gate_status=_gate_status(label, delta, p, dz, n, has_new_rows=True),
                    ci_source="wald_from_d",
                )
            )
            continue

        bundle = bundles_by_label.get(label)
        if bundle is None:
            raise KeyError(f"expected bundle {label!r} not found in _summary_live.json")

        outcome = bundle.get("outcome", {}) or {}
        delta = outcome.get("delta_observed")
        p = outcome.get("p_value")
        dz = outcome.get("cohens_d")
        n_paired = outcome.get("n_paired") or outcome.get("n_examples_used") or 0

        if label == "H_SWEB_live_ext":
            rows.append(
                LiveRow(
                    label=short, long_label=label, band=BAND_MAP[label],
                    delta=float(delta) if delta is not None else 0.0,
                    ci_low=0.0, ci_high=0.0,
                    p_value=None, cohens_d=None, n_paired=int(n_paired),
                    gate_status="unexercised",
                    ci_source="degenerate",
                )
            )
            continue

        ci_low = outcome.get("ci_low")
        ci_high = outcome.get("ci_high")
        if ci_low is None or ci_high is None:
            ci_low, ci_high = wald_ci_from_d(float(delta), float(dz), int(n_paired))
            ci_source = "wald_from_d"
        else:
            ci_source = "bootstrap"

        rows.append(
            LiveRow(
                label=short, long_label=label, band=BAND_MAP[label],
                delta=float(delta), ci_low=float(ci_low), ci_high=float(ci_high),
                p_value=float(p) if p is not None else None,
                cohens_d=float(dz) if dz is not None else None,
                n_paired=int(n_paired),
                gate_status=_gate_status(
                    label,
                    float(delta),
                    float(p) if p is not None else None,
                    float(dz) if dz is not None else None,
                    int(n_paired),
                    has_new_rows=True,
                ),
                ci_source=ci_source,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=144, bbox_inches="tight")
    plt.close(fig)


def _gate_colour(status: str) -> tuple[str, str]:
    """Return (facecolor, edgecolor). Hollow markers are used for unexercised."""
    if status == "met":
        return (COLOR_GREEN, COLOR_GREEN)
    if status == "shy":
        return (COLOR_AMBER, COLOR_AMBER)
    if status == "null":
        return (COLOR_GREY, COLOR_GREY)
    return ("white", COLOR_HOLLOW_EDGE)  # unexercised


def figure_F14(rows: list[LiveRow]) -> Path:
    """Forest plot: paired Δ + 95% CI across all 7 live-HF bundles."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    # y-axis: top = first row. Reverse to paint top-down.
    display_rows = list(rows)
    ys = np.arange(len(display_rows))[::-1]

    for y, r in zip(ys, display_rows):
        face, edge = _gate_colour(r.gate_status)
        err_low = r.delta - r.ci_low
        err_high = r.ci_high - r.delta
        if r.gate_status == "unexercised":
            ax.plot(
                [r.delta], [y],
                marker="o", markersize=8,
                markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.4,
                linestyle="none", zorder=3,
            )
        else:
            ax.errorbar(
                [r.delta], [y],
                xerr=[[err_low], [err_high]],
                fmt="o", color=edge, ecolor=edge,
                markerfacecolor=face, markeredgecolor=edge,
                capsize=3, lw=1.4, markersize=7, zorder=3,
            )
        # Annotate n, p, and d_z on the right edge so viewers can see both
        # halves of the preregistered gate without consulting the table.
        if r.gate_status == "unexercised":
            note = f"n={r.n_paired}  (no new rows)"
        else:
            parts = [f"n={r.n_paired}"]
            if r.p_value is not None:
                if r.p_value < 0.001:
                    parts.append("p<0.001")
                else:
                    parts.append(f"p={r.p_value:.3f}")
            if r.cohens_d is not None:
                parts.append(f"d_z={r.cohens_d:+.2f}")
            note = "  ".join(parts)
        ax.text(1.02, y, note, transform=ax.get_yaxis_transform(),
                fontsize=8, va="center", ha="left", color=COLOR_AXIS)

    # Gate guides.
    ax.axvline(0.0, color=COLOR_AXIS, lw=0.7, zorder=1)
    ax.axvline(0.05, color=COLOR_AXIS, lw=0.6, linestyle=":", zorder=1,
               label="Δ = +0.05 magnitude reference (gate is d_z + p)")

    # Band separator between v2.1 core4 and v2.1.1 ext.
    core_count = sum(1 for r in display_rows if r.band == "v2.1 core4")
    ext_count = len(display_rows) - core_count
    if core_count and ext_count:
        sep_y = (ys[core_count - 1] + ys[core_count]) / 2
        ax.axhline(sep_y, color=COLOR_LIGHT, lw=1, linestyle="-", zorder=0)

    ax.set_yticks(ys)
    ax.set_yticklabels([r.label for r in display_rows], fontsize=9)
    ax.set_xlabel("paired Δ (treatment − baseline, 95% CI)")
    ax.set_title(
        "F14 — Live Hugging Face rerun: paired Δ by bundle "
        "(preregistered gate: d_z ≥ 0.5 AND p ≤ 0.05)",
        fontsize=10,
    )

    # Side labels for bands.
    top_y = ys[0]
    bot_y = ys[-1]
    mid_core = np.mean(ys[:core_count]) if core_count else None
    mid_ext = np.mean(ys[core_count:]) if ext_count else None
    if mid_core is not None:
        ax.text(-0.30, mid_core, "v2.1\ncore4", transform=ax.get_yaxis_transform(),
                fontsize=8.5, ha="center", va="center", color=COLOR_AXIS,
                rotation=90, fontweight="bold")
    if mid_ext is not None:
        ax.text(-0.30, mid_ext, "v2.1.1\next", transform=ax.get_yaxis_transform(),
                fontsize=8.5, ha="center", va="center", color=COLOR_PURPLE,
                rotation=90, fontweight="bold")

    # Legend for gate status.
    legend_patches = [
        mpatches.Patch(color=COLOR_GREEN, label="preregistered gate met (d_z ≥ 0.5 and p ≤ 0.05)"),
        mpatches.Patch(color=COLOR_AMBER, label="directional, at least one gate shy"),
        mpatches.Patch(color=COLOR_GREY, label="null"),
        mpatches.Patch(facecolor="white", edgecolor=COLOR_HOLLOW_EDGE, label="un-exercised (no new rows)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.95)

    # Make room for right-side annotations (n, p, d_z).
    x_min = min(r.ci_low for r in display_rows) - 0.02
    x_max = max(r.ci_high for r in display_rows) + 0.42
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)

    fig.tight_layout()
    out = FIG / "F14_live_hf_forest.png"
    _save(fig, out)
    return out


def figure_F15() -> Path:
    """RULER 16K extension — pooled + per-model + per-cell composition panel."""
    summary = _load_json(P6A / "_summary_live.json")
    bundles_by_label = {b["label"]: b for b in summary.get("bundles", [])}
    ext = bundles_by_label["H1_ruler_16384_live_ext"]
    outcome = ext["outcome"]
    per_model = ext["per_model_outcome"]
    per_cell = ext["per_cell_counts"]

    fig = plt.figure(figsize=(8.6, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.35)
    ax_forest = fig.add_subplot(gs[0, 0])
    ax_cells = fig.add_subplot(gs[0, 1])

    # Left: pooled + per-model paired Δ with 95% bootstrap CI.
    haiku = per_model["claude-haiku-4-5"]
    sonnet = per_model["claude-sonnet-4-6"]
    entries = [
        ("pooled\n(n=103)", float(outcome["delta_observed"]),
         float(outcome["ci_low"]), float(outcome["ci_high"]),
         float(outcome["p_value"]), "pooled"),
        ("claude-haiku-4-5\n(n=60)", float(haiku["delta_observed"]),
         float(haiku["ci_low"]), float(haiku["ci_high"]),
         float(haiku["p_value"]), "model"),
        ("claude-sonnet-4-6\n(n=43, partial)", float(sonnet["delta_observed"]),
         float(sonnet["ci_low"]), float(sonnet["ci_high"]),
         float(sonnet["p_value"]), "model"),
    ]
    ys = np.arange(len(entries))[::-1]
    for y, (lab, d, lo, hi, p, kind) in zip(ys, entries):
        colour = COLOR_PURPLE if kind == "pooled" else COLOR_AMBER
        ax_forest.errorbar(
            [d], [y],
            xerr=[[d - lo], [hi - d]],
            fmt="o", color=colour, ecolor=colour,
            markerfacecolor=colour, markeredgecolor=colour,
            capsize=3, lw=1.6, markersize=7,
        )
        ax_forest.text(1.02, y, f"p={p:.3f}",
                       transform=ax_forest.get_yaxis_transform(),
                       fontsize=8, va="center", ha="left", color=COLOR_AXIS)
    ax_forest.axvline(0.0, color=COLOR_AXIS, lw=0.7)
    ax_forest.axvline(0.05, color=COLOR_AXIS, lw=0.6, linestyle=":",
                      label="Δ = +0.05 magnitude reference (gate is d_z + p)")
    ax_forest.set_yticks(ys)
    ax_forest.set_yticklabels([e[0] for e in entries], fontsize=9)
    ax_forest.set_xlabel("paired Δ (treatment − baseline, 95% CI)")
    ax_forest.set_title("F15 — RULER 16K (v2.1.1 ext): pooled + per-model")
    ax_forest.legend(loc="lower right", fontsize=8)
    x_min = min(e[2] for e in entries) - 0.02
    x_max = max(e[3] for e in entries) + 0.08
    ax_forest.set_xlim(x_min, x_max)
    ax_forest.set_ylim(min(ys) - 0.6, max(ys) + 0.6)

    # Right: per-cell counts grid. Rows = models × seeds, cols = harness_on / harness_off.
    # Expected N=30 per cell post-extension.
    expected = 30
    cell_rows: list[tuple[str, int, int]] = []  # (label, on, off)
    for model in ("claude-haiku-4-5", "claude-sonnet-4-6"):
        cells = per_cell.get(model, {})
        for seed in (0, 1):
            on = int(cells.get(f"seed{seed}_harness_on", 0))
            off = int(cells.get(f"seed{seed}_harness_off", 0))
            cell_rows.append((f"{model.replace('claude-', '')}  seed{seed}", on, off))

    nr = len(cell_rows)
    data = np.array([[r[1], r[2]] for r in cell_rows], dtype=float)
    # Shade: fully-extended cells (=30) = light green; partial cells (<30) = amber.
    shade = np.where(data >= expected, 0.0, 1.0)
    ax_cells.imshow(shade, cmap="Oranges", aspect="auto", alpha=0.55, vmin=0, vmax=1)
    ax_cells.set_xticks([0, 1])
    ax_cells.set_xticklabels(["harness_on", "harness_off"], fontsize=9)
    ax_cells.set_yticks(np.arange(nr))
    ax_cells.set_yticklabels([r[0] for r in cell_rows], fontsize=8)
    for i in range(nr):
        for j in range(2):
            v = int(data[i, j])
            if v >= expected:
                mark = f"{v} / {expected}"
                colour = "#065f46"
            else:
                mark = f"{v} / {expected}"
                colour = "#9a3412"
            ax_cells.text(j, i, mark, ha="center", va="center",
                          fontsize=9, color=colour, fontweight="bold")
    ax_cells.set_title("per-cell paired observations\n(green = fully extended, amber = partial)",
                       fontsize=9.5)
    for spine in ax_cells.spines.values():
        spine.set_edgecolor(COLOR_LIGHT)

    fig.suptitle(
        "F15 — RULER 16K power-extension composition (v2.1.1)",
        fontsize=11, y=1.02,
    )

    out = FIG / "F15_ruler16k_ext_per_model.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    rows = extract_rows()
    # Emit a tiny provenance dump alongside the figures so adversarial review
    # can diff numbers against the tables without re-reading the summary JSON.
    provenance = {
        "rows": [
            {
                "long_label": r.long_label,
                "band": r.band,
                "delta": r.delta,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "p_value": r.p_value,
                "cohens_d": r.cohens_d,
                "n_paired": r.n_paired,
                "gate_status": r.gate_status,
                "ci_source": r.ci_source,
            }
            for r in rows
        ]
    }
    (FIG / "F14_F15_provenance.json").write_text(json.dumps(provenance, indent=2))
    f14 = figure_F14(rows)
    f15 = figure_F15()
    print(f"wrote: {f14.relative_to(ROOT)}")
    print(f"wrote: {f15.relative_to(ROOT)}")
    print(f"wrote: paper/figures/F14_F15_provenance.json")


if __name__ == "__main__":
    main()
