"""Generate docs/validation_report.md from validation results JSON."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


_TARGETS = {
    "H1": "congruent_accuracy < incongruent_accuracy",
    "H2": "precision_rag_accuracy > vanilla_rag_accuracy",
    "H3": "two_stage_accuracy >= single_stage_accuracy",
    "H4": "boundary_retention >= threshold_retention",
    "H5": "conflict_rate_reduction ≥ 30%",
    "H6": "heuristic_accuracy ≥ 60%",
    "H7": "bādha-first LLM accuracy ≥ no-forgetting on post-shift",
}

_DESCRIPTIONS = {
    "H1": "Schema-congruence predicts context rot better than length",
    "H2": "Precision-weighted RAG outperforms top-k on conflicting sources",
    "H3": "Buddhi/manas two-stage outperforms single-stage reasoning",
    "H4": "Event-boundary compaction outperforms threshold compaction",
    "H5": "Avacchedaka annotation reduces multi-agent conflict rate ≥30%",
    "H6": "Khyātivāda classifier accurately identifies hallucination types",
    "H7": "Adaptive forgetting outperforms fixed schedule on post-shift tasks",
}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.1%}" if v <= 1.0 else str(v)
    return str(v)


def _key_metric(h: str, r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error'][:60]}"
    if r.get("skipped"):
        return "SKIPPED"
    try:
        m = {
            "H1": lambda: (f"congruent={r['congruent_accuracy']:.0%}, "
                           f"incongruent={r['incongruent_accuracy']:.0%}, "
                           f"delta={r['delta']:.0%}"),
            "H2": lambda: (f"precision_rag={r['precision_rag_accuracy']:.0%}, "
                           f"vanilla={r['vanilla_rag_accuracy']:.0%}, "
                           f"alg_precision={r['algorithmic_selection_precision_rag']:.0%}"),
            "H3": lambda: (f"two_stage={r['two_stage_accuracy']:.0%}, "
                           f"single={r['single_stage_accuracy']:.0%}, "
                           f"improvement={r['improvement']:+.0%}"),
            "H4": lambda: (f"boundary_ret={r['boundary_retention']:.0%}, "
                           f"threshold_ret={r['threshold_retention']:.0%}"),
            "H5": lambda: (f"with={r['with_avacchedaka_conflict_rate']:.0%}, "
                           f"without={r['without_avacchedaka_conflict_rate']:.0%}, "
                           f"reduction={r['reduction_pct']:.0f}%"),
            "H6": lambda: (f"accuracy={r['accuracy']:.0%} "
                           f"({r['n_correct']}/{r['n_total']})"),
            "H7": lambda: (f"bādha_llm={r['badha_first_llm_correct']}, "
                           f"no_forget_llm={r['no_forgetting_llm_correct']}, "
                           f"bādha_retention={r['badha_first_retention']}"),
        }
        return m[h]()
    except Exception as exc:
        return f"(parse error: {exc})"


def generate_report(
    results_path: str = "data/experiments/validation_results.json",
    output_path: str = "docs/validation_report.md",
) -> None:
    with open(results_path) as f:
        data = json.load(f)

    generated_at = data.get("generated_at", datetime.utcnow().isoformat())
    results = data.get("results", {})

    n_pass = sum(1 for r in results.values() if r.get("target_met") is True)
    n_fail = sum(1 for r in results.values() if r.get("target_met") is False)
    n_skip = sum(1 for r in results.values() if r.get("skipped"))

    lines = [
        "# Context Engineering Harness — Validation Report",
        "",
        f"**Generated:** {generated_at[:19].replace('T', ' ')} UTC  ",
        "**Models:** claude-haiku-4-5 (fast) / claude-sonnet-4-6 (smart) via Claude Code CLI  ",
        "**Use case:** NexusAPI Code Review Assistant (synthetic corpus)  ",
        f"**Summary:** {n_pass} PASS · {n_fail} FAIL · {n_skip} SKIP  ",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| ID | Description | Key Metrics | Target | Result |",
        "|---|---|---|---|---|",
    ]

    for h in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        r = results.get(h, {})
        target_met = r.get("target_met")
        if r.get("skipped"):
            status = "⏭ SKIP"
        elif "error" in r:
            status = "💥 ERROR"
        elif target_met:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        lines.append(
            f"| **{h}** | {_DESCRIPTIONS.get(h, '')} | {_key_metric(h, r)} | "
            f"{_TARGETS.get(h, '')} | {status} |"
        )

    lines += ["", "---", "", "## Detailed Results", ""]

    for h in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        r = results.get(h, {})
        lines += [f"### {h}: {_DESCRIPTIONS.get(h, '')}", ""]
        if r.get("skipped"):
            lines += ["*Skipped (--skip-llm mode)*", ""]
            continue
        if "error" in r:
            lines += [f"**Error:** `{r['error']}`", ""]
            continue
        elapsed = r.get("elapsed_s", "?")
        passed = "✅ PASS" if r.get("target_met") else "❌ FAIL"
        lines += [
            f"**Elapsed:** {elapsed}s  ",
            f"**Target:** {_TARGETS.get(h, '')}  ",
            f"**Result:** {passed}  ",
            "",
        ]
        for k, v in r.items():
            if k in ("hypothesis", "description", "target_description",
                      "details", "target_met", "elapsed_s", "note"):
                continue
            lines.append(f"- **{k}:** {v}")
        if "note" in r:
            lines += ["", f"> {r['note']}"]
        if "target_description" in r:
            lines += ["", f"**Target definition:** {r['target_description']}"]
        lines.append("")

    lines += [
        "---",
        "",
        "## Methodology",
        "",
        "All LLM calls route through `claude` CLI subprocess (Claude Code subscription auth).  ",
        "No `ANTHROPIC_API_KEY` is required for the validation suite.",
        "",
        "**Synthetic data:** NexusAPI is a fictional web service corpus with three controlled  ",
        "distribution shifts: JWT expiry 24h→1h, PostgreSQL 14→16, rate limit 100→50 req/min.",
        "",
        "| Hypothesis | LLM calls | Model |",
        "|---|---|---|",
        "| H1 | 6 (3 congruent + 3 incongruent) | claude-haiku-4-5 |",
        "| H2 | 6 (3 precision-RAG + 3 vanilla) | claude-haiku-4-5 |",
        "| H3 | 4–6 (two-stage + single-stage) | haiku-4-5 + sonnet-4-6 |",
        "| H4 | 0 (algorithmic) | — |",
        "| H5 | 0 (algorithmic) | — |",
        "| H6 | 0 (heuristic classifier) | — |",
        "| H7 | 3 (one per schedule) | claude-haiku-4-5 |",
        "",
        "_Generated by `make validate && make report`_",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "data/experiments/validation_results.json"
    generate_report(results_path=results_path)
