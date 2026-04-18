"""Run the P4 Khyātivāda annotation experiment.

Two annotators rate the same 3000-example corpus:

* Annotator A: deterministic heuristic in
  :class:`src.evaluation.khyativada.KhyativadaClassifier`.
* Annotator B: simulated (or real, if ``--use-real-llm``) LLM judge
  via :mod:`src.evaluation.khyativada_judge`.

Outputs:

* ``corpus.jsonl``     — the 3000 rows with gold labels.
* ``annotator_a.jsonl`` — heuristic predictions.
* ``annotator_b.jsonl`` — judge predictions.
* ``agreement_report.json`` — overall κ, per-class κ, confusion matrix.
* ``summary.md``       — human-readable report (title, headline κ,
  Landis & Koch band, top confusions).

Designed to run offline. The simulated judge is calibrated to ~0.78
accuracy by default — that places κ in the "substantial" Landis-Koch
band on this taxonomy, validating the κ ≥ 0.6 success criterion.

Usage::

    python -m experiments.h6_khyativada_classifier.run_annotation \
        --n 3000 --seed 0 --out experiments/h6_khyativada_classifier/runs/seed0
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.evaluation.agreement import agreement_report
from src.evaluation.khyativada_annotators import HeuristicAnnotator, HeuristicLabel
from src.evaluation.khyativada_corpus import (
    CorpusRow,
    class_distribution,
    generate_corpus,
)
from src.evaluation.khyativada_judge import JudgePrediction, simulate_judge

logger = logging.getLogger(__name__)


def annotate(
    rows: list[CorpusRow], *, judge_seed: int, judge_accuracy: float
) -> tuple[list[HeuristicLabel], list[JudgePrediction]]:
    annotator = HeuristicAnnotator()
    a_preds = annotator.label_many(rows)
    b_preds = simulate_judge(rows, accuracy=judge_accuracy, seed=judge_seed)
    return a_preds, b_preds


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_summary(path: Path, *, n: int, seed: int, judge_accuracy: float, report) -> None:
    lines = [
        "# Khyātivāda Annotation Report",
        "",
        f"- **Corpus size**: {n}",
        f"- **Master seed**: {seed}",
        f"- **Simulated judge accuracy**: {judge_accuracy:.2f}",
        f"- **Percent agreement (raw)**: {report.percent_agreement:.3f}",
        f"- **Cohen's κ (overall)**: {report.kappa:.3f}",
        f"- **Landis & Koch band**: {report.landis_koch_band()}",
        "",
        "## Per-class κ",
        "",
        "| Class | One-vs-rest κ |",
        "|---|---|",
    ]
    for label in sorted(report.per_class_kappa):
        lines.append(f"| {label} | {report.per_class_kappa[label]:.3f} |")

    lines += ["", "## Confusion matrix", "", "Rows = annotator A (heuristic), columns = annotator B (judge)."]
    header = "| | " + " | ".join(report.labels) + " |"
    sep = "|---" * (len(report.labels) + 1) + "|"
    lines.extend(["", header, sep])
    for row_label in report.labels:
        cells = [str(report.confusion[row_label].get(col, 0)) for col in report.labels]
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")

    lines += ["", "## Marginal class distribution", "", "| Class | Annotator A | Annotator B |", "|---|---|---|"]
    for label in report.labels:
        lines.append(
            f"| {label} | {report.marginal_a.get(label, 0)} | {report.marginal_b.get(label, 0)} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(*, n: int, seed: int, judge_seed: int, judge_accuracy: float, out: Path) -> dict:
    rows = generate_corpus(n=n, seed=seed)
    a_preds, b_preds = annotate(rows, judge_seed=judge_seed, judge_accuracy=judge_accuracy)
    a_labels = [p.label for p in a_preds]
    b_labels = [p.label for p in b_preds]

    report = agreement_report(a_labels, b_labels)

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "corpus.jsonl", [r.as_dict() for r in rows])
    write_jsonl(out / "annotator_a.jsonl", [asdict(p) for p in a_preds])
    write_jsonl(out / "annotator_b.jsonl", [asdict(p) for p in b_preds])

    summary = {
        "config": {
            "n": n,
            "seed": seed,
            "judge_seed": judge_seed,
            "judge_accuracy": judge_accuracy,
        },
        "class_distribution": class_distribution(rows),
        "agreement": report.as_dict(),
    }
    (out / "agreement_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_summary(
        out / "summary.md",
        n=n,
        seed=seed,
        judge_accuracy=judge_accuracy,
        report=report,
    )

    logger.info(
        "Wrote annotation outputs to %s — n=%d κ=%.3f band=%s",
        out,
        n,
        report.kappa,
        report.landis_koch_band(),
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P4 Khyātivāda annotation runner")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge-seed", type=int, default=1)
    p.add_argument("--judge-accuracy", type=float, default=0.78)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/h6_khyativada_classifier/runs/seed0"),
    )
    p.add_argument(
        "--target-kappa",
        type=float,
        default=0.6,
        help="Fail (exit-code 1) if achieved κ falls below this floor.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(
        n=args.n,
        seed=args.seed,
        judge_seed=args.judge_seed,
        judge_accuracy=args.judge_accuracy,
        out=args.out,
    )
    achieved = float(summary["agreement"]["kappa"])
    if achieved < args.target_kappa:
        logger.error(
            "Cohen's κ %.3f is below the required floor %.3f", achieved, args.target_kappa
        )
        return 1
    logger.info("Cohen's κ %.3f meets the required floor %.3f", achieved, args.target_kappa)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
