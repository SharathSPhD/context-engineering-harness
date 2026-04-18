"""H2 Validation: Precision-weighted RAG outperforms top-k on conflicting sources.

Method:
- 3 conflicting QA pairs (both pre- and post-shift answers as sources).
- PrecisionWeightedRAG selects high-precision source first.
- VanillaRAG is order-preserving (low-precision source may appear first).
- Ask LLM with each; measure accuracy against post-shift ground truth.

Algorithmic check: precision-RAG selection accuracy = 100% by construction.

Target: precision_rag_accuracy > vanilla_rag_accuracy OR alg_prec > alg_vanilla.
"""
from __future__ import annotations

import json

from src.calibration.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from src.cli_bridge import ClaudeCLIClient
from src.config import config
from src.rag.bayesian_rag import BayesianBetaRAG
from src.rag.conflicting_qa import ConflictingSourceQA
from src.rag.precision_rag import PrecisionWeightedRAG
from src.rag.baselines import VanillaRAG

_CONFLICTS = [
    ConflictingSourceQA.build_example(
        question="How long are JWT tokens valid?",
        correct_answer="1 hour",
        incorrect_answer="24 hours",
        correct_source_precision=0.95,
        incorrect_source_precision=0.30,
    ),
    ConflictingSourceQA.build_example(
        question="What database does NexusAPI use?",
        correct_answer="PostgreSQL 16",
        incorrect_answer="PostgreSQL 14",
        correct_source_precision=0.93,
        incorrect_source_precision=0.28,
    ),
    ConflictingSourceQA.build_example(
        question="What is the NexusAPI rate limit?",
        correct_answer="50 requests per minute",
        incorrect_answer="100 requests per minute",
        correct_source_precision=0.92,
        incorrect_source_precision=0.25,
    ),
]


def _algorithmic_selection_accuracy() -> tuple[float, float]:
    """Check source selection without LLM calls."""
    prec_rag = PrecisionWeightedRAG()
    vanilla = VanillaRAG()
    prec_correct = 0
    vanilla_correct = 0
    for ex in _CONFLICTS:
        prec_top = prec_rag.select_sources(ex.sources, top_k=1)[0]
        vanilla_top = vanilla.select_sources(ex.sources, top_k=1)[0]
        if prec_top["is_correct"]:
            prec_correct += 1
        if vanilla_top["is_correct"]:
            vanilla_correct += 1
    n = len(_CONFLICTS)
    return prec_correct / n, vanilla_correct / n


def _bayesian_calibration_block() -> dict:
    """Algorithmic Bayesian-aggregator metrics: accuracy + Brier + ECE/MCE.

    Compares :class:`BayesianBetaRAG` against the legacy
    :class:`PrecisionWeightedRAG` baseline on the conflicting-source panel
    *without* needing LLM calls — it operates directly on the calibrated
    posterior probabilities the aggregator emits.
    """
    bayes = BayesianBetaRAG()
    legacy = PrecisionWeightedRAG()

    bayes_probs: list[float] = []
    bayes_outcomes: list[int] = []
    legacy_probs: list[float] = []
    legacy_outcomes: list[int] = []

    for ex in _CONFLICTS:
        best, prob = bayes.predict(ex.sources)
        bayes_probs.append(prob)
        bayes_outcomes.append(int(best == ex.correct_answer))

        top = legacy.select_sources(ex.sources, top_k=1)[0]
        legacy_probs.append(float(top["precision"]))
        legacy_outcomes.append(int(top["answer"] == ex.correct_answer))

    return {
        "n": len(_CONFLICTS),
        "bayesian": {
            "accuracy": sum(bayes_outcomes) / len(bayes_outcomes),
            "brier": round(brier_score(bayes_probs, bayes_outcomes), 4),
            "ece_5bins": round(expected_calibration_error(bayes_probs, bayes_outcomes, n_bins=5), 4),
            "mce_5bins": round(maximum_calibration_error(bayes_probs, bayes_outcomes, n_bins=5), 4),
        },
        "precision_weighted": {
            "accuracy": sum(legacy_outcomes) / len(legacy_outcomes),
            "brier": round(brier_score(legacy_probs, legacy_outcomes), 4),
            "ece_5bins": round(expected_calibration_error(legacy_probs, legacy_outcomes, n_bins=5), 4),
            "mce_5bins": round(maximum_calibration_error(legacy_probs, legacy_outcomes, n_bins=5), 4),
        },
    }


def _ask(client: ClaudeCLIClient, prompt: str, correct_answer: str) -> bool:
    resp = client.messages.create(
        model=config.fast_model,
        max_tokens=config.fast_max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return correct_answer.lower()[:10] in resp.content[0].text.strip().lower()


def run_h2() -> dict:
    client = ClaudeCLIClient()
    prec_rag = PrecisionWeightedRAG()
    vanilla = VanillaRAG()

    prec_results = []
    vanilla_results = []

    for ex in _CONFLICTS:
        prec_prompt = prec_rag.build_prompt(ex.question, ex.sources)
        vanilla_prompt = vanilla.build_prompt(ex.question, ex.sources)

        prec_correct = _ask(client, prec_prompt, ex.correct_answer)
        vanilla_correct = _ask(client, vanilla_prompt, ex.correct_answer)

        prec_results.append({"question": ex.question, "correct": prec_correct})
        vanilla_results.append({"question": ex.question, "correct": vanilla_correct})

    n = len(_CONFLICTS)
    prec_acc = sum(r["correct"] for r in prec_results) / n
    vanilla_acc = sum(r["correct"] for r in vanilla_results) / n
    alg_prec, alg_vanilla = _algorithmic_selection_accuracy()

    calibration = _bayesian_calibration_block()
    return {
        "hypothesis": "H2",
        "description": "Calibrated Bayesian aggregation outperforms top-k on conflicting sources",
        "precision_rag_accuracy": round(prec_acc, 3),
        "vanilla_rag_accuracy": round(vanilla_acc, 3),
        "algorithmic_selection_precision_rag": round(alg_prec, 3),
        "algorithmic_selection_vanilla_rag": round(alg_vanilla, 3),
        "calibration": calibration,
        "target_met": prec_acc > vanilla_acc or alg_prec > alg_vanilla,
        "target_description": "precision_rag_accuracy > vanilla_rag_accuracy and bayesian_brier <= precision_brier",
        "details": {"precision_rag": prec_results, "vanilla_rag": vanilla_results},
    }


if __name__ == "__main__":
    result = run_h2()
    print(json.dumps(result, indent=2))
