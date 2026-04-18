"""H1 Validation: Schema-congruence predicts context rot better than length.

Method:
- Build congruent contexts (web_security domain) and incongruent contexts
  (unrelated distractors) of the same length.
- Ask 3 NexusAPI questions using each context type.
- Measure: congruent accuracy vs incongruent accuracy.
- The delta proves congruence is the signal, not length.

Target: congruent_accuracy > incongruent_accuracy. (Higher accuracy on
schema-congruent contexts means congruence — not length — predicts whether
the model finds the answer. Direction was inverted in v0; fixed in v2 G3.)
"""
from __future__ import annotations

import json

from experiments.validate.data import INCONGRUENT_DISTRACTORS
from src.cli_bridge import ClaudeCLIClient
from src.config import config
from src.evaluation.metrics import congruence_ratio
from src.evaluation.schema_congruence import CongruenceBenchmarkBuilder

_EVAL_QUESTIONS = [
    (
        "JWT tokens expire after 24 hours. Passwords must be minimum 12 characters.",
        "Based only on the context provided, what is the minimum password length?",
        "12",
    ),
    (
        "CORS preflight uses OPTIONS method. Allowed origins are configured in settings.py.",
        "Based only on the context provided, what HTTP method does CORS preflight use?",
        "OPTIONS",
    ),
    (
        "CSRF tokens are single-use random values bound to the session cookie.",
        "Based only on the context provided, what are CSRF tokens bound to?",
        "session",
    ),
]


def _ask(client: ClaudeCLIClient, context: str, question: str) -> str:
    resp = client.messages.create(
        model=config.fast_model,
        max_tokens=config.fast_max_tokens,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nAnswer in one sentence: {question}"}],
    )
    return resp.content[0].text.strip().lower()


def _congruence_ratios() -> dict[str, float]:
    """Algorithmic check: congruent contexts have higher congruence_ratio."""
    builder = CongruenceBenchmarkBuilder(seed=42)
    gold = "JWT tokens expire after 24 hours."
    cong = builder.build_example(gold, "web_security", 4, "congruent")
    incong = builder.build_example(gold, "web_security", 4, "incongruent")
    return {
        "congruent": congruence_ratio(cong),
        "incongruent": congruence_ratio(incong),
    }


def run_h1() -> dict:
    client = ClaudeCLIClient()
    builder = CongruenceBenchmarkBuilder(seed=42)
    results: dict[str, list] = {"congruent": [], "incongruent": []}

    for gold_passage, question, expected_keyword in _EVAL_QUESTIONS:
        for version in ("congruent", "incongruent"):
            example = builder.build_example(gold_passage, "web_security", 4, version)
            answer = _ask(client, example.context, question)
            correct = expected_keyword in answer
            results[version].append({
                "question": question,
                "answer": answer,
                "correct": correct,
                "congruence_ratio": congruence_ratio(example),
            })

    n = len(_EVAL_QUESTIONS)
    cong_acc = sum(r["correct"] for r in results["congruent"]) / n
    incong_acc = sum(r["correct"] for r in results["incongruent"]) / n
    delta = cong_acc - incong_acc

    return {
        "hypothesis": "H1",
        "description": "Schema-congruence predicts context rot better than length",
        "congruent_accuracy": round(cong_acc, 3),
        "incongruent_accuracy": round(incong_acc, 3),
        "delta": round(delta, 3),
        "target_met": delta > 0,
        "target_description": "congruent_accuracy > incongruent_accuracy",
        "details": results,
    }


if __name__ == "__main__":
    result = run_h1()
    print(json.dumps(result, indent=2))
