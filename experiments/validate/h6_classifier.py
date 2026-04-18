"""H6 Validation: Khyātivāda classifier accurately identifies hallucination types.

Method (no LLM — uses heuristic classifier):
- 9 annotated examples covering anyathakhyati, asatkhyati, akhyati error classes.
- Run classify_heuristic() on each.
- Measure accuracy (correct class / total).

Target: accuracy ≥ 60%.
"""
from __future__ import annotations

import json

from src.evaluation.khyativada import KhyativadaClassifier

_ANNOTATED = [
    # asatkhyati: ground truth indicates the referenced entity does not exist
    ("Use requests.get_json() to parse the response",
     "requests.get_json() does not exist; use response.json()", "asatkhyati"),
    ("Call nexus.refresh_token_async() to renew auth",
     "nexus.refresh_token_async() does not exist in the SDK", "asatkhyati"),
    ("Use the @nexus.cache_forever decorator for performance",
     "@nexus.cache_forever does not exist in the NexusAPI framework", "asatkhyati"),
    # akhyati: true components combined into a false relation
    ("Einstein won the Nobel Prize in 1921 for his theory of relativity",
     "Einstein won the Nobel Prize in 1921 but not for relativity", "akhyati"),
    ("NexusAPI v2 was released in 2025 for database improvements",
     "NexusAPI v2 was released in 2025 but not for database improvements", "akhyati"),
    ("The outage was caused by memory leaks",
     "The outage occurred but not due to memory leaks; it was a network partition", "akhyati"),
    # anyathakhyati: version/identifier mismatch — real entity misidentified
    ("Python GIL was removed in version 3.10",
     "Python GIL was removed in version 3.13", "anyathakhyati"),
    ("NexusAPI uses PostgreSQL 14.0",
     "NexusAPI uses PostgreSQL 16.0 after the 2026-02 migration", "anyathakhyati"),
    ("Redis 6.0 is used for caching",
     "Redis 7.0 is used for caching", "anyathakhyati"),
]


def run_h6() -> dict:
    clf = KhyativadaClassifier()
    correct = 0
    details = []

    for claim, ground_truth, expected_class in _ANNOTATED:
        result = clf.classify_heuristic(claim=claim, ground_truth=ground_truth)
        is_correct = result["class"] == expected_class
        if is_correct:
            correct += 1
        details.append({
            "claim": claim,
            "expected": expected_class,
            "predicted": result["class"],
            "confidence": result["confidence"],
            "correct": is_correct,
        })

    n = len(_ANNOTATED)
    accuracy = correct / n

    return {
        "hypothesis": "H6",
        "description": "Khyātivāda classifier accurately identifies hallucination error types",
        "accuracy": round(accuracy, 3),
        "n_correct": correct,
        "n_total": n,
        "target_met": accuracy >= 0.60,
        "target_description": "heuristic accuracy ≥ 60% on annotated examples",
        "details": details,
    }


if __name__ == "__main__":
    print(json.dumps(run_h6(), indent=2))
