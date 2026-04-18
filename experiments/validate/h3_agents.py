"""H3 Validation: Buddhi/manas two-stage outperforms single-stage.

Method:
- 3 QA tasks: grounded answer, ungrounded (should withhold), stable-doc question.
- Two-stage: ManusBuddhiOrchestrator (manas → buddhi with sakshi prefix).
- Single-stage: BuddhiAgent only, no manas scaffolding, uncertainty=0.5.
- Score: correct answers + correctly withheld = success.

Target: two_stage_accuracy >= single_stage_accuracy.
"""
from __future__ import annotations

import json

from experiments.validate.data import build_pre_shift_store, NEXUSAPI_DOCS, _make_element
from src.agents.buddhi import BuddhiAgent
from src.agents.orchestrator import ManusBuddhiOrchestrator
from src.avacchedaka.query import AvacchedakaQuery
from src.avacchedaka.store import ContextStore

_TASKS = [
    {
        "question": "How long are JWT tokens valid in NexusAPI?",
        "gold": "1 hour",
        "qualificand": "auth",
        "task_context": "task_type=code_review AND phase=post_shift",
        "expect_withhold": False,
    },
    {
        "question": "What is the exact millisecond timestamp of the last NexusAPI deployment?",
        "gold": None,  # no grounding — should withhold
        "qualificand": "deployment",
        "task_context": "task_type=code_review",
        "expect_withhold": True,
    },
]


def _build_store() -> ContextStore:
    store = build_pre_shift_store()
    store.insert(_make_element(NEXUSAPI_DOCS["auth_post"]))
    return store


def _score(task: dict, answer: str | None, confidence: float) -> bool:
    if task["gold"] is None:
        return answer is None or confidence < 0.6
    return task["gold"].lower()[:8] in (answer or "").lower()


def run_h3() -> dict:
    store = _build_store()
    orch = ManusBuddhiOrchestrator(store=store)
    buddhi = BuddhiAgent()

    two_stage_results = []
    single_stage_results = []

    for task in _TASKS:
        # Two-stage: manas → buddhi with sakshi prefix
        out = orch.run(
            question=task["question"],
            task_context=task["task_context"],
            qualificand=task["qualificand"],
        )
        two_stage_results.append({
            "question": task["question"],
            "success": _score(task, out.answer, out.confidence),
            "answer": out.answer,
            "confidence": out.confidence,
        })

        # Single-stage: buddhi alone, no manas scaffolding, uncertainty=0.5
        query = AvacchedakaQuery(qualificand=task["qualificand"], condition=task["task_context"])
        context_window = store.to_context_window(query, max_tokens=2048)
        out2 = buddhi.run(
            question=task["question"],
            context_window=context_window,
            manas_sketch="",
            uncertainty=0.5,
        )
        single_stage_results.append({
            "question": task["question"],
            "success": _score(task, out2.answer, out2.confidence),
            "answer": out2.answer,
            "confidence": out2.confidence,
        })

    n = len(_TASKS)
    two_acc = sum(r["success"] for r in two_stage_results) / n
    single_acc = sum(r["success"] for r in single_stage_results) / n

    return {
        "hypothesis": "H3",
        "description": "Buddhi/manas two-stage outperforms single-stage",
        "two_stage_accuracy": round(two_acc, 3),
        "single_stage_accuracy": round(single_acc, 3),
        "improvement": round(two_acc - single_acc, 3),
        "target_met": two_acc >= single_acc,
        "target_description": "two_stage_accuracy >= single_stage_accuracy",
        "details": {"two_stage": two_stage_results, "single_stage": single_stage_results},
    }


if __name__ == "__main__":
    result = run_h3()
    print(json.dumps(result, indent=2))
