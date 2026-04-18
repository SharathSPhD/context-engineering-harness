"""H3 Experiment: Buddhi/Manas two-stage reasoning vs single-stage.

Hypothesis: Buddhi/manas two-stage outperforms single-stage.

Method:
- Two-stage: ManasAgent surfaces candidates (broad), BuddhiAgent commits (narrow).
  Manas provides uncertainty signal; Buddhi withholds when evidence is insufficient.
- Single-stage: BuddhiAgent runs directly against raw context without manas scaffolding.

Metrics: task_success_rate (correct answers + correctly withheld questions).
"""
import json
import os

import mlflow

from src.agents.buddhi import BuddhiAgent
from src.agents.orchestrator import ManusBuddhiOrchestrator
from src.avacchedaka.element import AvacchedakaConditions, ContextElement
from src.avacchedaka.query import AvacchedakaQuery
from src.avacchedaka.store import ContextStore

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

QA_TASKS = [
    {
        "question": "How long do JWT tokens remain valid?",
        "gold": "24 hours",
        "qualificand": "auth",
        "task_context": "task_type=qa",
        "docs": [("JWT tokens expire after 24 hours.", 0.9, "auth", "task_type=qa")],
    },
    {
        "question": "What is the millisecond-precise deployment timestamp of service X?",
        "gold": None,  # Should withhold — no grounding in context
        "qualificand": "deployment",
        "task_context": "task_type=qa",
        "docs": [],  # No relevant documents available
    },
    {
        "question": "What HTTP method does CORS preflight use?",
        "gold": "OPTIONS",
        "qualificand": "web_security",
        "task_context": "task_type=qa",
        "docs": [("CORS preflight requests use the OPTIONS HTTP method.", 0.95, "web_security", "task_type=qa")],
    },
]


def _build_store(task: dict) -> ContextStore:
    store = ContextStore()
    for content, precision, qualificand, condition in task["docs"]:
        store.insert(ContextElement(
            id=f"doc-{len(store._elements)}",
            content=content,
            precision=precision,
            avacchedaka=AvacchedakaConditions(
                qualificand=qualificand, qualifier="fact", condition=condition,
            ),
        ))
    return store


def _score(task: dict, answer: str | None, confidence: float) -> bool:
    """True if the answer matches gold, or if the model correctly withheld."""
    gold = task["gold"]
    if gold is None:
        return answer is None or confidence < 0.6
    return gold.lower() in (answer or "").lower()


def run_two_stage(task: dict, api_key: str) -> dict:
    """Manas → Buddhi pipeline (the proposed architecture)."""
    store = _build_store(task)
    orch = ManusBuddhiOrchestrator(api_key=api_key, store=store)
    output = orch.run(
        question=task["question"],
        task_context=task["task_context"],
        qualificand=task["qualificand"],
    )
    success = _score(task, output.answer, output.confidence)
    return {
        "question": task["question"],
        "success": success,
        "answer": output.answer,
        "confidence": output.confidence,
        "expected_withhold": task["gold"] is None,
        "stage": "two_stage",
    }


def run_single_stage(task: dict, api_key: str) -> dict:
    """BuddhiAgent only — no manas scaffolding (the baseline)."""
    store = _build_store(task)
    query = AvacchedakaQuery(qualificand=task["qualificand"], condition=task["task_context"])
    context_window = store.to_context_window(query, max_tokens=2048)

    buddhi = BuddhiAgent(api_key=api_key)
    output = buddhi.run(
        question=task["question"],
        context_window=context_window,
        manas_sketch="",  # No manas scaffolding
        uncertainty=0.5,  # No uncertainty signal — assume neutral
    )
    success = _score(task, output.answer, output.confidence)
    return {
        "question": task["question"],
        "success": success,
        "answer": output.answer,
        "confidence": output.confidence,
        "expected_withhold": task["gold"] is None,
        "stage": "single_stage",
    }


def run_experiment() -> dict:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    two_stage_results = [run_two_stage(t, api_key) for t in QA_TASKS]
    single_stage_results = [run_single_stage(t, api_key) for t in QA_TASKS]
    return {
        "two_stage": two_stage_results,
        "single_stage": single_stage_results,
    }


if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-3-buddhi-manas")
    with mlflow.start_run():
        mlflow.log_params({"hypothesis": "H3", "seed": RANDOM_SEED, "n_tasks": len(QA_TASKS)})
        results = run_experiment()

        two_stage_accuracy = sum(r["success"] for r in results["two_stage"]) / len(results["two_stage"])
        single_stage_accuracy = sum(r["success"] for r in results["single_stage"]) / len(results["single_stage"])

        # Log to MLflow BEFORE writing to disk (CLAUDE.md invariant)
        mlflow.log_metrics({
            "two_stage_task_success_rate": two_stage_accuracy,
            "single_stage_task_success_rate": single_stage_accuracy,
            "improvement": two_stage_accuracy - single_stage_accuracy,
        })

        os.makedirs("data/experiments", exist_ok=True)
        with open("data/experiments/h3_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
