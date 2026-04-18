"""H5 Experiment: Avacchedaka multi-agent coordination.

Hypothesis: Avacchedaka annotation reduces multi-agent conflict rate ≥30%.

Method: Simulate two agents writing conflicting claims about the same qualificand.
Compare conflict rates (duplicate answers retrieved) with and without sublation.
"""
import json
import os

import mlflow

from src.avacchedaka.element import AvacchedakaConditions, ContextElement
from src.avacchedaka.query import AvacchedakaQuery
from src.avacchedaka.store import ContextStore

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

COORDINATION_TASKS = [
    {
        "task": "Deploy database version",
        "agent1_claim": {"id": "t1-a1", "content": "PostgreSQL 14", "precision": 0.8},
        "agent2_claim": {"id": "t1-a2", "content": "PostgreSQL 16 (correct)", "precision": 0.95},
        "qualificand": "database",
        "condition": "task_type=deploy",
        "correct": "PostgreSQL 16",
    },
    {
        "task": "Auth token expiry",
        "agent1_claim": {"id": "t2-a1", "content": "24 hours", "precision": 0.7},
        "agent2_claim": {"id": "t2-a2", "content": "1 hour (updated policy)", "precision": 0.9},
        "qualificand": "auth",
        "condition": "task_type=qa",
        "correct": "1 hour",
    },
    {
        "task": "API rate limit",
        "agent1_claim": {"id": "t3-a1", "content": "100 requests/min", "precision": 0.75},
        "agent2_claim": {"id": "t3-a2", "content": "50 requests/min (throttled after incident)", "precision": 0.92},
        "qualificand": "api",
        "condition": "task_type=ops",
        "correct": "50 requests/min",
    },
]


def _make_conds(task: dict) -> AvacchedakaConditions:
    return AvacchedakaConditions(
        qualificand=task["qualificand"],
        qualifier="version",
        condition=task["condition"],
    )


def run_with_avacchedaka(task: dict) -> dict:
    """Both agents share a store; Agent 2 sublates Agent 1's claim."""
    store = ContextStore()
    conds = _make_conds(task)
    store.insert(ContextElement(
        id=task["agent1_claim"]["id"],
        content=task["agent1_claim"]["content"],
        precision=task["agent1_claim"]["precision"],
        avacchedaka=conds,
        provenance="agent1",
    ))
    store.insert(ContextElement(
        id=task["agent2_claim"]["id"],
        content=task["agent2_claim"]["content"],
        precision=task["agent2_claim"]["precision"],
        avacchedaka=conds,
        provenance="agent2",
    ))
    store.sublate(task["agent1_claim"]["id"], task["agent2_claim"]["id"])
    results = store.retrieve(AvacchedakaQuery(qualificand=task["qualificand"], condition=task["condition"]))
    conflict = len(results) > 1
    correct = task["correct"].lower() in (results[0].content.lower() if results else "")
    return {"task": task["task"], "conflict": conflict, "correct": correct, "n_results": len(results)}


def run_without_avacchedaka(task: dict) -> dict:
    """Both agents share a store; no sublation — both claims remain."""
    store = ContextStore()
    conds = _make_conds(task)
    store.insert(ContextElement(
        id=task["agent1_claim"]["id"],
        content=task["agent1_claim"]["content"],
        precision=task["agent1_claim"]["precision"],
        avacchedaka=conds,
        provenance="agent1",
    ))
    store.insert(ContextElement(
        id=task["agent2_claim"]["id"],
        content=task["agent2_claim"]["content"],
        precision=task["agent2_claim"]["precision"],
        avacchedaka=conds,
        provenance="agent2",
    ))
    # No sublation — both claims visible
    results = store.retrieve(AvacchedakaQuery(qualificand=task["qualificand"], condition=task["condition"]))
    conflict = len(results) > 1
    correct = task["correct"].lower() in (results[0].content.lower() if results else "")
    return {"task": task["task"], "conflict": conflict, "correct": correct, "n_results": len(results)}


if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-5-avacchedaka-multiagent")
    with mlflow.start_run():
        mlflow.log_params({"hypothesis": "H5", "seed": RANDOM_SEED, "n_tasks": len(COORDINATION_TASKS)})

        with_results = [run_with_avacchedaka(t) for t in COORDINATION_TASKS]
        without_results = [run_without_avacchedaka(t) for t in COORDINATION_TASKS]

        with_conflict_rate = sum(r["conflict"] for r in with_results) / len(with_results)
        without_conflict_rate = sum(r["conflict"] for r in without_results) / len(without_results)
        reduction = (without_conflict_rate - with_conflict_rate) / max(without_conflict_rate, 0.001)

        with_accuracy = sum(r["correct"] for r in with_results) / len(with_results)
        without_accuracy = sum(r["correct"] for r in without_results) / len(without_results)

        # Log to MLflow BEFORE writing to disk (CLAUDE.md invariant)
        mlflow.log_metrics({
            "with_avacchedaka_conflict_rate": with_conflict_rate,
            "without_avacchedaka_conflict_rate": without_conflict_rate,
            "conflict_rate_reduction_pct": reduction * 100,
            "with_avacchedaka_accuracy": with_accuracy,
            "without_avacchedaka_accuracy": without_accuracy,
        })

        summary = {
            "with_avacchedaka": with_results,
            "without_avacchedaka": without_results,
            "conflict_rate_reduction_pct": reduction * 100,
            "h5_target_met": reduction >= 0.30,
        }
        os.makedirs("data/experiments", exist_ok=True)
        with open("data/experiments/h5_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
