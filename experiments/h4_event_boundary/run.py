import json
import os
import mlflow

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


def run_experiment() -> dict:
    from src.avacchedaka.element import ContextElement, AvacchedakaConditions
    from src.avacchedaka.store import ContextStore
    from src.compaction.detector import EventBoundaryDetector
    from src.compaction.compactor import BoundaryTriggeredSession, BoundaryTriggeredCompactor

    def make_store():
        store = ContextStore()
        for i in range(20):
            store.insert(ContextElement(
                id=f"e-{i:03d}", content=f"Step {i} observation.",
                precision=0.25 if i < 10 else 0.8,
                avacchedaka=AvacchedakaConditions(qualificand="task", qualifier="obs", condition="h4_test"),
            ))
        return store

    store_bt = make_store()
    detector = EventBoundaryDetector(surprise_threshold=0.75)
    session = BoundaryTriggeredSession(store_bt, detector, compress_threshold=0.3)
    surprises = [0.1] * 8 + [0.9] + [0.1] * 11
    bt_compressed = session.process_surprises(surprises, step=8)

    store_tt = make_store()
    compactor = BoundaryTriggeredCompactor(store_tt, compress_threshold=0.3)
    tt_compressed = compactor.threshold_compact(token_count=600, token_threshold=500)

    return {
        "boundary_triggered_compressed": len(bt_compressed),
        "threshold_triggered_compressed": len(tt_compressed),
        "compaction_events": session.compaction_events,
    }


if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-4-event-boundary")
    with mlflow.start_run():
        mlflow.log_params({"hypothesis": "H4", "seed": RANDOM_SEED})
        results = run_experiment()
        mlflow.log_metrics({
            "bt_compressed_count": results["boundary_triggered_compressed"],
            "tt_compressed_count": results["threshold_triggered_compressed"],
        })
        os.makedirs("data/experiments", exist_ok=True)
        with open("data/experiments/h4_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
