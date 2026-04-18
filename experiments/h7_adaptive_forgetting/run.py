import json
import os
import mlflow

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


def run_experiment() -> dict:
    from src.forgetting.distribution_shift import DistributionShiftBenchmark
    from src.forgetting.schedules import (
        NoForgetting, FixedCompaction, RecencyWeightedForgetting,
        RewardWeightedForgetting, BadhaFirstForgetting,
    )
    from src.avacchedaka.query import AvacchedakaQuery

    bench = DistributionShiftBenchmark()
    task = bench.build_jwt_shift()

    def accuracy_after_shift(schedule_cls, **kwargs) -> dict:
        store = bench.build_store_pre_shift(task)
        schedule = schedule_cls(store, **kwargs) if kwargs else schedule_cls(store)
        bench.apply_shift(store, task)
        schedule.apply()
        query = AvacchedakaQuery(qualificand="auth", condition="phase=post_shift")
        results = store.retrieve(query)
        answered = results[0].content if results else ""
        correct = task.post_shift_answer in answered
        return {"correct": correct, "n_results": len(results), "top_answer": answered[:60]}

    return {
        "no_forgetting": accuracy_after_shift(NoForgetting),
        "fixed_compaction": accuracy_after_shift(FixedCompaction, keep_newest=2),
        "recency_weighted": accuracy_after_shift(RecencyWeightedForgetting, decay_factor=0.9),
        "reward_weighted": accuracy_after_shift(RewardWeightedForgetting, keep_threshold=0.3),
        "badha_first": accuracy_after_shift(BadhaFirstForgetting),
    }


if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-7-adaptive-forgetting")
    with mlflow.start_run():
        mlflow.log_params({"hypothesis": "H7", "seed": RANDOM_SEED})
        results = run_experiment()
        mlflow.log_metrics({
            "no_forgetting_correct": int(results["no_forgetting"]["correct"]),
            "fixed_compaction_correct": int(results["fixed_compaction"]["correct"]),
            "recency_weighted_correct": int(results["recency_weighted"]["correct"]),
            "reward_weighted_correct": int(results["reward_weighted"]["correct"]),
            "badha_first_correct": int(results["badha_first"]["correct"]),
        })
        os.makedirs("data/experiments", exist_ok=True)
        with open("data/experiments/h7_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
