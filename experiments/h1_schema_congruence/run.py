import json
import os

import mlflow


RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
MODEL = "claude-haiku-4-5"


def run_experiment() -> list[dict]:
    import anthropic
    from src.evaluation.schema_congruence import CongruenceBenchmarkBuilder
    from src.evaluation.metrics import congruence_ratio

    client = anthropic.Anthropic()
    builder = CongruenceBenchmarkBuilder(seed=RANDOM_SEED)
    gold = "JWT tokens expire after 24 hours."
    results = []
    for version in ("congruent", "incongruent"):
        example = builder.build_example(
            gold_passage=gold,
            domain="web_security",
            target_length_k=4,
            version=version,
        )
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"{example.context}\n\nQuestion: {example.question}\nAnswer concisely:",
            }],
        )
        answer = response.content[0].text.strip()
        correct = gold.lower()[:30] in answer.lower()
        results.append({
            "version": version,
            "correct": correct,
            "congruence_ratio": congruence_ratio(example),
            "answer": answer,
        })
    return results


if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-1-schema-congruence")
    with mlflow.start_run():
        mlflow.log_params({"model": MODEL, "hypothesis": "H1", "seed": RANDOM_SEED})
        results = run_experiment()
        mlflow.log_metrics({
            "congruent_accuracy": int(next(r["correct"] for r in results if r["version"] == "congruent")),
            "incongruent_accuracy": int(next(r["correct"] for r in results if r["version"] == "incongruent")),
        })
        os.makedirs("data/experiments", exist_ok=True)
        with open("data/experiments/h1_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
