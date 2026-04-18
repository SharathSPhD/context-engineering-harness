import json
import os

import mlflow

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


def run_experiment() -> list[dict]:
    import anthropic
    from src.rag.conflicting_qa import ConflictingSourceQA
    from src.rag.precision_rag import PrecisionWeightedRAG

    client = anthropic.Anthropic()
    rag = PrecisionWeightedRAG()
    examples = [
        ConflictingSourceQA.build_example("What is the default JWT expiry?", "24 hours", "1 hour", 0.9, 0.3),
        ConflictingSourceQA.build_example("What HTTP method does CORS preflight use?", "OPTIONS", "GET", 0.85, 0.4),
    ]
    results = []
    for ex in examples:
        prompt = rag.build_prompt(ex.question, ex.sources)
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.content[0].text.strip()
        correct = ex.correct_answer.lower() in answer.lower()
        conflict_flagged = "uncertain" in answer.lower() or "conflict" in answer.lower()
        results.append({
            "question": ex.question,
            "correct": correct,
            "conflict_flagged": conflict_flagged,
            "answer": answer,
        })
    return results


if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-2-precision-rag")
    with mlflow.start_run():
        mlflow.log_params({"hypothesis": "H2", "seed": RANDOM_SEED})
        results = run_experiment()
        accuracy = sum(r["correct"] for r in results) / len(results)
        conflict_rate = sum(r["conflict_flagged"] for r in results) / len(results)
        mlflow.log_metrics({"accuracy": accuracy, "conflict_flag_rate": conflict_rate})
        os.makedirs("data/experiments", exist_ok=True)
        with open("data/experiments/h2_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
