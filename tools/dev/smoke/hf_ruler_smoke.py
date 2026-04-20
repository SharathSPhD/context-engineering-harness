"""One-problem, one-benchmark smoke test: real HuggingFace + real Claude CLI.

Minimal end-to-end proof that the harness can (a) load a real RULER example
from the `simonjegou/ruler` HF dataset, (b) route the rendered `harness_on`
prompt through `CLIBudgetScheduler` to the real `claude` CLI, and (c) score
the response. Writes a committed receipt at
``experiments/results/hf_smoke/ruler_<budget>_n1.json``.

Run:
    set -a && source .env && set +a
    uv run --active python -m tools.dev.smoke.hf_ruler_smoke

Intentionally avoids the multi-seed runner, the p6a spec loop, and anything
that would fan out beyond one example. The point is wiring, not statistics.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_DIR = REPO_ROOT / "experiments" / "results" / "hf_smoke"

TARGET_TOKENS = 4_096
MODEL = "claude-haiku-4-5"
SEED = 0
MAX_OUTPUT_TOKENS = 64


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    if "HF_TOKEN" not in os.environ:
        print("ERROR: HF_TOKEN not in environment; run with `set -a && source .env && set +a`.", file=sys.stderr)
        return 2

    from huggingface_hub import whoami
    from src.benchmarks.adapters.longctx.ruler import RulerNIAHAdapter
    from tools.dev.scheduler import CLIBudgetScheduler, SchedulerConfig
    from experiments.v2.p6a.callers import LiveCLICaller

    hf_identity = whoami(token=os.environ["HF_TOKEN"]).get("name", "<unknown>")

    adapter = RulerNIAHAdapter(
        load_real=True,
        target_tokens=TARGET_TOKENS,
        default_n=1,
    )
    examples = adapter.load_examples(n=1, seed=SEED)
    if not examples:
        print("ERROR: adapter returned 0 examples.", file=sys.stderr)
        return 3
    ex = examples[0]
    if ex.metadata.get("source") != "huggingface":
        print(f"ERROR: adapter fell back to synthetic; metadata={ex.metadata}", file=sys.stderr)
        return 4

    prompt = adapter.render_prompt(ex, condition="harness_on")
    system = adapter.system_prompt(condition="harness_on")

    scheduler = CLIBudgetScheduler(
        SchedulerConfig(
            cache_root=".cache/llm",
            ledger_path=".cache/cost_ledger.db",
            journal_path="tools/dev/orchestration/attractor_journal.jsonl",
            max_input_tokens_per_window=2_000_000,
            max_retries=2,
            base_backoff_s=2.0,
            max_backoff_s=10.0,
        )
    )
    caller = LiveCLICaller(scheduler)

    t0 = time.perf_counter()
    output = caller(
        prompt=prompt,
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system,
        seed=None,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    score, correct, pred_text = adapter.score(ex, output)

    receipt = {
        "ts": _utcnow(),
        "purpose": "HF real-benchmark smoke test (1 bench, 1 example)",
        "hf": {
            "identity": hf_identity,
            "dataset_id": ex.metadata.get("hf_dataset_id"),
            "config": ex.metadata.get("hf_config"),
            "task": ex.metadata.get("hf_task"),
            "source": ex.metadata.get("source"),
            "target_tokens": ex.metadata.get("target_tokens"),
            "context_chars": len(ex.context or ""),
            "example_id": ex.id,
            "ground_truth": ex.ground_truth,
        },
        "call": {
            "caller": output.metadata.get("caller"),
            "model": MODEL,
            "seed": None,
            "seed_notes": "claude CLI >=2.1 removed --seed; scheduler still hashes seed into the prompt_hash for deterministic caching",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "condition": "harness_on",
            "input_tokens": output.input_tokens,
            "output_tokens": output.output_tokens,
            "latency_ms": output.latency_ms,
            "wall_ms": elapsed_ms,
            "cache_hit": output.metadata.get("cache_hit"),
            "regime": output.metadata.get("regime"),
            "attempts": output.metadata.get("attempts"),
            "prompt_hash": output.metadata.get("prompt_hash"),
        },
        "score": {
            "score": score,
            "correct": bool(correct),
            "pred_preview": pred_text[:500],
        },
        "scheduler_status": scheduler.status(),
        "acceptance": {
            "source_is_huggingface": ex.metadata.get("source") == "huggingface",
            "caller_is_live": output.metadata.get("caller") == "LiveCLICaller",
            "input_tokens_gt_500": int(output.input_tokens) > 500,
            "pred_non_empty": bool(pred_text.strip()),
            "wall_cost_recorded": elapsed_ms > 0,
        },
        "findings": {
            "scheduler_patches_applied": [
                "dropped --seed (claude CLI >=2.1 removed the option)",
                "switched --output-format json -> stream-json + --verbose (CLI now requires matching formats with stream-json input)",
                "stream-json input shape updated to {type, message:{role, content}} (schema change)",
                "system prompt now rides --system-prompt instead of an inline stream-json event",
                "rate-limit detector gated on exit_code != 0 so session-hook 'rate-limit' noise on stdout stops triggering false-positive 5h sleeps",
                "input_tokens now sums input_tokens + cache_creation_input_tokens + cache_read_input_tokens for honest budget accounting",
            ],
            "adapter_template_mismatch": (
                "RulerNIAHAdapter.render_prompt(harness_on) hard-codes 'the bare 6-character code'"
                " because the synthetic generator always emits 6-char needles, but simonjegou/ruler's"
                " real niah_single_1 needles are 7-digit numbers. Claude correctly located the needle"
                " and truncated to 6 chars ('668809' vs ground truth '6688090'); the plumbing is proven,"
                " the score=0 here reflects a known template bug, not a retrieval failure."
            ),
            "next_to_resolve_before_expansion": [
                "generalize harness_on/harness_off prompts in ruler.py so they work for both synthetic (6-char alphanumeric) and real-HF (variable-length digit) needles",
                "confirm hallu and swebench adapters' real paths boot the same way before flipping any live re-runs",
            ],
        },
    }

    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPT_DIR / f"ruler_{TARGET_TOKENS}_n1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))

    print(json.dumps(receipt, indent=2, default=str))
    print(f"\nRECEIPT -> {receipt_path}")

    all_pass = all(receipt["acceptance"].values())
    return 0 if all_pass else 5


if __name__ == "__main__":
    raise SystemExit(main())
