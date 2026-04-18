"""Run all H1-H7 validation modules and collect results.

Usage:
    python experiments/validate/runner.py            # full run (uses claude CLI)
    python experiments/validate/runner.py --skip-llm # algorithmic only (H4/H5/H6)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime


def run_all(skip_llm: bool = False) -> dict:
    """Run all hypotheses. Returns dict of results.

    Args:
        skip_llm: If True, skip H1/H2/H3/H7 which make claude CLI calls.
                  Useful for CI without claude auth.
    """
    from experiments.validate import (
        h1_schema,
        h2_rag,
        h3_agents,
        h4_compaction,
        h5_multiagent,
        h6_classifier,
        h7_forgetting,
    )

    modules = [
        ("H1", h1_schema.run_h1, True),    # (name, fn, needs_llm)
        ("H2", h2_rag.run_h2, True),
        ("H3", h3_agents.run_h3, True),
        ("H4", h4_compaction.run_h4, False),
        ("H5", h5_multiagent.run_h5, False),
        ("H6", h6_classifier.run_h6, False),
        ("H7", h7_forgetting.run_h7, True),
    ]

    results = {"generated_at": datetime.utcnow().isoformat(), "results": {}}

    for name, fn, needs_llm in modules:
        if skip_llm and needs_llm:
            results["results"][name] = {
                "hypothesis": name,
                "skipped": True,
                "reason": "skip_llm=True",
                "target_met": None,
            }
            print(f"  {name}... SKIPPED (--skip-llm)")
            continue

        print(f"  Running {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = fn()
            elapsed = time.time() - t0
            r["elapsed_s"] = round(elapsed, 1)
            results["results"][name] = r
            status = "✅ PASS" if r.get("target_met") else "❌ FAIL"
            print(f"{status} ({elapsed:.1f}s)")
        except Exception as exc:
            elapsed = time.time() - t0
            results["results"][name] = {
                "hypothesis": name,
                "error": str(exc),
                "target_met": False,
                "elapsed_s": round(elapsed, 1),
            }
            print(f"💥 ERROR: {exc}")

    return results


if __name__ == "__main__":
    skip = "--skip-llm" in sys.argv
    print("Context Engineering Harness — H1-H7 Validation")
    print("=" * 50)
    results = run_all(skip_llm=skip)

    os.makedirs("data/experiments", exist_ok=True)
    out_path = "data/experiments/validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Run 'make report' to generate docs/validation_report.md")
