"""Tiny CLI wrapper for the scheduler.

Usage:
    python -m tools.dev.scheduler.cli status
    python -m tools.dev.scheduler.cli ask --model claude-haiku-4-5 \\
        --system "be brief" --user "two-line summary of what TRIZ is"
    python -m tools.dev.scheduler.cli clear-cache
"""
from __future__ import annotations

import argparse
import json
import sys

from .cli_budget import CLIBudgetScheduler, SchedulerConfig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scheduler-cli")
    p.add_argument("--cache-root", default=".cache/llm")
    p.add_argument("--ledger-path", default=".cache/cost_ledger.db")
    p.add_argument("--window-hours", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Print current window status as JSON.")

    ask = sub.add_parser("ask", help="Send one prompt through the scheduler.")
    ask.add_argument("--model", required=True)
    ask.add_argument("--system", default="")
    ask.add_argument("--user", required=True)
    ask.add_argument("--max-tokens", type=int, default=512)
    ask.add_argument("--temperature", type=float, default=0.0)

    sub.add_parser("clear-cache", help="Delete every cached response.")

    args = p.parse_args(argv)

    cfg = SchedulerConfig(
        cache_root=args.cache_root,
        ledger_path=args.ledger_path,
        window_hours=args.window_hours,
        dry_run=args.dry_run,
    )
    sched = CLIBudgetScheduler(cfg)

    if args.cmd == "status":
        json.dump(sched.status(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    if args.cmd == "ask":
        result = sched.submit(
            model=args.model,
            system=args.system,
            messages=[{"role": "user", "content": args.user}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        json.dump(
            {
                "text": result.text,
                "cache_hit": result.cache_hit,
                "attempts": result.attempts,
                "regime": result.regime,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": result.latency_ms,
                "prompt_hash": result.prompt_hash,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    if args.cmd == "clear-cache":
        n = sched.cache.clear()
        sys.stdout.write(f"cleared {n} cached entries\n")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
