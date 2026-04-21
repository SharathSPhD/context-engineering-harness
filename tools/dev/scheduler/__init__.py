"""CLI-strict, token-aware scheduler for the v2 rebuild.

Wraps the `claude` CLI with:
- content-addressable disk cache (free reruns)
- prompt-cache markers (cheap reruns within a window)
- per-call sqlite cost ledger (auditable spend trail; expected $0)
- rate-limit detection + exponential backoff + sleep to next 5h window
- regime emission (HALT / OK / CYCLING) for AttractorFlow integration

This is dev tooling only. The shipped plugin uses a much simpler
`/budget` command that just reads the user's local ledger.
"""
from .cache import DiskCache, PromptCache, prompt_hash_of
from .cost_ledger import CostLedger
from .rate_limit import RateLimitDetector, RateLimitState
from .cli_budget import (
    CLIBudgetScheduler,
    QuotaExhausted,
    SchedulerConfig,
    SchedulerError,
    SubmitResult,
)

__all__ = [
    "CLIBudgetScheduler",
    "CostLedger",
    "DiskCache",
    "PromptCache",
    "QuotaExhausted",
    "RateLimitDetector",
    "RateLimitState",
    "SchedulerConfig",
    "SchedulerError",
    "SubmitResult",
    "prompt_hash_of",
]
