"""src.benchmarks — the validation harness used by every hypothesis (H1-H7),
the live case study, and the SWE-bench A/B head-to-head.

The public surface intentionally stays small:

    BenchmarkAdapter        — abstract base every dataset adapter inherits
    BenchmarkExample        — one input row (prompt, ground truth, metadata)
    BenchmarkResult         — one model outcome on one row
    BenchmarkRun            — the aggregate of N results under one condition
    HypothesisSpec          — declarative hypothesis declaration
    HypothesisOutcome       — the post-run verdict (target_met, p, CI, effect)
    ModelCaller (Protocol)  — pluggable callable; the dev tree wraps the
                              CLIBudgetScheduler, the shipped plugin uses a
                              direct claude CLI invocation
    MultiSeedRunner         — orchestrates seed-replicated A/B runs
    bootstrap_ci            — non-parametric percentile bootstrap CI
    paired_permutation_test — exact-when-small / sampled-otherwise paired test
    cohens_d                — paired Cohen's d effect size

Adapters live under src/benchmarks/adapters/ and self-register at import time.
"""
from .base import (
    BenchmarkAdapter,
    BenchmarkExample,
    BenchmarkResult,
    BenchmarkRun,
    ModelCaller,
    ModelOutput,
)
from .hypothesis import HypothesisOutcome, HypothesisSpec, TargetDirection
from .runner import MultiSeedRunner, RunnerConfig
from .stats import bootstrap_ci, cohens_d, paired_permutation_test

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkExample",
    "BenchmarkResult",
    "BenchmarkRun",
    "HypothesisOutcome",
    "HypothesisSpec",
    "ModelCaller",
    "ModelOutput",
    "MultiSeedRunner",
    "RunnerConfig",
    "TargetDirection",
    "bootstrap_ci",
    "cohens_d",
    "paired_permutation_test",
]
