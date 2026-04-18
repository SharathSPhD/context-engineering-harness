"""P6-A — multi-seed × multi-model re-runs of H1 (RULER) and H2 (HELMET-RAG / NoCha).

Two execution modes:
  - "mock"  (default): a deterministic harness simulator stands in for the
    `claude` CLI so the entire pipeline (adapter → runner → stats →
    HypothesisOutcome) can be exercised free in CI on the dev box.
  - "live"           : routes every model call through CLIBudgetScheduler,
    which pays $$ but writes a full audit trail in the cost ledger.

Both modes use exactly the same `MultiSeedRunner` and emit identical JSON
schema, so swapping in real CLI traffic does not require any downstream
changes to analysis or paper-figure code.
"""
