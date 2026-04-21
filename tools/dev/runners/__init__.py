"""Runtime runners for live-HF benchmark re-executions.

Exports the ``CheckpointedBundleRunner`` used by the live-HF entry point to
execute per-example benchmark calls with:

* Per-example JSONL checkpointing (resume across process restarts)
* Graceful :class:`QuotaExhausted` handling (emit ``*_partial.json`` + exit 0)
* Compatibility with the existing ``experiments/v2/p6a/_summary.json`` schema

This module is dev tooling only; the shipped plugin never imports it.
"""
from .checkpointed_runner import (
    CheckpointedBundleRunner,
    CheckpointRecord,
    PartialRunExit,
    load_checkpoint,
)

__all__ = [
    "CheckpointedBundleRunner",
    "CheckpointRecord",
    "PartialRunExit",
    "load_checkpoint",
]
