"""HuggingFace loader for real long-context benchmarks.

Centralized so every adapter handles the "datasets isn't installed" and
"network isn't available" cases identically. The loader returns a list of
plain dicts so adapter code never has to care about whether a `Dataset` or
a `DatasetDict` came back.

Cache directory: respects HF defaults (`HF_HOME`, `HF_DATASETS_CACHE`).
Calls are network-blocking on first hit; once cached they are local.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class HFUnavailable(RuntimeError):
    """Raised when `datasets` isn't installed or the dataset cannot be fetched.

    Adapters catch this and fall back to the synthetic generator with a
    clear log line so we never silently report a fake number for a tier
    the user explicitly asked us to validate against the real dataset.
    """


def load_hf_examples(
    *,
    dataset_id: str,
    split: str = "test",
    config: str | None = None,
    n: int | None = None,
    seed: int = 0,
    streaming: bool = False,
) -> list[dict[str, Any]]:
    """Load a HuggingFace dataset and return up to `n` examples as dicts.

    Examples are deterministically subsampled with a hash-stable shuffle
    so different (seed, n) tuples are reproducible across machines.
    """
    if os.environ.get("CEH_DISABLE_HF") == "1":
        raise HFUnavailable("CEH_DISABLE_HF=1 set; refusing to fetch from HuggingFace")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover — depends on extras
        raise HFUnavailable(
            "datasets is not installed; `pip install '.[benchmarks]'` to enable real loaders"
        ) from exc

    try:
        ds = load_dataset(
            dataset_id,
            name=config,
            split=split,
            streaming=streaming,
        )
    except Exception as exc:  # noqa: BLE001 — datasets surfaces many error classes
        raise HFUnavailable(
            f"failed to load HF dataset {dataset_id} (split={split}, config={config}): {exc}"
        ) from exc

    if streaming:
        examples = list(ds.take(n)) if n else list(ds)
        return examples

    rows = list(ds)
    if n is not None and n < len(rows):
        import random as _random

        rng = _random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:n]
    return rows
