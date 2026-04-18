"""Tokenizer helper used to budget context windows precisely.

Anthropic does not currently publish their tokenizer; tiktoken's `o200k_base`
(GPT-4o family) is the closest publicly-available tokenizer and is what we
use as a conservative proxy for token-budget arithmetic. The estimate is an
overcount for Claude in practice (Claude uses fewer tokens for whitespace),
which is what we want — we'd rather under-fill the window than overflow it.

Falls back to `len(text) // 4` (the historical char-count heuristic) only
when tiktoken is unavailable; this keeps cold environments operational while
producing a clear log line that says the precise budget is degraded.
"""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=4)
def _encoder(name: str):
    try:
        import tiktoken
    except ImportError:  # pragma: no cover — fallback path
        logger.warning(
            "tiktoken not installed; falling back to char/4 heuristic. "
            "Install tiktoken for tokenizer-exact context budgeting."
        )
        return None
    try:
        return tiktoken.get_encoding(name)
    except (KeyError, ValueError) as exc:
        logger.warning("tiktoken encoding %r unavailable (%s); falling back", name, exc)
        return None


def count_tokens(text: str, *, encoding: str = DEFAULT_ENCODING) -> int:
    """Tokenizer-exact count when tiktoken is available; char/4 fallback otherwise."""
    enc = _encoder(encoding)
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))
