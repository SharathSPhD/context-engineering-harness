"""Content-addressable LLM I/O cache + Anthropic prompt-cache helpers.

Two distinct caches:

1. `DiskCache` — hashes the *full call signature* (model, system, messages,
   max_tokens, temperature, seed) and stores the response JSON. A cache hit
   is a $0 / 0-token replay. This is what makes reruns of the experiment
   pipeline free.

2. `PromptCache` — wraps an Anthropic message with the `cache_control`
   marker so the upstream service can skip re-tokenizing the prefix on
   subsequent calls within the cache TTL. This is a within-window
   optimization (the sakshi prefix is the canonical target since it never
   changes).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def prompt_hash_of(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float = 0.0,
    seed: int | None = None,
) -> str:
    """Stable SHA-256 over the canonical call signature.

    JSON-canonicalize messages with sort_keys to make the hash deterministic
    across dict insertion order.
    """
    sig = {
        "model": model,
        "system": system,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "seed": seed,
    }
    payload = json.dumps(sig, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    raw: dict[str, Any]


class DiskCache:
    """Two-level content-addressable cache rooted at `root`.

    Layout: root/{first2}/{rest}.json — keeps any single dir under ~256 entries
    per top-level shard, which is fine for filesystems we'd realistically use.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if len(key) < 3:
            raise ValueError("cache key too short")
        return self.root / key[:2] / f"{key[2:]}.json"

    def get(self, key: str) -> CacheEntry | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return CacheEntry(
            text=data.get("text", ""),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            model=data.get("model", ""),
            raw=data,
        )

    def put(self, key: str, entry: CacheEntry) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": entry.text,
            "input_tokens": int(entry.input_tokens),
            "output_tokens": int(entry.output_tokens),
            "model": entry.model,
            **entry.raw,
        }
        # Atomic write: write to temp in same dir, fsync, rename.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(p.parent), delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, str(p))

    def clear(self) -> int:
        """Remove all cached entries; returns count removed (test helper)."""
        n = 0
        for shard in self.root.iterdir():
            if not shard.is_dir():
                continue
            for f in shard.iterdir():
                if f.suffix == ".json":
                    f.unlink()
                    n += 1
        return n


class PromptCache:
    """Helpers for Anthropic prompt-cache markers.

    The shipped plugin uses these on the sakshi prefix (which never changes),
    keeping per-call cost down without paying for re-tokenization every turn.
    """

    @staticmethod
    def cacheable_block(content: str, *, ttl: str = "5m") -> dict[str, Any]:
        """Wrap a string into a content-block with cache_control set."""
        return {
            "type": "text",
            "text": content,
            "cache_control": {"type": "ephemeral", "ttl": ttl},
        }

    @staticmethod
    def system_with_cache(static_prefix: str, dynamic_suffix: str = "") -> list[dict[str, Any]]:
        """Build a system-prompt list where the long static prefix is cacheable.

        Returns a list of content blocks suitable for Anthropic's `system=[...]`
        message API. The static prefix is wrapped with cache_control; the
        dynamic suffix (if any) is plain text appended after.
        """
        blocks: list[dict[str, Any]] = [PromptCache.cacheable_block(static_prefix)]
        if dynamic_suffix:
            blocks.append({"type": "text", "text": dynamic_suffix})
        return blocks
