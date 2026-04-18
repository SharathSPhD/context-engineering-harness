"""Project-wide configuration.

Loads ``config.toml`` from the repo root using the stdlib ``tomllib`` module
(Python 3.11+, no extra dependency). Falls back to built-in defaults when
``config.toml`` is absent or a key is missing.

Usage::

    from src.config import config

    model = config.fast_model      # e.g. "claude-haiku-4-5"
    threshold = config.compress_threshold  # e.g. 0.3
"""
from __future__ import annotations

import tomllib
from pathlib import Path

# ── Built-in defaults ─────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "models": {
        "fast": "claude-haiku-4-5",
        "smart": "claude-sonnet-4-6",
    },
    "tokens": {
        "fast_max": 256,
        "smart_max": 1024,
    },
    "avacchedaka": {
        "compress_threshold": 0.3,
        "default_precision_threshold": 0.3,
    },
    "compaction": {
        "surprise_threshold": 0.75,
        "token_threshold": 500,
    },
    "forgetting": {
        "decay_factor": 0.9,
        "keep_threshold": 0.3,
        "keep_newest": 4,
    },
    "random_seed": 42,
}

# Allow tests (or reload callers) to pre-set _CONFIG_PATH before reload.
# globals().get() preserves any externally patched value during importlib.reload().
_CONFIG_PATH: Path = globals().get(
    "_CONFIG_PATH", Path(__file__).parent.parent / "config.toml"
)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursing into nested dicts."""
    result = {**base}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load() -> dict:
    try:
        with open(_CONFIG_PATH, "rb") as f:
            user = tomllib.load(f)
        return _deep_merge(_DEFAULTS, user)
    except FileNotFoundError:
        return _DEFAULTS.copy()


_raw: dict = _load()


class _Config:
    """Typed property access to configuration values."""

    # ── Models ───────────────────────────────────────────────────────────────
    @property
    def fast_model(self) -> str:
        """Model for low-latency calls (ManasAgent, validation H1/H2/H7)."""
        return _raw["models"]["fast"]

    @property
    def smart_model(self) -> str:
        """Model for high-accuracy calls (BuddhiAgent, KhyativadaClassifier)."""
        return _raw["models"]["smart"]

    # ── Token limits ──────────────────────────────────────────────────────────
    @property
    def fast_max_tokens(self) -> int:
        return _raw["tokens"]["fast_max"]

    @property
    def smart_max_tokens(self) -> int:
        return _raw["tokens"]["smart_max"]

    # ── Avacchedaka ───────────────────────────────────────────────────────────
    @property
    def compress_threshold(self) -> float:
        return _raw["avacchedaka"]["compress_threshold"]

    @property
    def default_precision_threshold(self) -> float:
        return _raw["avacchedaka"]["default_precision_threshold"]

    # ── Compaction ────────────────────────────────────────────────────────────
    @property
    def surprise_threshold(self) -> float:
        return _raw["compaction"]["surprise_threshold"]

    @property
    def token_threshold(self) -> int:
        return _raw["compaction"]["token_threshold"]

    # ── Forgetting ────────────────────────────────────────────────────────────
    @property
    def decay_factor(self) -> float:
        return _raw["forgetting"]["decay_factor"]

    @property
    def keep_threshold(self) -> float:
        return _raw["forgetting"]["keep_threshold"]

    @property
    def keep_newest(self) -> int:
        return _raw["forgetting"]["keep_newest"]

    # ── Misc ──────────────────────────────────────────────────────────────────
    @property
    def random_seed(self) -> int:
        return _raw["random_seed"]


#: Singleton config object — import and use this directly.
config = _Config()
