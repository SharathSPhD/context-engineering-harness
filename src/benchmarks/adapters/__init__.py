"""Benchmark adapters. Importing this package self-registers all adapters."""
from . import hallu, longctx  # noqa: F401  (registration side effect)

__all__ = ["hallu", "longctx"]
