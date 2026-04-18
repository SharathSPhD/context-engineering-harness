"""Lightweight name → adapter-class registry. Adapters self-register on import."""
from __future__ import annotations

from typing import Type

from .base import BenchmarkAdapter

_REGISTRY: dict[str, Type[BenchmarkAdapter]] = {}


def register(adapter_cls: Type[BenchmarkAdapter]) -> Type[BenchmarkAdapter]:
    """Decorator: register adapter under its `.name` class attribute."""
    name = getattr(adapter_cls, "name", "")
    if not name:
        raise ValueError(f"adapter {adapter_cls.__name__} must set a non-empty `name`")
    if name in _REGISTRY and _REGISTRY[name] is not adapter_cls:
        raise ValueError(f"adapter name {name!r} already registered")
    _REGISTRY[name] = adapter_cls
    return adapter_cls


def get(name: str) -> Type[BenchmarkAdapter]:
    if name not in _REGISTRY:
        raise KeyError(f"no adapter registered under {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_names() -> list[str]:
    return sorted(_REGISTRY)
