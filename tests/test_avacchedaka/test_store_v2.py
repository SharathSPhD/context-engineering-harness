"""v2 store invariants — duplicate-id rejection, scoped compress, token-exact window."""
from __future__ import annotations

import pytest

from src.avacchedaka.element import AvacchedakaConditions, ContextElement
from src.avacchedaka.query import AvacchedakaQuery
from src.avacchedaka.store import ContextStore


def _elem(id_: str, content: str, precision: float, qualificand: str = "auth", condition: str = "task_type=qa"):
    return ContextElement(
        id=id_,
        content=content,
        precision=precision,
        avacchedaka=AvacchedakaConditions(
            qualificand=qualificand, qualifier="prop", condition=condition
        ),
    )


def test_insert_duplicate_id_raises_by_default():
    store = ContextStore()
    store.insert(_elem("dup", "first", 0.8))
    with pytest.raises(ValueError, match="already exists"):
        store.insert(_elem("dup", "second", 0.9))


def test_insert_overwrite_replaces_in_place():
    store = ContextStore()
    store.insert(_elem("dup", "first", 0.8))
    store.insert(_elem("dup", "second", 0.9), overwrite=True)
    assert store.get("dup").content == "second"
    assert store.get("dup").precision == 0.9


def test_compress_scoped_by_qualificand_only_touches_matching():
    store = ContextStore()
    store.insert(_elem("a-1", "auth", 0.2, qualificand="auth"))
    store.insert(_elem("d-1", "db", 0.2, qualificand="db"))
    compressed = store.compress(precision_threshold=0.3, qualificand="auth")
    assert compressed == ["a-1"]
    assert store.get("d-1").precision == 0.2
    assert store.get("a-1").precision == 0.0


def test_compress_scoped_by_task_context_subset():
    store = ContextStore()
    store.insert(_elem("e1", "x", 0.2, condition="task_type=qa AND tier=prod"))
    store.insert(_elem("e2", "y", 0.2, condition="task_type=qa AND tier=dev"))
    compressed = store.compress(precision_threshold=0.3, task_context="task_type=qa AND tier=prod")
    assert compressed == ["e1"]
    assert store.get("e2").precision == 0.2


def test_to_context_window_respects_token_budget():
    store = ContextStore()
    long_content = " ".join(["alpha"] * 500)
    short_content = "short fact"
    store.insert(_elem("long", long_content, 0.95))
    store.insert(_elem("short", short_content, 0.9))
    query = AvacchedakaQuery(qualificand="auth", condition="task_type=qa")
    window = store.to_context_window(query, max_tokens=20)
    assert long_content not in window
