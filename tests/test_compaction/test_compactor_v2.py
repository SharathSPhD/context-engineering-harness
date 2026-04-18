"""G7 fix — compactor threads `qualificand` and `task_context` through to the store."""
from __future__ import annotations

from src.avacchedaka.element import AvacchedakaConditions, ContextElement
from src.avacchedaka.store import ContextStore
from src.compaction.compactor import BoundaryTriggeredCompactor


def _store_with_two_qualificands() -> ContextStore:
    store = ContextStore()
    store.insert(ContextElement(
        id="auth-low",
        content="x",
        precision=0.2,
        avacchedaka=AvacchedakaConditions(qualificand="auth", qualifier="p", condition="task_type=qa"),
    ))
    store.insert(ContextElement(
        id="db-low",
        content="x",
        precision=0.2,
        avacchedaka=AvacchedakaConditions(qualificand="db", qualifier="p", condition="task_type=qa"),
    ))
    return store


def test_compact_at_boundary_with_qualificand_scope_is_targeted():
    store = _store_with_two_qualificands()
    compactor = BoundaryTriggeredCompactor(store, compress_threshold=0.3)
    compressed = compactor.compact_at_boundary(qualificand="auth")
    assert compressed == ["auth-low"]
    assert store.get("db-low").precision == 0.2


def test_threshold_compact_with_task_context_scope_is_targeted():
    store = ContextStore()
    store.insert(ContextElement(
        id="prod",
        content="x",
        precision=0.2,
        avacchedaka=AvacchedakaConditions(qualificand="auth", qualifier="p", condition="task_type=qa AND tier=prod"),
    ))
    store.insert(ContextElement(
        id="dev",
        content="x",
        precision=0.2,
        avacchedaka=AvacchedakaConditions(qualificand="auth", qualifier="p", condition="task_type=qa AND tier=dev"),
    ))
    compactor = BoundaryTriggeredCompactor(store, compress_threshold=0.3)
    compressed = compactor.threshold_compact(
        token_count=1000,
        token_threshold=500,
        qualificand="auth",
        task_context="task_type=qa AND tier=prod",
    )
    assert compressed == ["prod"]
    assert store.get("dev").precision == 0.2
