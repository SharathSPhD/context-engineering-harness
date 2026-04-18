import pytest
from datetime import datetime, timedelta
from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.store import ContextStore
from src.forgetting.schedules import (
    NoForgetting, FixedCompaction, RecencyWeightedForgetting,
    RewardWeightedForgetting, BadhaFirstForgetting,
)


def _make_store(n: int, base_precision: float = 0.8, age_hours: float = 0.0) -> ContextStore:
    store = ContextStore()
    for i in range(n):
        elem = ContextElement(
            id=f"e-{i:03d}",
            content=f"Element {i}",
            precision=base_precision,
            avacchedaka=AvacchedakaConditions(qualificand="test", qualifier="prop", condition="z"),
            timestamp=datetime.utcnow() - timedelta(hours=age_hours + (n - i)),
        )
        store.insert(elem)
    return store


def test_no_forgetting_retains_all():
    store = _make_store(5)
    schedule = NoForgetting(store)
    assert schedule.apply() == []


def test_fixed_compaction_keeps_newest():
    store = _make_store(10)
    schedule = FixedCompaction(store, keep_newest=5)
    removed = schedule.apply()
    assert len(removed) == 5


def test_fixed_compaction_keeps_zero_removes_all():
    store = _make_store(5)
    schedule = FixedCompaction(store, keep_newest=0)
    removed = schedule.apply()
    assert len(removed) == 5


def test_recency_weighted_removes_old_low_precision():
    store = _make_store(3, base_precision=0.8, age_hours=1000)
    schedule = RecencyWeightedForgetting(store, decay_factor=0.5)
    removed = schedule.apply()
    assert len(removed) == 3


def test_recency_weighted_keeps_fresh():
    store = _make_store(3, base_precision=0.9, age_hours=0)
    schedule = RecencyWeightedForgetting(store, decay_factor=0.9)
    removed = schedule.apply()
    assert removed == []


def test_recency_weighted_invalid_decay_raises():
    store = _make_store(1)
    with pytest.raises(ValueError, match="decay_factor"):
        RecencyWeightedForgetting(store, decay_factor=0.0)


def test_reward_weighted_removes_low_salience():
    store = _make_store(3)
    for eid, elem in list(store._elements.items()):
        import dataclasses
        store._elements[eid] = dataclasses.replace(elem, salience={"task_relevance": 0.1})
    schedule = RewardWeightedForgetting(store, keep_threshold=0.5)
    removed = schedule.apply()
    assert len(removed) == 3


def test_reward_weighted_keeps_high_salience():
    store = _make_store(3)
    for eid, elem in list(store._elements.items()):
        import dataclasses
        store._elements[eid] = dataclasses.replace(elem, salience={"task_relevance": 0.9})
    schedule = RewardWeightedForgetting(store, keep_threshold=0.5)
    removed = schedule.apply()
    assert removed == []


def test_badha_first_clears_sublated():
    store = _make_store(3)
    newer = ContextElement(
        id="newer-001", content="Updated.", precision=0.9,
        avacchedaka=AvacchedakaConditions(qualificand="test", qualifier="prop", condition="z"),
    )
    store.insert(newer)
    store.sublate("e-000", "newer-001")
    schedule = BadhaFirstForgetting(store)
    removed = schedule.apply()
    assert "e-000" in removed
    assert "newer-001" not in removed


def test_badha_first_leaves_non_sublated():
    store = _make_store(3)
    schedule = BadhaFirstForgetting(store)
    removed = schedule.apply()
    assert removed == []
