from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.store import ContextStore
from src.compaction.detector import EventBoundaryDetector
from src.compaction.compactor import BoundaryTriggeredCompactor, BoundaryTriggeredSession


def _make_store(n: int, precision: float = 0.8) -> ContextStore:
    store = ContextStore()
    for i in range(n):
        store.insert(ContextElement(
            id=f"e-{i:03d}",
            content=f"Element {i}",
            precision=precision,
            avacchedaka=AvacchedakaConditions(qualificand="test", qualifier="p", condition="z"),
        ))
    return store


def test_compact_at_boundary_compresses_low_precision():
    store = _make_store(3, precision=0.2)
    compactor = BoundaryTriggeredCompactor(store, compress_threshold=0.3)
    removed = compactor.compact_at_boundary()
    assert len(removed) == 3


def test_threshold_compact_triggers_above_threshold():
    store = _make_store(5, precision=0.2)
    compactor = BoundaryTriggeredCompactor(store, compress_threshold=0.3)
    removed = compactor.threshold_compact(token_count=1000, token_threshold=500)
    assert len(removed) == 5


def test_threshold_compact_no_action_below_threshold():
    store = _make_store(5, precision=0.2)
    compactor = BoundaryTriggeredCompactor(store, compress_threshold=0.3)
    removed = compactor.threshold_compact(token_count=100, token_threshold=500)
    assert removed == []


def test_boundary_session_compacts_on_surprise_spike():
    store = _make_store(3, precision=0.2)
    detector = EventBoundaryDetector(surprise_threshold=0.8)
    session = BoundaryTriggeredSession(store, detector, compress_threshold=0.3)
    surprises = [0.1, 0.9, 0.1]
    compressed = session.process_surprises(surprises, step=1)
    assert len(compressed) == 3
    assert len(session.compaction_events) == 1
    assert session.compaction_events[0]["step"] == 1


def test_boundary_session_no_compaction_without_spike():
    store = _make_store(3, precision=0.2)
    detector = EventBoundaryDetector(surprise_threshold=0.8)
    session = BoundaryTriggeredSession(store, detector, compress_threshold=0.3)
    compressed = session.process_surprises([0.1, 0.2, 0.1])
    assert compressed == []
    assert len(session.compaction_events) == 0
