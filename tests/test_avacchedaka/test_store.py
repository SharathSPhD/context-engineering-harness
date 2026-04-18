import pytest
from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.store import ContextStore
from src.avacchedaka.query import AvacchedakaQuery


def _make_element(id_, content, precision, qualificand="auth_module", condition="task_type=code_review"):
    return ContextElement(
        id=id_,
        content=content,
        precision=precision,
        avacchedaka=AvacchedakaConditions(
            qualificand=qualificand,
            qualifier="token_expiry",
            condition=condition,
        ),
    )


def test_insert_and_retrieve(sample_element):
    store = ContextStore()
    store.insert(sample_element)
    query = AvacchedakaQuery(qualificand="auth_module", condition="task_type=code_review")
    results = store.retrieve(query)
    assert len(results) == 1
    assert results[0].id == "test-001"


def test_retrieve_excludes_sublated(sample_element):
    store = ContextStore()
    store.insert(sample_element)
    newer = _make_element("test-002", "JWT tokens now use 1h expiry.", 0.95)
    store.insert(newer)
    store.sublate(element_id="test-001", by_element_id="test-002")
    query = AvacchedakaQuery(qualificand="auth_module", condition="task_type=code_review")
    results = store.retrieve(query)
    assert len(results) == 1
    assert results[0].id == "test-002"


def test_sublation_does_not_delete(sample_element):
    store = ContextStore()
    store.insert(sample_element)
    newer = _make_element("test-002", "Updated.", 0.95)
    store.insert(newer)
    store.sublate("test-001", "test-002")
    elem = store.get("test-001")
    assert elem is not None
    assert elem.precision == 0.0
    assert elem.sublated_by == "test-002"


def test_retrieve_below_precision_threshold_excluded(sample_element):
    store = ContextStore()
    store.insert(sample_element)
    query = AvacchedakaQuery(qualificand="auth_module", condition="task_type=code_review")
    results = store.retrieve(query, precision_threshold=0.95)
    assert len(results) == 0


def test_compress_returns_ids_of_low_precision():
    store = ContextStore()
    low = _make_element("low-001", "Old info.", 0.2)
    store.insert(low)
    compressed = store.compress(precision_threshold=0.3)
    assert "low-001" in compressed


def test_sublate_raises_on_unknown_id():
    store = ContextStore()
    with pytest.raises(KeyError):
        store.sublate("nonexistent", "other")


def test_to_context_window_format(sample_element):
    store = ContextStore()
    store.insert(sample_element)
    query = AvacchedakaQuery(qualificand="auth_module", condition="task_type=code_review")
    window = store.to_context_window(query)
    assert "auth_module" in window
    assert "precision=0.90" in window


def test_insert_rejects_sublated_with_nonzero_precision():
    store = ContextStore()
    with pytest.raises(ValueError, match="sublation invariant"):
        store.insert(ContextElement(
            id="bad",
            content="content",
            precision=0.9,
            avacchedaka=AvacchedakaConditions(
                qualificand="auth_module", qualifier="prop", condition="task_type=code_review"
            ),
            sublated_by="other-id",
        ))


def test_retrieve_sorted_by_precision_descending():
    store = ContextStore()
    store.insert(_make_element("e1", "low", 0.6))
    store.insert(_make_element("e2", "high", 0.9))
    store.insert(_make_element("e3", "mid", 0.75))
    query = AvacchedakaQuery(qualificand="auth_module", condition="task_type=code_review")
    results = store.retrieve(query)
    precisions = [r.precision for r in results]
    assert precisions == sorted(precisions, reverse=True)
