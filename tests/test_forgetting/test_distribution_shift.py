from src.forgetting.distribution_shift import DistributionShiftBenchmark
from src.avacchedaka.query import AvacchedakaQuery


def test_shift_task_has_pre_and_post_elements():
    bench = DistributionShiftBenchmark()
    task = bench.build_jwt_shift()
    assert len(task.pre_shift_elements) >= 1
    assert len(task.post_shift_elements) >= 1
    assert task.pre_shift_answer != task.post_shift_answer


def test_pre_shift_store_only_has_pre_elements():
    bench = DistributionShiftBenchmark()
    task = bench.build_jwt_shift()
    store = bench.build_store_pre_shift(task)
    query = AvacchedakaQuery(qualificand="auth", condition="phase=pre_shift")
    results = store.retrieve(query, precision_threshold=0.0)
    assert len(results) == len(task.pre_shift_elements)


def test_apply_shift_sublates_pre_elements():
    bench = DistributionShiftBenchmark()
    task = bench.build_jwt_shift()
    store = bench.build_store_pre_shift(task)
    bench.apply_shift(store, task)
    pre_query = AvacchedakaQuery(qualificand="auth", condition="phase=pre_shift")
    pre_results = store.retrieve(pre_query)
    assert pre_results == []
    post_query = AvacchedakaQuery(qualificand="auth", condition="phase=post_shift")
    post_results = store.retrieve(post_query)
    assert len(post_results) == len(task.post_shift_elements)


def test_apply_shift_uses_sublation_not_deletion():
    """CLAUDE.md invariant: sublation never deletes — it sets precision=0.0 and sublated_by."""
    bench = DistributionShiftBenchmark()
    task = bench.build_jwt_shift()
    store = bench.build_store_pre_shift(task)
    pre_ids = [e.id for e in task.pre_shift_elements]
    bench.apply_shift(store, task)
    # Pre-shift elements must still exist in the store (audit trail preserved)
    for pre_id in pre_ids:
        assert pre_id in store._elements, f"{pre_id} was deleted — use sublate(), not delete()"
        elem = store._elements[pre_id]
        assert elem.precision == 0.0, f"{pre_id} precision should be 0.0 after sublation"
        assert elem.sublated_by is not None, f"{pre_id} sublated_by should be set"


def test_apply_shift_default_task():
    """apply_shift(store) with no task argument should use the JWT shift by default."""
    bench = DistributionShiftBenchmark()
    task = bench.build_jwt_shift()
    store = bench.build_store_pre_shift(task)
    # Call with only the store — task defaults to None → uses build_jwt_shift()
    bench.apply_shift(store)
    # Post-shift elements should be present (since apply_shift uses default jwt task)
    post_query = AvacchedakaQuery(qualificand="auth", condition="phase=post_shift")
    post_results = store.retrieve(post_query)
    assert len(post_results) > 0
