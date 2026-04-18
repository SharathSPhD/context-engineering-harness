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
