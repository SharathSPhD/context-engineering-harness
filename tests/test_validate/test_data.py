from experiments.validate.data import (
    NEXUSAPI_DOCS,
    QA_PAIRS,
    INCONGRUENT_DISTRACTORS,
    build_pre_shift_store,
    build_post_shift_store,
)
from src.avacchedaka.query import AvacchedakaQuery


def test_nexusapi_docs_has_all_domains():
    domains = {d["domain"] for d in NEXUSAPI_DOCS.values()}
    assert "web_security" in domains
    assert "infrastructure" in domains


def test_qa_pairs_cover_all_shift_topics():
    topics = {q["qualificand"] for q in QA_PAIRS}
    assert "auth" in topics
    assert "database" in topics
    assert "rate_limiting" in topics


def test_pre_shift_store_has_pre_elements():
    store = build_pre_shift_store()
    q = AvacchedakaQuery(qualificand="auth", condition="phase=pre_shift")
    results = store.retrieve(q, precision_threshold=0.0)
    assert len(results) >= 1
    assert any("24" in r.content for r in results)


def test_post_shift_store_sublates_pre_elements():
    store = build_pre_shift_store()
    build_post_shift_store(store)
    q_pre = AvacchedakaQuery(qualificand="auth", condition="phase=pre_shift")
    assert store.retrieve(q_pre) == []
    q_post = AvacchedakaQuery(qualificand="auth", condition="phase=post_shift")
    assert len(store.retrieve(q_post)) >= 1


def test_incongruent_distractors_are_unrelated():
    for d in INCONGRUENT_DISTRACTORS:
        assert "jwt" not in d.lower()
        assert "postgresql" not in d.lower()
