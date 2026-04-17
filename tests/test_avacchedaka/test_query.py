from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.query import AvacchedakaQuery


def _elem(qualificand="auth", condition="task_type=qa", precision=0.9):
    return ContextElement(
        id="q-test",
        content="test",
        precision=precision,
        avacchedaka=AvacchedakaConditions(
            qualificand=qualificand, qualifier="prop", condition=condition
        ),
    )


def test_matches_same_qualificand():
    query = AvacchedakaQuery(qualificand="auth", condition="task_type=qa")
    assert query.matches(_elem(qualificand="auth", condition="task_type=qa")) is True


def test_matches_qualificand_mismatch():
    query = AvacchedakaQuery(qualificand="auth", condition="task_type=qa")
    assert query.matches(_elem(qualificand="deploy", condition="task_type=qa")) is False


def test_matches_empty_condition_passthrough():
    query = AvacchedakaQuery(qualificand="auth", condition="")
    assert query.matches(_elem(qualificand="auth", condition="task_type=qa")) is True


def test_matches_single_token_match():
    query = AvacchedakaQuery(qualificand="auth", condition="task_type=qa")
    assert query.matches(_elem(qualificand="auth", condition="task_type=qa AND env=prod")) is True


def test_matches_multi_token_and_all_present():
    query = AvacchedakaQuery(qualificand="auth", condition="task_type=qa AND env=prod")
    assert query.matches(_elem(qualificand="auth", condition="task_type=qa AND env=prod")) is True


def test_matches_multi_token_partial_miss():
    query = AvacchedakaQuery(qualificand="auth", condition="task_type=qa AND env=prod")
    assert query.matches(_elem(qualificand="auth", condition="task_type=qa")) is False
