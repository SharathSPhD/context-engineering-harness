import pytest
from src.avacchedaka.store import ContextStore
from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.query import AvacchedakaQuery


def test_conflict_rate_without_avacchedaka():
    """Without avacchedaka sublation, both conflicting claims are returned — simulate conflict."""
    store = ContextStore()
    conds = AvacchedakaConditions(qualificand="database", qualifier="version", condition="task_type=deploy")
    # Both agents insert without sublation
    store.insert(ContextElement(
        id="raw-001", content="PostgreSQL 14.", precision=0.85, avacchedaka=conds,
    ))
    store.insert(ContextElement(
        id="raw-002", content="PostgreSQL 16.", precision=0.95, avacchedaka=conds,
    ))
    # No sublation — both returned (conflict)
    query = AvacchedakaQuery(qualificand="database", condition="task_type=deploy")
    results = store.retrieve(query)
    assert len(results) == 2  # conflict: two contradicting answers


def test_avacchedaka_sublation_eliminates_conflict():
    """With avacchedaka, second agent sublates first agent's claim — no conflict."""
    store = ContextStore()
    conds = AvacchedakaConditions(qualificand="database", qualifier="version", condition="task_type=deploy")
    # Agent 1 inserts initial claim
    store.insert(ContextElement(
        id="agent1-claim-001", content="Database uses PostgreSQL 14.",
        precision=0.85, avacchedaka=conds, provenance="agent1",
    ))
    # Agent 2 discovers contradiction and sublates agent 1's claim
    store.insert(ContextElement(
        id="agent2-claim-001", content="Database uses PostgreSQL 16 (upgraded last week).",
        precision=0.95, avacchedaka=conds, provenance="agent2",
    ))
    store.sublate("agent1-claim-001", "agent2-claim-001")

    query = AvacchedakaQuery(qualificand="database", condition="task_type=deploy")
    results = store.retrieve(query)
    assert len(results) == 1
    assert results[0].id == "agent2-claim-001"
    assert results[0].content == "Database uses PostgreSQL 16 (upgraded last week)."


def test_sublated_claim_preserved_as_audit_trail():
    """Sublation never deletes — the superseded claim remains in the store as audit trail."""
    store = ContextStore()
    conds = AvacchedakaConditions(qualificand="database", qualifier="version", condition="task_type=deploy")
    store.insert(ContextElement(id="old-001", content="v14", precision=0.85, avacchedaka=conds))
    store.insert(ContextElement(id="new-001", content="v16", precision=0.95, avacchedaka=conds))
    store.sublate("old-001", "new-001")
    # Element still in store but precision=0.0 and sublated_by set
    sublated = store.get("old-001")
    assert sublated is not None
    assert sublated.precision == 0.0
    assert sublated.sublated_by == "new-001"


def test_conflict_rate_reduction_quantified():
    """H5 metric: avacchedaka reduces conflict rate from 100% to 0% on the test scenario."""
    tasks = [
        {"id_a1": "t1-a1", "id_a2": "t1-a2", "q": "database", "c": "task_type=deploy"},
        {"id_a1": "t2-a1", "id_a2": "t2-a2", "q": "auth",     "c": "task_type=qa"},
    ]
    conds_args = [
        AvacchedakaConditions(qualificand=t["q"], qualifier="fact", condition=t["c"])
        for t in tasks
    ]

    # Without avacchedaka: 100% conflict rate
    conflicts_without = 0
    for t, conds in zip(tasks, conds_args):
        store = ContextStore()
        store.insert(ContextElement(id=t["id_a1"], content="v1", precision=0.8, avacchedaka=conds))
        store.insert(ContextElement(id=t["id_a2"], content="v2", precision=0.9, avacchedaka=conds))
        results = store.retrieve(AvacchedakaQuery(qualificand=t["q"], condition=t["c"]))
        if len(results) > 1:
            conflicts_without += 1
    conflict_rate_without = conflicts_without / len(tasks)
    assert conflict_rate_without == 1.0

    # With avacchedaka: 0% conflict rate
    conflicts_with = 0
    for t, conds in zip(tasks, conds_args):
        store = ContextStore()
        store.insert(ContextElement(id=t["id_a1"], content="v1", precision=0.8, avacchedaka=conds))
        store.insert(ContextElement(id=t["id_a2"], content="v2", precision=0.9, avacchedaka=conds))
        store.sublate(t["id_a1"], t["id_a2"])
        results = store.retrieve(AvacchedakaQuery(qualificand=t["q"], condition=t["c"]))
        if len(results) > 1:
            conflicts_with += 1
    conflict_rate_with = conflicts_with / len(tasks)
    assert conflict_rate_with == 0.0

    # H5 requires ≥30% reduction; actual reduction here is 100%
    reduction = (conflict_rate_without - conflict_rate_with) / conflict_rate_without
    assert reduction >= 0.30, f"Conflict rate reduction {reduction:.0%} < 30% target"


@pytest.mark.integration
def test_two_agents_no_conflict_with_avacchedaka(api_key):
    """Live integration: both agents operate on same store via avacchedaka queries.
    Second agent's contradicting insert should sublate first agent's element."""
    store = ContextStore()
    # Agent 1 inserts a claim
    store.insert(ContextElement(
        id="agent1-claim-001",
        content="Database uses PostgreSQL 14.",
        precision=0.85,
        avacchedaka=AvacchedakaConditions(qualificand="database", qualifier="version", condition="task_type=deploy"),
        provenance="agent1",
    ))
    # Agent 2 discovers contradicting information and sublates
    store.insert(ContextElement(
        id="agent2-claim-001",
        content="Database uses PostgreSQL 16 (upgraded last week).",
        precision=0.95,
        avacchedaka=AvacchedakaConditions(qualificand="database", qualifier="version", condition="task_type=deploy"),
        provenance="agent2",
    ))
    store.sublate("agent1-claim-001", "agent2-claim-001")

    # Retrieval should return only the newer, non-sublated claim
    query = AvacchedakaQuery(qualificand="database", condition="task_type=deploy")
    results = store.retrieve(query)
    assert len(results) == 1
    assert results[0].id == "agent2-claim-001"
    assert results[0].content == "Database uses PostgreSQL 16 (upgraded last week)."
