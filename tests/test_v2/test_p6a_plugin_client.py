"""Smoke tests for the in-process plugin client used by P6-A H3/H4/H5.

These tests confirm:
  * the plugin module imports cleanly from disk,
  * `PratyakshaPluginClient.reset()` truly clears state between trials,
  * all eight tool families surface the expected response shapes.

If these break, no H3/H4/H5 result can be trusted — they are the
foundation the runner is built on.
"""
from __future__ import annotations

import pytest

from experiments.v2.p6a.plugin_client import PratyakshaPluginClient


@pytest.fixture()
def client() -> PratyakshaPluginClient:
    return PratyakshaPluginClient()


def test_reset_clears_state(client: PratyakshaPluginClient) -> None:
    client.insert(
        id="r-1", content="x", precision=0.9,
        qualificand="q", qualifier="qual", condition="c=1",
    )
    assert client.state_size == 1
    client.reset()
    assert client.state_size == 0
    assert client.get_sakshi()["sakshi"] is None


def test_insert_and_retrieve_round_trip(client: PratyakshaPluginClient) -> None:
    out = client.insert(
        id="el-1", content="The cache TTL is 60s.", precision=0.92,
        qualificand="cache", qualifier="ttl", condition="case=A",
    )
    assert out["ok"]
    assert out["element"]["precision"] == pytest.approx(0.92, rel=0, abs=1e-4)

    res = client.retrieve(qualificand="cache", condition="case=A", precision_threshold=0.5)
    assert res["ok"]
    assert res["count"] == 1
    assert res["elements"][0]["id"] == "el-1"


def test_retrieve_respects_condition_and_threshold(client: PratyakshaPluginClient) -> None:
    client.insert(
        id="hi", content="HIGH", precision=0.9,
        qualificand="db", qualifier="v", condition="case=A",
    )
    client.insert(
        id="lo", content="LOW", precision=0.2,
        qualificand="db", qualifier="v", condition="case=A",
    )
    client.insert(
        id="other", content="OTHER", precision=0.9,
        qualificand="db", qualifier="v", condition="case=B",
    )

    res = client.retrieve(qualificand="db", condition="case=A", precision_threshold=0.5)
    ids = [e["id"] for e in res["elements"]]
    assert ids == ["hi"]


def test_sublate_with_evidence_atomically_replaces(client: PratyakshaPluginClient) -> None:
    client.insert(
        id="old", content="OLD VALUE", precision=0.6,
        qualificand="auth", qualifier="ttl", condition="case=X",
    )
    out = client.sublate_with_evidence(
        older_id="old",
        newer_content="NEW VALUE",
        newer_precision=0.95,
        qualificand="auth", qualifier="ttl", condition="case=X",
    )
    assert out["ok"]
    new_id = out["newer_id"]
    assert new_id != "old"

    res = client.retrieve(qualificand="auth", condition="case=X", precision_threshold=0.0)
    active_ids = [e["id"] for e in res["elements"]]
    assert active_ids == [new_id]
    assert res["elements"][0]["content"] == "NEW VALUE"

    older = client.get("old")
    assert older["ok"]
    assert older["element"]["sublated_by"] == new_id
    assert older["element"]["precision"] == pytest.approx(0.0)


def test_compact_drops_low_precision_only(client: PratyakshaPluginClient) -> None:
    client.insert(
        id="hi", content="x", precision=0.9,
        qualificand="z", qualifier="q", condition="c=1",
    )
    client.insert(
        id="lo", content="y", precision=0.2,
        qualificand="z", qualifier="q", condition="c=1",
    )
    out = client.compact(precision_threshold=0.5)
    assert out["ok"]
    assert "lo" in out["compressed_ids"]
    assert "hi" not in out["compressed_ids"]


def test_boundary_compact_returns_detection_payload(client: PratyakshaPluginClient) -> None:
    text = "lorem ipsum dolor sit amet " * 8 + "BREAKING NEW ENTITY ALERT XYZ123 " * 4
    out = client.boundary_compact(text_window=text, threshold_z=2.0, precision_threshold=0.30)
    assert out["ok"]
    assert "boundary_detected" in out


def test_set_and_get_sakshi(client: PratyakshaPluginClient) -> None:
    out = client.set_sakshi("Decline to answer when uncertain.")
    assert out["ok"]
    got = client.get_sakshi()
    assert got["sakshi"] == "Decline to answer when uncertain."


def test_classify_khyativada_returns_known_class(client: PratyakshaPluginClient) -> None:
    out = client.classify_khyativada(
        claim="The system uses Postgres 14.",
        ground_truth="The system actually uses Postgres 16.0",
    )
    assert out["ok"]
    assert "class" in out
    assert out["class"] in {
        "anyathakhyati", "atmakhyati", "anirvacaniyakhyati",
        "asatkhyati", "viparitakhyati", "akhyati", "none",
    }


def test_budget_record_then_status(client: PratyakshaPluginClient) -> None:
    client.budget_record(tokens=123, model="claude-haiku-4-5", note="test")
    s = client.budget_status(last_n=5)
    assert s["ok"]
    assert s["budget_used"] >= 123
    assert s["ledger_n_calls"] >= 1


# --- P9 critical-fix regressions: B1 idempotence, B2 ID collision, B3 qualifier ---

def test_sublate_with_evidence_is_idempotent(client: PratyakshaPluginClient) -> None:
    """B1: a second sublation of an already-sublated element must be a no-op
    rather than silently creating a second sublator (which would corrupt the
    audit trail and inflate state.elements)."""
    client.insert(
        id="old", content="OLD", precision=0.6,
        qualificand="auth", qualifier="ttl", condition="case=X",
    )
    first = client.sublate_with_evidence(
        older_id="old", newer_content="NEW1", newer_precision=0.95,
        qualificand="auth", qualifier="ttl", condition="case=X",
    )
    assert first["ok"] and not first.get("already_sublated")
    size_after_first = client.state_size

    second = client.sublate_with_evidence(
        older_id="old", newer_content="NEW2", newer_precision=0.99,
        qualificand="auth", qualifier="ttl", condition="case=X",
    )
    assert second["ok"]
    assert second.get("already_sublated") is True
    assert second["by"] == first["newer_id"]
    assert client.state_size == size_after_first


def test_sublate_with_evidence_no_id_collision_within_one_ms(
    client: PratyakshaPluginClient,
) -> None:
    """B2: rapid back-to-back sublations of *different* older elements within
    the same millisecond must produce distinct newer_ids — otherwise the
    second one would overwrite the first in STATE.elements."""
    seen: set[str] = set()
    for i in range(50):
        client.insert(
            id=f"old-{i}", content=f"v{i}", precision=0.5,
            qualificand="cfg", qualifier="param", condition=f"case={i}",
        )
        out = client.sublate_with_evidence(
            older_id=f"old-{i}", newer_content=f"v{i}+1", newer_precision=0.9,
            qualificand="cfg", qualifier="param", condition=f"case={i}",
        )
        assert out["ok"] and not out.get("already_sublated")
        assert out["newer_id"] not in seen
        seen.add(out["newer_id"])


def test_retrieve_respects_qualifier(client: PratyakshaPluginClient) -> None:
    """B3: two elements that share a (qualificand, condition) but differ in
    qualifier must not collide. Retrieval scoped to a specific qualifier
    must return only the matching element."""
    client.insert(
        id="ttl", content="TTL=60s", precision=0.9,
        qualificand="cache", qualifier="ttl", condition="case=A",
    )
    client.insert(
        id="size", content="SIZE=1MB", precision=0.9,
        qualificand="cache", qualifier="size", condition="case=A",
    )

    res_ttl = client._mod.context_retrieve(  # type: ignore[attr-defined]
        client._mod.RetrieveInput(  # type: ignore[attr-defined]
            qualificand="cache", qualifier="ttl",
            condition="case=A", precision_threshold=0.0, max_elements=20,
        )
    )
    assert res_ttl["ok"]
    assert [e["id"] for e in res_ttl["elements"]] == ["ttl"]

    res_size = client._mod.context_retrieve(  # type: ignore[attr-defined]
        client._mod.RetrieveInput(  # type: ignore[attr-defined]
            qualificand="cache", qualifier="size",
            condition="case=A", precision_threshold=0.0, max_elements=20,
        )
    )
    assert res_size["ok"]
    assert [e["id"] for e in res_size["elements"]] == ["size"]

    res_any = client._mod.context_retrieve(  # type: ignore[attr-defined]
        client._mod.RetrieveInput(  # type: ignore[attr-defined]
            qualificand="cache", qualifier="",
            condition="case=A", precision_threshold=0.0, max_elements=20,
        )
    )
    assert res_any["ok"]
    assert sorted(e["id"] for e in res_any["elements"]) == ["size", "ttl"]
