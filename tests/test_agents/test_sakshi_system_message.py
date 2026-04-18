"""G9 fix — Sakshi invariant rides the system message, not the user-visible context."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.agents.buddhi import BuddhiAgent
from src.agents.manas import ManasAgent


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeContent(text)]


def _agent_with_capture(cls):
    agent = cls.__new__(cls)
    captured: dict = {}
    fake_client = MagicMock()

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse('{"answer": null, "confidence": 0.0, "uncertainty": 0.5}')

    fake_client.messages.create.side_effect = fake_create
    agent.client = fake_client
    agent.model = "test-model"
    agent.max_tokens = 256
    return agent, captured


def test_manas_passes_sakshi_as_system_message_not_user_context():
    agent, captured = _agent_with_capture(ManasAgent)
    agent.run(
        question="Q?",
        context_window="some retrieved context",
        task_context="task_type=qa",
        qualificand="auth",
        sakshi_invariant="WITHHOLD when evidence is weak.",
    )
    assert "<sakshi_prefix>" in captured["system"]
    assert "WITHHOLD when evidence is weak." in captured["system"]
    user_text = captured["messages"][0]["content"]
    assert "<sakshi_prefix>" not in user_text


def test_buddhi_passes_sakshi_as_system_message_not_user_context():
    agent, captured = _agent_with_capture(BuddhiAgent)
    agent.run(
        question="Q?",
        context_window="some retrieved context",
        manas_sketch="sketch",
        uncertainty=0.4,
        candidate_ids=["x"],
        sakshi_invariant="Never fabricate.",
    )
    assert "<sakshi_prefix>" in captured["system"]
    assert "Never fabricate." in captured["system"]
    user_text = captured["messages"][0]["content"]
    assert "<sakshi_prefix>" not in user_text
