"""Mocked tests for ManasAgent and BuddhiAgent to cover API-interaction code paths."""
import json
from unittest.mock import MagicMock, patch
from src.agents.manas import ManasAgent
from src.agents.buddhi import BuddhiAgent


def _fake_response(text: str):
    resp = MagicMock()
    block = MagicMock()
    block.text = text
    resp.content = [block]
    return resp


def test_manas_run_parses_valid_json():
    with patch("src.agents.manas.get_client") as mock_get_client:
        instance = MagicMock()
        mock_get_client.return_value = instance
        payload = {
            "candidate_summary": "possible answers",
            "uncertainty": 0.3,
            "recommended_queries": [
                {"qualificand": "auth", "condition": "task_type=qa"},
                {"qualificand": "session", "condition": "task_type=qa"},
            ],
            "reasoning_sketch": "Tokens may expire.",
        }
        instance.messages.create.return_value = _fake_response(json.dumps(payload))

        agent = ManasAgent(api_key="fake")
        out = agent.run(
            question="Q?",
            context_window="ctx",
            task_context="task_type=qa",
            qualificand="auth",
        )

    assert out.uncertainty == 0.3
    assert len(out.recommended_queries) == 2
    assert out.recommended_queries[0].qualificand == "auth"
    assert out.reasoning_sketch == "Tokens may expire."


def test_manas_run_handles_invalid_json():
    with patch("src.agents.manas.get_client") as mock_get_client:
        instance = MagicMock()
        mock_get_client.return_value = instance
        instance.messages.create.return_value = _fake_response("not-json")

        agent = ManasAgent(api_key="fake")
        out = agent.run(
            question="Q?",
            context_window="ctx",
            task_context="task_type=qa",
            qualificand="auth",
        )

    # Falls back to defaults
    assert out.uncertainty == 0.9
    assert len(out.recommended_queries) == 1
    assert out.recommended_queries[0].qualificand == "auth"
    assert out.reasoning_sketch == ""


def test_manas_run_empty_content_defaults():
    with patch("src.agents.manas.get_client") as mock_get_client:
        instance = MagicMock()
        mock_get_client.return_value = instance
        resp = MagicMock()
        resp.content = []
        instance.messages.create.return_value = resp

        agent = ManasAgent(api_key="fake")
        out = agent.run(
            question="Q?",
            context_window="ctx",
            task_context="task_type=qa",
            qualificand="auth",
        )

    assert out.uncertainty == 0.9


def test_buddhi_run_parses_valid_json():
    with patch("src.agents.buddhi.get_client") as mock_get_client:
        instance = MagicMock()
        mock_get_client.return_value = instance
        payload = {
            "answer": "24 hours",
            "confidence": 0.92,
            "reasoning_trace": "Derived from doc.",
            "sublated_candidates": ["12h"],
            "khyativada_flags": ["anyatha"],
        }
        instance.messages.create.return_value = _fake_response(json.dumps(payload))

        agent = BuddhiAgent(api_key="fake")
        out = agent.run(
            question="Q?",
            context_window="ctx",
            manas_sketch="sketch",
            uncertainty=0.4,
        )

    assert out.answer == "24 hours"
    assert out.confidence == 0.92
    assert out.sublated == ["12h"]
    assert out.reasoning_trace == "Derived from doc."
    assert out.khyativada_flags == ["anyatha"]


def test_buddhi_run_handles_invalid_json():
    with patch("src.agents.buddhi.get_client") as mock_get_client:
        instance = MagicMock()
        mock_get_client.return_value = instance
        instance.messages.create.return_value = _fake_response("not-json")

        agent = BuddhiAgent(api_key="fake")
        out = agent.run(
            question="Q?",
            context_window="ctx",
            manas_sketch="sketch",
            uncertainty=0.5,
        )

    assert out.answer is None
    assert out.confidence == 0.0
    assert out.reasoning_trace == "parse error"
