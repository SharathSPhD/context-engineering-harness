import pytest
from unittest.mock import MagicMock, patch
from src.agents.orchestrator import ManusBuddhiOrchestrator
from src.agents.manas import ManasOutput
from src.agents.buddhi import BuddhiOutput
from src.avacchedaka.store import ContextStore
from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.query import AvacchedakaQuery


@pytest.fixture
def store_with_auth_doc():
    store = ContextStore()
    store.insert(ContextElement(
        id="auth-001",
        content="JWT tokens expire after 24 hours.",
        precision=0.9,
        avacchedaka=AvacchedakaConditions(
            qualificand="auth", qualifier="token_expiry", condition="task_type=qa"
        ),
    ))
    return store


def test_orchestrator_run_calls_manas_then_buddhi(store_with_auth_doc):
    """Unit test: verifies orchestrator wires manas → buddhi correctly without API calls."""
    mock_manas_output = ManasOutput(
        candidate_ids=[],
        uncertainty=0.4,
        recommended_queries=[AvacchedakaQuery(qualificand="auth", condition="task_type=qa")],
        reasoning_sketch="JWT tokens likely expire in 24 hours.",
    )
    mock_buddhi_output = BuddhiOutput(answer="24 hours", confidence=0.9)

    with patch("src.agents.orchestrator.ManasAgent") as MockManas, \
         patch("src.agents.orchestrator.BuddhiAgent") as MockBuddhi:
        MockManas.return_value.run.return_value = mock_manas_output
        MockBuddhi.return_value.run.return_value = mock_buddhi_output

        orch = ManusBuddhiOrchestrator(api_key="fake-key", store=store_with_auth_doc)
        result = orch.run(
            question="How long do JWT tokens last?",
            task_context="task_type=qa",
            qualificand="auth",
        )

    assert result.answer == "24 hours"
    assert result.confidence == 0.9
    MockManas.return_value.run.assert_called_once()
    MockBuddhi.return_value.run.assert_called_once()


def test_orchestrator_passes_enriched_context_to_buddhi(store_with_auth_doc):
    """Context from store is included in buddhi's input."""
    mock_manas_output = ManasOutput(
        candidate_ids=[],
        uncertainty=0.5,
        recommended_queries=[],
        reasoning_sketch="Sketch.",
    )
    mock_buddhi_output = BuddhiOutput(answer=None, confidence=0.2)

    with patch("src.agents.orchestrator.ManasAgent") as MockManas, \
         patch("src.agents.orchestrator.BuddhiAgent") as MockBuddhi:
        MockManas.return_value.run.return_value = mock_manas_output
        MockBuddhi.return_value.run.return_value = mock_buddhi_output

        orch = ManusBuddhiOrchestrator(api_key="fake-key", store=store_with_auth_doc)
        orch.run(question="Q?", task_context="task_type=qa", qualificand="auth")

        buddhi_call_kwargs = MockBuddhi.return_value.run.call_args
        context_window = buddhi_call_kwargs[1].get("context_window") or buddhi_call_kwargs[0][1]
        assert "JWT tokens" in context_window


@pytest.mark.integration
def test_orchestrator_live_api(api_key, store_with_auth_doc):
    """Integration test: real API call. Requires ANTHROPIC_API_KEY."""
    orch = ManusBuddhiOrchestrator(api_key=api_key, store=store_with_auth_doc)
    result = orch.run(
        question="How long do JWT tokens last?",
        task_context="task_type=qa",
        qualificand="auth",
    )
    assert result.answer is not None or result.confidence < 0.6
    assert 0.0 <= result.confidence <= 1.0
