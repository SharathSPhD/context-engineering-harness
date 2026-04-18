import os
import pytest


@pytest.fixture
def api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def sample_element():
    from src.avacchedaka.element import ContextElement, AvacchedakaConditions
    return ContextElement(
        id="test-001",
        content="The auth module uses JWT tokens with 24h expiry.",
        precision=0.9,
        avacchedaka=AvacchedakaConditions(
            qualificand="auth_module",
            qualifier="token_expiry",
            condition="task_type=code_review",
        ),
        provenance="retrieved_doc",
    )
