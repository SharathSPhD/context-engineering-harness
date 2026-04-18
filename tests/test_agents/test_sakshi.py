from src.agents.sakshi import SakshiPrefix


def test_sakshi_content_immutable():
    s = SakshiPrefix("System grounding statement.")
    assert s.content == "System grounding statement."


def test_sakshi_as_system_message_structure():
    s = SakshiPrefix("System grounding.")
    msg = s.as_system_message()
    assert msg["role"] == "system"
    assert "sakshi_prefix" in msg["content"]
    assert "System grounding." in msg["content"]


def test_sakshi_content_returns_init_value():
    s = SakshiPrefix("Original content.")
    assert s.content == "Original content."


def test_sakshi_as_system_message_role_is_system():
    s = SakshiPrefix("Grounding.")
    msg = s.as_system_message()
    assert msg["role"] == "system"
    assert "Grounding." in msg["content"]
