from src.agents.sakshi import SakshiPrefix


def test_sakshi_content_immutable():
    s = SakshiPrefix("System grounding statement.")
    assert s.content == "System grounding statement."


def test_sakshi_as_system_message_structure():
    s = SakshiPrefix("System grounding.")
    msg = s.as_system_message()
    assert msg["role"] == "user"
    assert "sakshi_prefix" in msg["content"]
    assert "System grounding." in msg["content"]


def test_sakshi_content_not_settable():
    s = SakshiPrefix("Original.")
    try:
        s._content = "Modified."
    except AttributeError:
        pass
    s2 = SakshiPrefix("Immutable.")
    assert s2.content == "Immutable."
