import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_run(stdout_dict: dict, returncode: int = 0):
    mock = MagicMock()
    mock.returncode = returncode
    mock.stdout = json.dumps(stdout_dict)
    mock.stderr = ""
    return mock


def test_bridge_returns_text_from_result_field():
    from src.cli_bridge import ClaudeCLIClient
    with patch("subprocess.run", return_value=_mock_run({"result": "Hello!", "type": "result"})):
        client = ClaudeCLIClient()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi"}],
        )
    assert resp.content[0].text == "Hello!"


def test_bridge_passes_system_prompt_as_flag():
    from src.cli_bridge import ClaudeCLIClient
    with patch("subprocess.run", return_value=_mock_run({"result": "ok"})) as mock_run:
        ClaudeCLIClient().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            system="You are a test assistant.",
            messages=[{"role": "user", "content": "Hi"}],
        )
    cmd = mock_run.call_args[0][0]
    assert "--system-prompt" in cmd
    assert "You are a test assistant." in cmd


def test_bridge_raises_on_nonzero_exit():
    from src.cli_bridge import ClaudeCLIClient
    bad = MagicMock()
    bad.returncode = 1
    bad.stderr = "auth error"
    bad.stdout = ""
    with patch("subprocess.run", return_value=bad):
        with pytest.raises(RuntimeError, match="claude CLI error"):
            ClaudeCLIClient().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=50,
                messages=[{"role": "user", "content": "Hi"}],
            )


def test_bridge_falls_back_to_raw_stdout_on_non_json():
    from src.cli_bridge import ClaudeCLIClient
    m = MagicMock()
    m.returncode = 0
    m.stdout = "plain text response"
    m.stderr = ""
    with patch("subprocess.run", return_value=m):
        resp = ClaudeCLIClient().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi"}],
        )
    assert resp.content[0].text == "plain text response"


def test_get_client_returns_cli_bridge_when_no_key():
    from src.cli_bridge import get_client, ClaudeCLIClient
    client = get_client(api_key="")
    assert isinstance(client, ClaudeCLIClient)


def test_get_client_returns_anthropic_when_key_provided():
    from src.cli_bridge import get_client
    import anthropic
    client = get_client(api_key="sk-ant-test-key-xxx")
    assert isinstance(client, anthropic.Anthropic)
