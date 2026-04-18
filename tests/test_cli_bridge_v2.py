"""v2 CLI bridge — preserves role-typed turns; forwards seed; trims to max_tokens."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.cli_bridge import ClaudeCLIClient


class _Result:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _ok_stdout(text: str = "OK") -> str:
    return json.dumps({"result": text, "usage": {"input_tokens": 10, "output_tokens": 3}})


def test_cli_bridge_emits_stream_json_with_role_turns():
    captured: dict = {}

    def fake_run(cmd, *, input, capture_output, text, timeout, check):  # noqa: ARG001
        captured["cmd"] = cmd
        captured["stdin"] = input
        return _Result(0, _ok_stdout("hi"))

    with patch("src.cli_bridge.shutil.which", return_value="/usr/bin/claude"), \
         patch("src.cli_bridge.subprocess.run", side_effect=fake_run):
        client = ClaudeCLIClient()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=128,
            system="be helpful",
            messages=[
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
            ],
        )

    assert resp.content[0].text == "hi"
    cmd = captured["cmd"]
    assert cmd[cmd.index("--input-format") + 1] == "stream-json"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert cmd[cmd.index("--system-prompt") + 1] == "be helpful"

    lines = [json.loads(l) for l in captured["stdin"].strip().splitlines()]
    assert [l["type"] for l in lines] == ["user", "assistant", "user"]
    assert [l["message"]["role"] for l in lines] == ["user", "assistant", "user"]
    assert [l["message"]["content"] for l in lines] == ["Q1", "A1", "Q2"]


def test_cli_bridge_forwards_seed_when_provided():
    captured: dict = {}

    def fake_run(cmd, *, input, capture_output, text, timeout, check):  # noqa: ARG001
        captured["cmd"] = cmd
        return _Result(0, _ok_stdout())

    with patch("src.cli_bridge.shutil.which", return_value="/usr/bin/claude"), \
         patch("src.cli_bridge.subprocess.run", side_effect=fake_run):
        client = ClaudeCLIClient()
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=64,
            messages=[{"role": "user", "content": "x"}],
            seed=42,
        )

    assert "--seed" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--seed") + 1] == "42"


def test_cli_bridge_trims_to_max_tokens_upper_bound():
    huge = "X" * 10_000

    def fake_run(cmd, *, input, capture_output, text, timeout, check):  # noqa: ARG001
        return _Result(0, _ok_stdout(huge))

    with patch("src.cli_bridge.shutil.which", return_value="/usr/bin/claude"), \
         patch("src.cli_bridge.subprocess.run", side_effect=fake_run):
        client = ClaudeCLIClient()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "x"}],
        )
    # Trim cap is max_tokens * 6 chars (cushion above a strict 4-chars/token).
    assert len(resp.content[0].text) <= 100 * 6


def test_cli_bridge_rejects_unknown_role():
    with patch("src.cli_bridge.shutil.which", return_value="/usr/bin/claude"):
        client = ClaudeCLIClient()
        with pytest.raises(ValueError, match="unsupported role"):
            client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=64,
                messages=[{"role": "system", "content": "no"}],
            )


def test_cli_bridge_raises_when_cli_missing():
    with patch("src.cli_bridge.shutil.which", return_value=None):
        client = ClaudeCLIClient()
        with pytest.raises(RuntimeError, match="claude CLI not found"):
            client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=64,
                messages=[{"role": "user", "content": "hi"}],
            )
