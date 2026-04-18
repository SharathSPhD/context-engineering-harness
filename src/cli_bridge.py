"""Claude CLI bridge — drop-in replacement for anthropic.Anthropic.

Routes LLM calls through `claude -p "..." --output-format json` subprocess,
using Claude Code subscription auth instead of a raw API key.

Usage:
    client = ClaudeCLIClient()           # no api_key needed
    # or via factory (returns Anthropic SDK client if api_key provided):
    client = get_client(api_key="")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hello"}],
    )
    text = response.content[0].text
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class _Content:
    text: str


@dataclass
class _Response:
    content: list[_Content]


class _MessagesNamespace:
    def __init__(self, client: "ClaudeCLIClient") -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: str = "",
    ) -> _Response:
        return self._client._call(
            model=model, max_tokens=max_tokens, messages=messages, system=system
        )


class ClaudeCLIClient:
    """Drop-in replacement for anthropic.Anthropic that uses the claude CLI.

    The api_key parameter is accepted but ignored — authentication is handled
    by the claude CLI's existing session (Claude Code subscription).
    """

    def __init__(self, api_key: str = "") -> None:  # noqa: ARG002
        self.messages = _MessagesNamespace(self)

    def _call(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: str = "",
    ) -> _Response:
        # Collapse conversation to a single user-turn string
        user_content = "\n".join(
            m["content"] for m in messages if m["role"] == "user"
        )

        cmd = [
            "claude",
            "-p", user_content,
            "--output-format", "json",
            "--model", model,
        ]
        if system:
            cmd += ["--system-prompt", system]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(f"claude CLI error: {result.stderr.strip()}")

        try:
            data = json.loads(result.stdout)
            text = data.get("result", result.stdout.strip())
        except json.JSONDecodeError:
            text = result.stdout.strip()

        return _Response(content=[_Content(text=text)])


def get_client(api_key: str = "") -> "ClaudeCLIClient | anthropic.Anthropic":
    """Factory: returns ClaudeCLIClient when api_key is empty, Anthropic SDK otherwise."""
    if api_key:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    return ClaudeCLIClient()
