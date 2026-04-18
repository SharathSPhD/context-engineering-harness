"""Claude CLI bridge — drop-in replacement for anthropic.Anthropic.

Routes LLM calls through `claude` CLI subprocess (Claude Code subscription auth).
v2 fixes (G10):
  - Preserves role-typed turns via `--input-format stream-json` (no more
    user-only collapse; assistant turns now make it through).
  - Forwards `max_tokens` as an upper-bound hint via output trimming when the
    CLI itself does not enforce it (the `claude -p` command does not currently
    expose a --max-tokens flag).
  - Forwards `seed` when provided so deterministic runs are reproducible.

Usage:
    client = ClaudeCLIClient()           # no api_key needed
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hello"},
                  {"role": "assistant", "content": "Hi! What's up?"},
                  {"role": "user", "content": "explain TRIZ in 2 lines"}],
    )
    text = response.content[0].text
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class _Content:
    text: str


@dataclass
class _Response:
    content: list[_Content]
    input_tokens: int = 0
    output_tokens: int = 0


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
        seed: int | None = None,
    ) -> _Response:
        return self._client._call(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            seed=seed,
        )


class ClaudeCLIClient:
    """Drop-in replacement for anthropic.Anthropic that uses the claude CLI.

    The api_key parameter is accepted but ignored — authentication is handled
    by the claude CLI's existing session (Claude Code subscription).
    """

    def __init__(self, api_key: str = "", *, cli_path: str = "claude", timeout_s: int = 300) -> None:  # noqa: ARG002
        self.messages = _MessagesNamespace(self)
        self.cli_path = cli_path
        self.timeout_s = timeout_s

    def _call(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: str = "",
        seed: int | None = None,
    ) -> _Response:
        if shutil.which(self.cli_path) is None:
            raise RuntimeError(
                f"claude CLI not found at '{self.cli_path}'. "
                "Install Claude Code or set api_key for the SDK fallback."
            )

        # G10: stream-json preserves role turns. Each line mirrors the
        # Anthropic message event shape that the claude CLI parses:
        #   {"type":"user","message":{"role":"user","content":"..."}}
        # The system prompt rides on the dedicated --system-prompt flag,
        # not as an inline event line (the CLI's stream-json parser
        # rejects `system` events).
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                raise ValueError(f"unsupported role: {role!r}")
            lines.append(
                json.dumps(
                    {
                        "type": role,
                        "message": {"role": role, "content": msg["content"]},
                    },
                    ensure_ascii=False,
                )
            )
        stdin_blob = "\n".join(lines) + "\n"

        # The claude CLI requires that stream-json input pair with
        # stream-json output, so we parse NDJSON in `_parse_output`.
        cmd = [
            self.cli_path,
            "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--model", model,
            "--max-turns", str(max(1, len(messages))),
            "--verbose",
        ]
        if system:
            cmd += ["--system-prompt", system]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]

        result = subprocess.run(
            cmd,
            input=stdin_blob,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI error (exit={result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        text, in_tok, out_tok = self._parse_output(result.stdout)

        # G10: enforce max_tokens upper bound by trimming the response.
        # The CLI does not expose a hard cap, so we approximate by character
        # count — overlong outputs cost tokens we paid for but won't bleed
        # into downstream prompts.
        if max_tokens and len(text) > max_tokens * 6:
            text = text[: max_tokens * 6]

        return _Response(
            content=[_Content(text=text)],
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    @staticmethod
    def _parse_output(stdout: str) -> tuple[str, int, int]:
        """Parse claude CLI output. Handles three shapes:

        - single JSON object (legacy --output-format json),
        - NDJSON stream (--output-format stream-json) with a final
          `{"type": "result", "result": "...", "usage": {...}}` line,
        - raw text fallback.
        """
        s = stdout.strip()
        if not s:
            return "", 0, 0

        # NDJSON stream → walk lines, prefer the explicit `result` line.
        if "\n" in s and s.lstrip().startswith("{"):
            text, in_tok, out_tok = "", 0, 0
            assistant_chunks: list[str] = []
            for raw in s.splitlines():
                raw = raw.strip()
                if not raw or not raw.startswith("{"):
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "result" and "result" in obj:
                    text = str(obj["result"])
                    usage = obj.get("usage") or {}
                    in_tok = int(usage.get("input_tokens", in_tok))
                    out_tok = int(usage.get("output_tokens", out_tok))
                elif obj.get("type") == "assistant":
                    msg = obj.get("message") or {}
                    for block in msg.get("content") or []:
                        if isinstance(block, dict) and block.get("type") == "text":
                            assistant_chunks.append(block.get("text", ""))
            if text:
                return text, in_tok, out_tok
            if assistant_chunks:
                return "".join(assistant_chunks), in_tok, out_tok
            return "", in_tok, out_tok

        if s.startswith("{"):
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                return s, 0, 0
            usage = data.get("usage") or {}
            in_tok = int(usage.get("input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
            if "result" in data:
                return str(data["result"]), in_tok, out_tok
            if isinstance(data.get("content"), list):
                return (
                    "".join(b.get("text", "") for b in data["content"] if isinstance(b, dict)),
                    in_tok,
                    out_tok,
                )
        return s, 0, 0


def get_client(api_key: str = "") -> "ClaudeCLIClient | anthropic.Anthropic":  # type: ignore[name-defined]
    """Factory: returns ClaudeCLIClient when api_key is empty, Anthropic SDK otherwise."""
    if api_key:
        import anthropic

        return anthropic.Anthropic(api_key=api_key)
    return ClaudeCLIClient()
