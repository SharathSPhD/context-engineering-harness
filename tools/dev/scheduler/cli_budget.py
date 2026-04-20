"""Token-aware scheduler around the claude CLI.

Wraps a single CLI call with:
  * disk cache (free reruns)
  * cost-ledger logging (auditable spend)
  * rate-limit detection
  * exponential backoff
  * sleep-to-next-window on rate-limit
  * regime emission for AttractorFlow integration

Every call is durable: the cost ledger row is committed before the call
returns, so a crash mid-pipeline leaves a complete trail.

Public surface:
    sched = CLIBudgetScheduler(SchedulerConfig(...))
    result = sched.submit(model="claude-sonnet-4-6",
                          system="...",
                          messages=[{"role":"user", "content":"..."}],
                          max_tokens=512)
    result.text  # str
    result.cache_hit
    result.regime  # 'OK' | 'CYCLING' | 'HALT'
    result.attempts
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .cache import CacheEntry, DiskCache, prompt_hash_of
from .cost_ledger import CostLedger, next_window_start
from .rate_limit import RateLimitDetector, RateLimitState


logger = logging.getLogger(__name__)


class SchedulerError(RuntimeError):
    """Raised when the scheduler exhausts retries or hits a non-retryable error."""


@dataclass(frozen=True)
class SchedulerConfig:
    cache_root: str | Path = ".cache/llm"
    ledger_path: str | Path = ".cache/cost_ledger.db"
    journal_path: str | Path = "tools/dev/orchestration/attractor_journal.jsonl"
    cli_path: str = "claude"
    window_hours: float = 5.0
    # Soft input-token cap per window. None = uncapped.
    max_input_tokens_per_window: int | None = 2_000_000
    # Soft non-cached call cap per window.
    max_calls_per_window: int | None = None
    max_retries: int = 5
    base_backoff_s: float = 10.0
    max_backoff_s: float = 600.0
    # If True, the scheduler returns a deterministic mock instead of calling the CLI.
    # Useful for unit tests; never enabled in production runs.
    dry_run: bool = False
    dry_run_text: str = "[dry-run scheduler stub]"
    # When True, persists the AttractorFlow journal entry on every call.
    write_journal: bool = True
    timeout_s: int = 300


@dataclass(frozen=True)
class SubmitResult:
    text: str
    cache_hit: bool
    attempts: int
    regime: str  # OK | CYCLING | HALT
    input_tokens: int
    output_tokens: int
    latency_ms: int
    prompt_hash: str
    raw: dict[str, Any] = field(default_factory=dict)


# Test-injectable CLI runner: receives (cmd, stdin, timeout_s), returns (exit, stdout, stderr).
CLIRunner = Callable[
    [list[str], str, int],
    tuple[int, str, str],
]


def _default_cli_runner(cmd: list[str], stdin: str, timeout_s: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        input=stdin if stdin else None,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


class CLIBudgetScheduler:
    """Synchronous, single-call scheduler.

    Multi-job coordination is the caller's responsibility; this class is a
    primitive ("submit one call, get one result"). The reason: keeping it
    synchronous means the entire failure model (caching, retries, sleeps,
    journal writes) is local to one method and easy to reason about.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        cli_runner: CLIRunner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.config = config or SchedulerConfig()
        self.cli_runner = cli_runner or _default_cli_runner
        self._sleep = sleep
        self._clock = clock
        self.cache = DiskCache(self.config.cache_root)
        self.ledger = CostLedger(self.config.ledger_path)
        self.detector = RateLimitDetector()

    # ----------------------------------------------------------------- public

    def submit(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        bypass_cache: bool = False,
    ) -> SubmitResult:
        """Execute one CLI call with caching, rate-limit handling, and ledger logging."""
        ph = prompt_hash_of(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )

        if not bypass_cache:
            entry = self.cache.get(ph)
            if entry is not None:
                self.ledger.record(
                    prompt_hash=ph,
                    model=model,
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    cache_hit=True,
                    latency_ms=0,
                    exit_code=0,
                    regime="OK",
                    attempt=0,
                    note="disk-cache hit",
                )
                return SubmitResult(
                    text=entry.text,
                    cache_hit=True,
                    attempts=0,
                    regime="OK",
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    latency_ms=0,
                    prompt_hash=ph,
                    raw=entry.raw,
                )

        # Pre-call window check.
        if self._window_exhausted():
            self._sleep_until_next_window(reason="pre-call window cap reached")

        last_err: str = ""
        backoff = self.config.base_backoff_s
        for attempt in range(1, self.config.max_retries + 1):
            t0 = time.monotonic()
            try:
                exit_code, stdout, stderr = self._invoke(
                    model=model,
                    system=system,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                )
            except subprocess.TimeoutExpired as exc:
                exit_code, stdout, stderr = 124, "", f"timeout: {exc}"
            latency_ms = int((time.monotonic() - t0) * 1000)

            rl: RateLimitState = self.detector.detect(
                stderr=stderr, stdout=stdout, exit_code=exit_code
            )

            if rl.is_rate_limited:
                self.ledger.record(
                    prompt_hash=ph,
                    model=model,
                    cache_hit=False,
                    latency_ms=latency_ms,
                    exit_code=exit_code if exit_code != 0 else 1,
                    regime="HALT",
                    attempt=attempt,
                    note=f"rate-limit: {rl.matched_pattern} in {rl.matched_in}",
                )
                self._emit_journal(
                    regime="HALT",
                    note=f"attempt {attempt}: rate-limit {rl.matched_pattern}",
                )
                self._sleep_until_next_window(
                    reason=f"rate-limit detected (attempt {attempt})"
                )
                last_err = f"rate-limit: {rl.raw_excerpt}"
                continue

            if exit_code == 0:
                text, in_tok, out_tok, raw = self._parse_cli_output(stdout)
                entry = CacheEntry(
                    text=text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    model=model,
                    raw=raw,
                )
                self.cache.put(ph, entry)
                self.ledger.record(
                    prompt_hash=ph,
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cache_hit=False,
                    latency_ms=latency_ms,
                    exit_code=0,
                    regime="OK" if attempt == 1 else "CYCLING",
                    attempt=attempt,
                )
                if attempt > 1:
                    self._emit_journal(
                        regime="OK", note=f"recovered on attempt {attempt}"
                    )
                return SubmitResult(
                    text=text,
                    cache_hit=False,
                    attempts=attempt,
                    regime="OK" if attempt == 1 else "CYCLING",
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=latency_ms,
                    prompt_hash=ph,
                    raw=raw,
                )

            # Non-rate-limit error: log, exponential backoff, retry.
            self.ledger.record(
                prompt_hash=ph,
                model=model,
                cache_hit=False,
                latency_ms=latency_ms,
                exit_code=exit_code,
                regime="CYCLING",
                attempt=attempt,
                note=(stderr[:240] or "non-zero exit"),
            )
            last_err = stderr[:280] or stdout[:280]
            self._emit_journal(
                regime="CYCLING",
                note=f"attempt {attempt} exit={exit_code}",
            )
            self._sleep(min(backoff, self.config.max_backoff_s))
            backoff *= 2

        raise SchedulerError(
            f"Exhausted {self.config.max_retries} retries for prompt {ph[:12]}; "
            f"last error: {last_err}"
        )

    # --- introspection --------------------------------------------------

    def status(self) -> dict[str, Any]:
        s = self.ledger.window_summary(window_hours=self.config.window_hours)
        return {
            "window_hours": self.config.window_hours,
            "n_calls": s.n_calls,
            "n_cache_hits": s.n_cache_hits,
            "cache_hit_rate": round(s.cache_hit_rate, 3),
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "n_rate_limited": s.n_rate_limited,
            "max_input_tokens_per_window": self.config.max_input_tokens_per_window,
            "max_calls_per_window": self.config.max_calls_per_window,
            "exhausted": self._window_exhausted(),
            "next_window_at": next_window_start(
                self._clock(), window_hours=self.config.window_hours
            ).isoformat(),
            "total": self.ledger.total(),
        }

    # --- internals ------------------------------------------------------

    def _window_exhausted(self) -> bool:
        return self.ledger.is_window_exhausted(
            window_hours=self.config.window_hours,
            max_input_tokens=self.config.max_input_tokens_per_window,
            max_calls=self.config.max_calls_per_window,
        )

    def _sleep_until_next_window(self, *, reason: str) -> None:
        target = next_window_start(self._clock(), window_hours=self.config.window_hours)
        now = self._clock()
        wait_s = max(0.0, (target - now).total_seconds())
        # Cap absurd sleeps in tests; in production we'd actually wait.
        wait_s = min(wait_s, self.config.window_hours * 3600.0)
        logger.warning(
            "scheduler sleeping %.0fs until %s (%s)",
            wait_s,
            target.isoformat(),
            reason,
        )
        self._emit_journal(regime="HALT", note=f"sleep_until={target.isoformat()} reason={reason}")
        self._sleep(wait_s)

    def _invoke(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> tuple[int, str, str]:
        if self.config.dry_run:
            payload = {
                "type": "result",
                "result": self.config.dry_run_text,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
            return 0, json.dumps(payload), ""

        cli = self.config.cli_path
        # Only check PATH when using the default runner — tests inject their own.
        if (
            self.cli_runner is _default_cli_runner
            and shutil.which(cli) is None
        ):
            return 127, "", f"claude CLI not found at '{cli}' on PATH"

        # stream-json stdin: one event per line. The CLI schema expects
        # {"type": "<role>", "message": {"role": "<role>", "content": "..."}}.
        # `system` is NOT an inline event — it rides on --system-prompt.
        lines: list[str] = []
        for msg in messages:
            role = msg["role"]
            if role not in ("user", "assistant"):
                continue
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

        # The current claude CLI (>=2.1) requires stream-json pairing AND
        # --verbose when output is stream-json. It also dropped --seed.
        cmd = [
            cli,
            "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", model,
            "--max-turns", str(max(1, len(messages))),
        ]
        if system:
            cmd += ["--system-prompt", system]
        # Note: --seed was removed in claude CLI 2.1. Keep `seed` as a prompt-
        # hash component (already threaded via prompt_hash_of) so we still get
        # reproducible cache keys; deterministic inference is no longer
        # CLI-controllable. If a future CLI re-adds --seed we can restore it.
        _ = seed  # explicitly consumed — see comment above

        try:
            return self.cli_runner(cmd, stdin_blob, self.config.timeout_s)
        except subprocess.TimeoutExpired as exc:
            return 124, "", f"timeout after {self.config.timeout_s}s: {exc}"

    def _parse_cli_output(self, stdout: str) -> tuple[str, int, int, dict[str, Any]]:
        """Parse claude CLI output; supports both single-JSON and NDJSON stream.

        Modern (>=2.1) CLI emits NDJSON when --output-format=stream-json. We
        walk the lines, collect assistant text, and prefer the final
        ``{"type": "result", "result": "...", "usage": {...}}`` line for the
        canonical answer + token accounting. Legacy single-object JSON is
        still accepted for back-compat with injected test runners.
        """
        s = stdout.strip()
        if not s:
            return "", 0, 0, {}

        if "\n" in s and s.lstrip().startswith("{"):
            text = ""
            in_tok = 0
            out_tok = 0
            raw: dict[str, Any] = {}
            assistant_chunks: list[str] = []
            for raw_line in s.splitlines():
                line = raw_line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "result" and "result" in obj:
                    text = str(obj["result"])
                    raw = obj
                    in_tok, out_tok = self._usage_tokens(obj.get("usage") or {}, in_tok, out_tok)
                elif obj.get("type") == "assistant":
                    msg = obj.get("message") or {}
                    for block in msg.get("content") or []:
                        if isinstance(block, dict) and block.get("type") == "text":
                            assistant_chunks.append(block.get("text", ""))
            if text:
                return text, in_tok, out_tok, raw
            if assistant_chunks:
                return "".join(assistant_chunks), in_tok, out_tok, raw
            return "", in_tok, out_tok, raw

        text = s
        in_tok = 0
        out_tok = 0
        raw_obj: dict[str, Any] = {}
        if text.startswith("{"):
            try:
                data = json.loads(text)
                raw_obj = data
                if "result" in data:
                    text = str(data["result"])
                elif "content" in data and isinstance(data["content"], list):
                    text = "".join(
                        b.get("text", "") for b in data["content"] if isinstance(b, dict)
                    )
                in_tok, out_tok = self._usage_tokens(data.get("usage") or {}, 0, 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return text, in_tok, out_tok, raw_obj

    @staticmethod
    def _usage_tokens(usage: dict[str, Any], fallback_in: int, fallback_out: int) -> tuple[int, int]:
        """Sum all input-token buckets the CLI reports.

        Claude CLI's usage object splits input tokens across up to three
        buckets: ``input_tokens`` (newly transmitted non-cached),
        ``cache_creation_input_tokens`` (first-seen tokens now stored in the
        prompt cache, billed at a premium), and ``cache_read_input_tokens``
        (tokens served from the prompt cache, billed at a discount). For
        honest budget tracking and acceptance tests we need the sum.
        """
        def _as_int(v: Any) -> int:
            try:
                return int(v or 0)
            except (TypeError, ValueError):
                return 0

        in_plain = _as_int(usage.get("input_tokens", fallback_in))
        in_cache_create = _as_int(usage.get("cache_creation_input_tokens"))
        in_cache_read = _as_int(usage.get("cache_read_input_tokens"))
        out_plain = _as_int(usage.get("output_tokens", fallback_out))
        return in_plain + in_cache_create + in_cache_read, out_plain

    def _emit_journal(self, *, regime: str, note: str) -> None:
        if not self.config.write_journal:
            return
        path = Path(self.config.journal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": self._clock().isoformat(),
            "session_id": "context-harness-v2-rebuild",
            "regime": regime,
            "note": note,
            "ledger_window": self.ledger.window_summary(
                window_hours=self.config.window_hours
            ).__dict__,
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str))
            fh.write("\n")
