"""Unit tests for tools.dev.scheduler.

These tests inject a `cli_runner` so we never actually shell out to claude.
Coverage targets:
  - cost ledger schema, window summary, exhaustion check
  - disk-cache atomic write + replay
  - rate-limit detector across stderr / stdout / json shapes
  - scheduler: cache hit replay, retry-on-rate-limit, sleep-to-next-window,
    exponential backoff on transient errors, prompt_hash stability.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.dev.scheduler import (
    CLIBudgetScheduler,
    CostLedger,
    DiskCache,
    PromptCache,
    RateLimitDetector,
    SchedulerConfig,
    SchedulerError,
    prompt_hash_of,
)
from tools.dev.scheduler.cache import CacheEntry
from tools.dev.scheduler.cost_ledger import next_window_start


# ---------- helpers ---------------------------------------------------------


def _ok_payload(text: str = "hi", in_tok: int = 5, out_tok: int = 7) -> str:
    return json.dumps(
        {"type": "result", "result": text, "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}
    )


def _make_runner(*responses: tuple[int, str, str]):
    """Returns a runner whose successive calls produce the given (exit, stdout, stderr) tuples."""
    seq = list(responses)
    calls: list[tuple[list[str], str]] = []

    def runner(cmd: list[str], stdin: str, timeout_s: int) -> tuple[int, str, str]:
        calls.append((cmd, stdin))
        if not seq:
            raise AssertionError("runner exhausted")
        return seq.pop(0)

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def _sched(tmp_path: Path, runner, *, dry_run: bool = False, **cfg_kw):
    cfg = SchedulerConfig(
        cache_root=tmp_path / "llm",
        ledger_path=tmp_path / "ledger.db",
        journal_path=tmp_path / "journal.jsonl",
        write_journal=False,
        dry_run=dry_run,
        base_backoff_s=0.0,
        max_backoff_s=0.0,
        **cfg_kw,
    )
    sleeps: list[float] = []
    fixed_now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)

    def clock():
        return fixed_now

    return CLIBudgetScheduler(
        cfg,
        cli_runner=runner,
        sleep=lambda s: sleeps.append(s),
        clock=clock,
    ), sleeps


# ---------- cost ledger -----------------------------------------------------


def test_cost_ledger_records_and_summarizes(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.db")
    ledger.record(prompt_hash="a", model="m", input_tokens=100, output_tokens=20)
    ledger.record(prompt_hash="a", model="m", cache_hit=True)
    ledger.record(prompt_hash="b", model="m", regime="HALT", exit_code=1)

    s = ledger.window_summary(window_hours=5)
    assert s.n_calls == 3
    assert s.n_cache_hits == 1
    assert s.input_tokens == 100
    assert s.n_rate_limited == 1


def test_cost_ledger_window_exhaustion(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.db")
    for _ in range(3):
        ledger.record(prompt_hash="x", model="m", input_tokens=400)
    assert ledger.is_window_exhausted(max_input_tokens=1500) is False
    ledger.record(prompt_hash="x", model="m", input_tokens=400)
    assert ledger.is_window_exhausted(max_input_tokens=1500) is True


def test_next_window_start_aligns_to_5h_grid():
    now = datetime(2026, 4, 18, 7, 30, tzinfo=timezone.utc)
    nxt = next_window_start(now, window_hours=5.0)
    # 5h windows from UTC midnight: 0, 5, 10, 15, 20. 7:30 -> next is 10:00.
    assert nxt == datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)


# ---------- disk cache ------------------------------------------------------


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(tmp_path / "llm")
    key = "a" * 64
    entry = CacheEntry(text="hello", input_tokens=3, output_tokens=4, model="m", raw={})
    assert cache.get(key) is None
    cache.put(key, entry)
    got = cache.get(key)
    assert got is not None and got.text == "hello"
    assert got.input_tokens == 3 and got.output_tokens == 4


def test_prompt_hash_stable_across_dict_orders():
    a = prompt_hash_of(model="m", system="s", messages=[{"role": "user", "content": "x"}], max_tokens=10)
    b = prompt_hash_of(messages=[{"role": "user", "content": "x"}], system="s", model="m", max_tokens=10)
    assert a == b


def test_prompt_cache_emits_cache_control():
    block = PromptCache.cacheable_block("static prefix")
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert block["text"] == "static prefix"


# ---------- rate-limit detector --------------------------------------------


@pytest.mark.parametrize(
    "stderr,stdout,exit_code,expected",
    [
        ("", _ok_payload(), 0, False),
        ("Error: rate_limit_error from upstream\n", "", 1, True),
        ("", '{"is_error": true, "subtype": "usage_limit"}', 1, True),
        ("HTTP 429 Too Many Requests", "", 1, True),
        ("Some other error", "", 1, False),
        ("", '{"is_error": true, "subtype": "weekly_limit_reached"}', 1, True),
    ],
)
def test_rate_limit_detector(stderr, stdout, exit_code, expected):
    state = RateLimitDetector().detect(stderr=stderr, stdout=stdout, exit_code=exit_code)
    assert state.is_rate_limited is expected


# ---------- scheduler integration ------------------------------------------


def test_submit_records_success_and_caches(tmp_path):
    runner = _make_runner((0, _ok_payload("answer", 10, 4), ""))
    sched, sleeps = _sched(tmp_path, runner)

    res = sched.submit(
        model="claude-haiku-4-5",
        system="be brief",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
    )
    assert res.text == "answer"
    assert res.cache_hit is False
    assert res.attempts == 1
    assert res.regime == "OK"
    assert res.input_tokens == 10 and res.output_tokens == 4
    assert sleeps == []  # no sleeps on a clean run

    # Replaying the same call hits cache, makes no CLI invocation.
    res2 = sched.submit(
        model="claude-haiku-4-5",
        system="be brief",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
    )
    assert res2.cache_hit is True
    assert len(runner.calls) == 1  # type: ignore[attr-defined]


def test_submit_retries_on_rate_limit_and_sleeps_to_next_window(tmp_path):
    runner = _make_runner(
        (1, "", "Error: rate_limit_error from upstream\n"),
        (0, _ok_payload("recovered", 5, 6), ""),
    )
    sched, sleeps = _sched(tmp_path, runner, max_retries=3)

    res = sched.submit(
        model="claude-sonnet-4-6",
        system="",
        messages=[{"role": "user", "content": "go"}],
        max_tokens=64,
    )
    assert res.text == "recovered"
    assert res.attempts == 2
    assert res.regime == "CYCLING"
    assert any(s > 0 for s in sleeps), "expected at least one window-aligned sleep"


def test_submit_raises_after_exhausting_retries(tmp_path):
    runner = _make_runner(*[(1, "", "transient flake\n") for _ in range(5)])
    sched, _ = _sched(tmp_path, runner, max_retries=3)

    with pytest.raises(SchedulerError):
        sched.submit(
            model="claude-haiku-4-5",
            system="",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_dry_run_returns_stub_without_invoking_cli(tmp_path):
    runner = _make_runner()  # no responses; would raise if called
    sched, _ = _sched(tmp_path, runner, dry_run=True)

    res = sched.submit(
        model="claude-haiku-4-5",
        system="",
        messages=[{"role": "user", "content": "anything"}],
    )
    assert res.text == "[dry-run scheduler stub]"
    assert res.cache_hit is False
    assert len(runner.calls) == 0  # type: ignore[attr-defined]


def test_pre_call_window_cap_triggers_sleep(tmp_path):
    runner = _make_runner((0, _ok_payload("ok"), ""))
    sched, sleeps = _sched(
        tmp_path,
        runner,
        max_calls_per_window=1,  # any prior non-cached call exhausts
    )
    # Pre-populate the ledger to simulate a full window already used.
    sched.ledger.record(prompt_hash="prev", model="m", input_tokens=100)

    sched.submit(
        model="claude-haiku-4-5",
        system="",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert any(s > 0 for s in sleeps), "expected sleep before the call"


def test_status_summarizes_window(tmp_path):
    runner = _make_runner((0, _ok_payload("ok", 10, 5), ""))
    sched, _ = _sched(tmp_path, runner)
    sched.submit(
        model="claude-haiku-4-5",
        system="",
        messages=[{"role": "user", "content": "hi"}],
    )
    s = sched.status()
    assert s["n_calls"] == 1
    assert s["input_tokens"] == 10
    assert s["window_hours"] == 5.0
    assert s["exhausted"] is False
