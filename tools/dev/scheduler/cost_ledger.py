"""Per-call sqlite cost ledger.

Every CLI invocation lands here so we have a durable, auditable trail of:
- which prompt hashes hit cache vs. fired the CLI
- total input/output tokens per model per window
- exit codes (0 = OK, non-zero = error / rate-limit)
- AttractorFlow regime at the time of the call (OK, CYCLING, HALT)

Schema is stable; new columns added only via ALTER TABLE in `_migrate`.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_unix         REAL    NOT NULL,
    prompt_hash     TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_hit       INTEGER NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    exit_code       INTEGER NOT NULL DEFAULT 0,
    regime          TEXT    NOT NULL DEFAULT 'OK',
    attempt         INTEGER NOT NULL DEFAULT 1,
    note            TEXT
);
CREATE INDEX IF NOT EXISTS calls_ts        ON calls(ts_unix);
CREATE INDEX IF NOT EXISTS calls_prompt    ON calls(prompt_hash);
CREATE INDEX IF NOT EXISTS calls_regime    ON calls(regime);
"""


@dataclass(frozen=True)
class WindowSummary:
    """Aggregated stats over a rolling window."""

    window_hours: float
    started_at_unix: float
    n_calls: int
    n_cache_hits: int
    input_tokens: int
    output_tokens: int
    n_rate_limited: int

    @property
    def cache_hit_rate(self) -> float:
        return (self.n_cache_hits / self.n_calls) if self.n_calls else 0.0


class CostLedger:
    """Thread-safe-enough sqlite ledger.

    Sqlite's default locking is fine for a single-process scheduler;
    multi-process callers should serialize writes.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(
        self,
        *,
        prompt_hash: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        latency_ms: int = 0,
        exit_code: int = 0,
        regime: str = "OK",
        attempt: int = 1,
        note: str | None = None,
        ts_unix: float | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO calls
               (ts_unix, prompt_hash, model, input_tokens, output_tokens,
                cache_hit, latency_ms, exit_code, regime, attempt, note)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts_unix if ts_unix is not None else time.time(),
                prompt_hash,
                model,
                int(input_tokens),
                int(output_tokens),
                1 if cache_hit else 0,
                int(latency_ms),
                int(exit_code),
                regime,
                int(attempt),
                note,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def window_summary(self, *, window_hours: float = 5.0) -> WindowSummary:
        cutoff = time.time() - window_hours * 3600.0
        row = self._conn.execute(
            """SELECT
                 COUNT(*)                              AS n_calls,
                 COALESCE(SUM(cache_hit), 0)           AS n_cache_hits,
                 COALESCE(SUM(input_tokens), 0)        AS input_tokens,
                 COALESCE(SUM(output_tokens), 0)       AS output_tokens,
                 SUM(CASE WHEN regime = 'HALT' THEN 1 ELSE 0 END)
                                                       AS n_rate_limited
               FROM calls
               WHERE ts_unix >= ?""",
            (cutoff,),
        ).fetchone()
        return WindowSummary(
            window_hours=window_hours,
            started_at_unix=cutoff,
            n_calls=int(row["n_calls"] or 0),
            n_cache_hits=int(row["n_cache_hits"] or 0),
            input_tokens=int(row["input_tokens"] or 0),
            output_tokens=int(row["output_tokens"] or 0),
            n_rate_limited=int(row["n_rate_limited"] or 0),
        )

    def is_window_exhausted(
        self,
        *,
        window_hours: float = 5.0,
        max_input_tokens: int | None = None,
        max_calls: int | None = None,
    ) -> bool:
        s = self.window_summary(window_hours=window_hours)
        if max_input_tokens is not None and s.input_tokens >= max_input_tokens:
            return True
        if max_calls is not None and s.n_calls - s.n_cache_hits >= max_calls:
            return True
        return False

    def last_rate_limit_at(self) -> float | None:
        row = self._conn.execute(
            "SELECT MAX(ts_unix) AS ts FROM calls WHERE regime = 'HALT'"
        ).fetchone()
        return float(row["ts"]) if row and row["ts"] is not None else None

    def calls_since(self, since_unix: float) -> Iterable[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM calls WHERE ts_unix >= ? ORDER BY ts_unix",
            (since_unix,),
        ).fetchall()

    def total(self) -> dict[str, int]:
        row = self._conn.execute(
            """SELECT
                 COUNT(*)                              AS n_calls,
                 COALESCE(SUM(input_tokens), 0)        AS input_tokens,
                 COALESCE(SUM(output_tokens), 0)       AS output_tokens,
                 COALESCE(SUM(cache_hit), 0)           AS n_cache_hits
               FROM calls"""
        ).fetchone()
        return {
            "n_calls": int(row["n_calls"] or 0),
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "n_cache_hits": int(row["n_cache_hits"] or 0),
        }

    def close(self) -> None:
        self._conn.close()


def next_window_start(
    now: datetime | None = None, *, window_hours: float = 5.0
) -> datetime:
    """Return the next clock-aligned 5h window boundary in UTC.

    We align to multiples of `window_hours` from UTC midnight so all callers
    agree on when a new window opens, regardless of when they were paused.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (now - midnight).total_seconds() / 3600.0
    next_step = (int(elapsed // window_hours) + 1) * window_hours
    return midnight + timedelta(hours=next_step)
