"""Rate-limit detection for the claude CLI.

The claude CLI surfaces rate-limit / usage-cap errors in three different ways
depending on the version:

1. Non-zero exit code (typically 1 or 2) with a recognizable marker in stderr.
2. JSON output on stdout with an `is_error` field and a `subtype` matching
   one of `usage_limit`, `rate_limit`, `quota_exceeded`.
3. Plain-text 429 banner.

We detect all three. On detection, callers should mark the current 5h window
as exhausted and sleep to the next clock-aligned boundary.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


# Conservative pattern set — match anywhere in the surfaced text, case-insensitive.
_PATTERNS = [
    re.compile(r"rate[\s_-]?limit", re.I),
    re.compile(r"usage[\s_-]?limit", re.I),
    re.compile(r"quota[\s_-]?exceeded", re.I),
    re.compile(r"too\s+many\s+requests", re.I),
    re.compile(r"\b429\b"),
    re.compile(r"limit[_\s-]?reached", re.I),
    re.compile(r"weekly[_\s-]?limit", re.I),
]


@dataclass(frozen=True)
class RateLimitState:
    is_rate_limited: bool
    matched_pattern: str | None = None
    matched_in: str | None = None  # 'stderr' | 'stdout' | 'json'
    raw_excerpt: str | None = None

    @classmethod
    def ok(cls) -> "RateLimitState":
        return cls(is_rate_limited=False)

    @classmethod
    def hit(cls, *, pattern: str, source: str, excerpt: str) -> "RateLimitState":
        return cls(
            is_rate_limited=True,
            matched_pattern=pattern,
            matched_in=source,
            raw_excerpt=excerpt[:280],
        )


class RateLimitDetector:
    """Inspects (stderr, stdout, exit_code) for rate-limit signals."""

    def detect(
        self, *, stderr: str, stdout: str, exit_code: int
    ) -> RateLimitState:
        # 1. JSON-structured error, against the FINAL result event.
        #    Supports both legacy single-JSON and modern NDJSON stream output
        #    (--output-format=stream-json). The final ``{"type":"result", ...}``
        #    line carries ``is_error`` and any API error subtype.
        result_obj = self._final_result_obj(stdout)
        if isinstance(result_obj, dict):
            subtype = str(
                (
                    result_obj.get("subtype")
                    or (
                        result_obj.get("error", {}).get("type", "")
                        if isinstance(result_obj.get("error"), dict)
                        else ""
                    )
                )
            ).lower()
            if any(s in subtype for s in ("rate_limit", "usage_limit", "quota_exceeded")):
                return RateLimitState.hit(
                    pattern=subtype, source="json", excerpt=json.dumps(result_obj)[:280]
                )
            if result_obj.get("is_error"):
                blob = json.dumps(result_obj)
                for pat in _PATTERNS:
                    m = pat.search(blob)
                    if m:
                        return RateLimitState.hit(
                            pattern=m.re.pattern,
                            source="json",
                            excerpt=blob[:280],
                        )
            if result_obj.get("is_error") is False and exit_code == 0:
                # Canonical "success" - do not false-positive on incidental
                # mentions of "rate-limit" / "limit" that appear in session
                # hook additionalContext echoed on stdout by stream-json.
                return RateLimitState.ok()

        # 2. Pattern-match stderr/stdout, but ONLY when exit_code != 0.
        #    A zero-exit call with a clean result event cannot be rate limited;
        #    any incidental "rate-limit" tokens in stream-json hook output are
        #    documentation noise, not a rate-limit signal.
        if exit_code != 0:
            for pat in _PATTERNS:
                m = pat.search(stderr)
                if m:
                    return RateLimitState.hit(
                        pattern=m.re.pattern, source="stderr", excerpt=stderr[:280]
                    )
                m = pat.search(stdout)
                if m:
                    return RateLimitState.hit(
                        pattern=m.re.pattern, source="stdout", excerpt=stdout[:280]
                    )

        return RateLimitState.ok()

    @staticmethod
    def _final_result_obj(stdout: str) -> dict | None:
        """Return the terminating ``{"type":"result", ...}`` object if present.

        Handles single-JSON and NDJSON stream transparently. Returns None for
        plain-text output or malformed JSON.
        """
        s = stdout.strip()
        if not s or not s.lstrip().startswith("{"):
            return None
        # Single-JSON (legacy --output-format=json)
        if "\n" not in s:
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                return None
            return obj if isinstance(obj, dict) else None
        # NDJSON stream: walk forward, keep the last `result` object seen.
        last_result: dict | None = None
        for line in s.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "result":
                last_result = obj
        return last_result
