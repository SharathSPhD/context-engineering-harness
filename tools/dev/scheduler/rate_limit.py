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
        # 1. JSON-structured error (most reliable when present)
        if stdout.strip().startswith("{"):
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                subtype = str(
                    data.get("subtype")
                    or data.get("error", {}).get("type", "")
                    if isinstance(data.get("error"), dict)
                    else data.get("subtype", "")
                ).lower()
                if any(s in subtype for s in ("rate_limit", "usage_limit", "quota_exceeded")):
                    return RateLimitState.hit(
                        pattern=subtype, source="json", excerpt=stdout[:280]
                    )
                if data.get("is_error") and any(
                    p.search(json.dumps(data)) for p in _PATTERNS
                ):
                    return RateLimitState.hit(
                        pattern="is_error+pattern",
                        source="json",
                        excerpt=stdout[:280],
                    )

        # 2. Pattern-match stderr (exit_code != 0 path)
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
