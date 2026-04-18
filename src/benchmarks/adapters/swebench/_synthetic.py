"""Deterministic synthetic SWE-bench-style examples for CI/offline mode.

Each synthetic instance carries:
  - a small, plausible Python "bug report" problem statement
  - the file path that should be modified
  - a "gold" unified diff that fixes it

Real validation lives behind `load_real=True` and the SWE-bench docker
harness; this synthetic generator exists so unit tests, smoke tests, and
the plugin's offline mode never block on `princeton-nlp/SWE-bench_Verified`.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    file_path: str
    gold_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]


_BUG_TEMPLATES: tuple[dict, ...] = (
    {
        "repo": "synthorg/utils",
        "file": "synth_utils/strings.py",
        "summary": "snake_to_camel returns wrong case for first segment",
        "details": (
            "snake_to_camel('user_name') currently returns 'User_Name' but "
            "should return 'userName' (lower-cased first segment)."
        ),
        "patch": (
            "diff --git a/synth_utils/strings.py b/synth_utils/strings.py\n"
            "--- a/synth_utils/strings.py\n"
            "+++ b/synth_utils/strings.py\n"
            "@@ -10,7 +10,8 @@ def snake_to_camel(s: str) -> str:\n"
            "     parts = s.split('_')\n"
            "-    return ''.join(p.capitalize() for p in parts)\n"
            "+    head, *tail = parts\n"
            "+    return head.lower() + ''.join(p.capitalize() for p in tail)\n"
        ),
        "fail_to_pass": ["tests/test_strings.py::test_snake_to_camel"],
    },
    {
        "repo": "synthorg/timekit",
        "file": "timekit/parser.py",
        "summary": "parse_iso8601 silently drops timezone offset",
        "details": (
            "parse_iso8601('2025-04-18T12:00:00+02:00') currently returns a "
            "naive datetime instead of preserving the +02:00 offset."
        ),
        "patch": (
            "diff --git a/timekit/parser.py b/timekit/parser.py\n"
            "--- a/timekit/parser.py\n"
            "+++ b/timekit/parser.py\n"
            "@@ -22,7 +22,7 @@ def parse_iso8601(s: str) -> datetime:\n"
            "-    return datetime.fromisoformat(s.split('+')[0])\n"
            "+    return datetime.fromisoformat(s)\n"
        ),
        "fail_to_pass": ["tests/test_parser.py::test_parse_iso_with_offset"],
    },
    {
        "repo": "synthorg/cachekit",
        "file": "cachekit/lru.py",
        "summary": "LRUCache eviction order is wrong on tied access counts",
        "details": (
            "When two keys have identical hit counts, the cache evicts the "
            "most recently inserted key instead of the least recently used."
        ),
        "patch": (
            "diff --git a/cachekit/lru.py b/cachekit/lru.py\n"
            "--- a/cachekit/lru.py\n"
            "+++ b/cachekit/lru.py\n"
            "@@ -45,7 +45,7 @@ class LRUCache:\n"
            "-        evict_key = max(self._store, key=lambda k: self._inserted_at[k])\n"
            "+        evict_key = min(self._store, key=lambda k: self._last_used[k])\n"
        ),
        "fail_to_pass": ["tests/test_lru.py::test_eviction_on_tied_count"],
    },
    {
        "repo": "synthorg/jsonkit",
        "file": "jsonkit/serializer.py",
        "summary": "serialize() crashes on dict with bytes keys",
        "details": (
            "serialize({b'k': 1}) raises TypeError because the serializer "
            "passes bytes objects directly to json.dumps."
        ),
        "patch": (
            "diff --git a/jsonkit/serializer.py b/jsonkit/serializer.py\n"
            "--- a/jsonkit/serializer.py\n"
            "+++ b/jsonkit/serializer.py\n"
            "@@ -8,6 +8,8 @@ def serialize(obj):\n"
            "     if isinstance(obj, dict):\n"
            "+        obj = {k.decode() if isinstance(k, bytes) else k: v\n"
            "+               for k, v in obj.items()}\n"
            "         return json.dumps(obj)\n"
        ),
        "fail_to_pass": ["tests/test_serializer.py::test_bytes_keys"],
    },
    {
        "repo": "synthorg/httpkit",
        "file": "httpkit/client.py",
        "summary": "Client.get() does not retry on 502 responses",
        "details": (
            "The retry policy mentions 502 in the docstring but the "
            "implementation only retries on 503 and 504."
        ),
        "patch": (
            "diff --git a/httpkit/client.py b/httpkit/client.py\n"
            "--- a/httpkit/client.py\n"
            "+++ b/httpkit/client.py\n"
            "@@ -34,7 +34,7 @@ RETRY_STATUSES = {503, 504}\n"
            "-RETRY_STATUSES = {503, 504}\n"
            "+RETRY_STATUSES = {502, 503, 504}\n"
        ),
        "fail_to_pass": ["tests/test_client.py::test_retry_on_502"],
    },
    {
        "repo": "synthorg/configkit",
        "file": "configkit/loader.py",
        "summary": "load_yaml ignores environment variable interpolation",
        "details": (
            "Values like '${HOME}' in the YAML file are returned literally "
            "instead of being expanded against os.environ."
        ),
        "patch": (
            "diff --git a/configkit/loader.py b/configkit/loader.py\n"
            "--- a/configkit/loader.py\n"
            "+++ b/configkit/loader.py\n"
            "@@ -15,4 +15,5 @@ def load_yaml(path):\n"
            "     with open(path) as f:\n"
            "-        return yaml.safe_load(f)\n"
            "+        raw = yaml.safe_load(f)\n"
            "+    return _expand_env(raw)\n"
        ),
        "fail_to_pass": ["tests/test_loader.py::test_env_interpolation"],
    },
)


def generate_swe_examples(*, n: int, seed: int) -> list[SyntheticInstance]:
    """`n` deterministic synthetic SWE-bench-style instances."""
    if n <= 0:
        raise ValueError("n must be > 0")
    rng = random.Random(seed)
    out: list[SyntheticInstance] = []
    for i in range(n):
        tmpl = _BUG_TEMPLATES[rng.randrange(len(_BUG_TEMPLATES))]
        digest = hashlib.sha1(f"swe-{seed}-{i}-{tmpl['file']}".encode()).hexdigest()[:12]
        problem_statement = (
            f"# {tmpl['summary']}\n\n"
            f"{tmpl['details']}\n\n"
            f"## Reproduce\n"
            f"Run the failing test in `tests/` against `{tmpl['file']}`."
        )
        out.append(
            SyntheticInstance(
                instance_id=f"synth_swe_{seed:02d}_{i:04d}_{digest}",
                repo=tmpl["repo"],
                base_commit=digest,
                problem_statement=problem_statement,
                file_path=tmpl["file"],
                gold_patch=tmpl["patch"],
                fail_to_pass=list(tmpl["fail_to_pass"]),
                pass_to_pass=[],
            )
        )
    return out
