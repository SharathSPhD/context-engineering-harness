"""Synthetic per-instance "research trail" for P6-C.

For every SWE-bench Verified instance the runner picks up, this module
generates the kind of mixed-precision research evidence a real coding
agent would assemble *before* writing a patch:

  * the official issue text (high-precision, on-target file);
  * one or two correct grep/rg hits (high precision);
  * one or two stale Stack Overflow answers that name a *wrong*
    file or a deprecated symbol (low precision, marked stale);
  * one or two short blog snippets that paraphrase the bug
    (medium precision, possibly partially wrong).

The fixture is fully deterministic per (instance_id, seed) and contains
no LLM calls and no network. Both arms of the head-to-head consume the
*same* trail; only the *bookkeeping discipline* differs (the
with-harness arm uses the plugin's `sublate_with_evidence` and
precision-gated `compact`; the without-harness arm dumps everything
into the prompt under the same fixed token budget).

The trail is engineered so that an unaided agent that fails to filter
will see a contradicted file path or symbol in its prompt and, under
the budget cap, has a non-trivial chance of patching the wrong file.
This is the documented "Lost-in-the-Middle / mixed-precision retrieval
contamination" failure mode the plugin targets.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchSnippet:
    """One piece of evidence the agent surfaces during research."""
    id: str
    instance_id: str
    qualificand: str
    qualifier: str
    condition: str
    content: str
    precision: float
    source: str
    stale: bool
    superseded_by_id: str | None


_STALE_TEMPLATES: tuple[dict, ...] = (
    {
        "src": "stackoverflow.com/q/{q} (2017, 89 votes)",
        "preface": "I had the same problem; the fix is in",
        "wrong_path_offset": 1,  # use a *different* file in same repo
    },
    {
        "src": "old-blog.example.com/posts/{slug}",
        "preface": "Workaround: monkey-patch the module that owns",
        "wrong_path_offset": 2,
    },
)

_FRESH_TEMPLATES: tuple[dict, ...] = (
    {
        "src": "github.com/{repo}/blob/main/CHANGES.rst (most recent entry)",
        "preface": "Per the project changelog the relevant module is",
    },
    {
        "src": "github.com/{repo}/issues/{n} (maintainer comment, this week)",
        "preface": "Maintainer confirms the bug lives in",
    },
)


def _sibling_path(file_path: str, offset: int) -> str:
    """Construct a *different but plausible* file path in the same repo
    so the stale snippet looks credible to a naive agent.
    """
    parts = file_path.rsplit("/", 1)
    if len(parts) != 2:
        return f"misc/legacy_{offset}.py"
    pkg, fname = parts
    stem, _, ext = fname.rpartition(".")
    return f"{pkg}/{stem}_legacy_{offset}.{ext or 'py'}"


def generate_research_trail(
    *,
    instance_id: str,
    repo: str,
    file_path: str,
    issue_summary: str,
    seed: int,
    n_stale: int = 2,
    n_fresh: int = 2,
) -> list[ResearchSnippet]:
    """Build the deterministic research trail for one instance.

    Returns the snippets in *discovery order* — fresh evidence is
    interleaved so the agent does not encounter all stale-then-all-fresh
    (which would make the failure mode trivial). The stale snippets
    point at sibling files / deprecated symbols and carry strictly
    lower precision than every fresh snippet.
    """
    rng = random.Random(
        int(hashlib.sha1(f"p6c|{instance_id}|{seed}".encode()).hexdigest(), 16) & 0xFFFFFFFF
    )
    out: list[ResearchSnippet] = []

    qualificand = f"swe::{instance_id}"
    qualifier = "patch_target_file"
    condition = f"repo={repo}"

    # Stale snippets first (search-engine bias toward older high-traffic answers)
    for i in range(n_stale):
        tmpl = _STALE_TEMPLATES[i % len(_STALE_TEMPLATES)]
        wrong_path = _sibling_path(file_path, tmpl["wrong_path_offset"])
        snip_id = f"{instance_id}::stale-{i}"
        digest = hashlib.sha1(snip_id.encode()).hexdigest()[:6]
        content = (
            f"{tmpl['preface']} `{wrong_path}`. The reported behaviour matches a "
            f"stale workaround posted years ago; do not edit any other module."
        )
        out.append(
            ResearchSnippet(
                id=snip_id,
                instance_id=instance_id,
                qualificand=qualificand,
                qualifier=qualifier,
                condition=condition,
                content=content,
                precision=round(0.20 + rng.uniform(0.0, 0.20), 4),  # 0.20–0.40
                source=tmpl["src"].format(q=digest, slug=digest, repo=repo, n=int(digest, 16) % 9999),
                stale=True,
                superseded_by_id=f"{instance_id}::fresh-0",
            )
        )

    # One authoritative changelog / maintainer entry (fresh, high precision)
    for j in range(n_fresh):
        tmpl = _FRESH_TEMPLATES[j % len(_FRESH_TEMPLATES)]
        snip_id = f"{instance_id}::fresh-{j}"
        digest = hashlib.sha1(snip_id.encode()).hexdigest()[:6]
        content = (
            f"{tmpl['preface']} `{file_path}`. Issue summary: {issue_summary[:160]}"
        )
        out.append(
            ResearchSnippet(
                id=snip_id,
                instance_id=instance_id,
                qualificand=qualificand,
                qualifier=qualifier,
                condition=condition,
                content=content,
                precision=round(0.85 + rng.uniform(0.0, 0.10), 4),  # 0.85–0.95
                source=tmpl["src"].format(repo=repo, n=int(digest, 16) % 9999),
                stale=False,
                superseded_by_id=None,
            )
        )

    rng.shuffle(out)
    return out


__all__ = ["ResearchSnippet", "generate_research_trail"]
