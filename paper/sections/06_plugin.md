# The Cursor / Claude-Code Plugin

The harness ships as a single plugin, **`pratyaksha-context-eng-harness`** (v1.0.0, MIT-licensed), installable into Cursor, Claude Code (CLI and the VS Code extension), and Claude Desktop via the Model Context Protocol \citep{anthropic2025mcp}. This section specifies the shipped artefact in enough detail that an external reviewer can verify its claims directly against the repository at `github.com/SharathSPhD/pratyaksha-context-eng-harness`.

## Manifest and discovery

Two manifests are shipped:

- `.claude-plugin/plugin.json` — Claude-Code-format component manifest declaring the 3 agents, 4 commands, 3 skills, and the `hooks/hooks.json` registration.
- `marketplace.json` — top-level marketplace manifest used for `/plugin install` discovery, listing the canonical id, version, repository URL, license, keywords, and feature catalogue.

Both manifests are version-pinned to `1.0.0`, point to the same upstream URL, and resolve identically in Cursor (via the plugin marketplace \citep{cursor2025plugins}) and Claude Code (via the new plugin marketplace surface \citep{anthropic2025claudecode}). Setup is a single command: `curl -LsSf https://astral.sh/uv/install.sh | sh`; the MCP server's Python dependencies are installed lazily on first tool call by `uv` \citep{uv2025}, so a fresh user can install the plugin without ever invoking `pip`, creating a virtualenv, or running `claude mcp add`.

## The 15 MCP tools

The MCP server (`mcp/server.py`, FastMCP-style) exposes exactly fifteen tools. Their names, signatures, and operational roles are:

| # | Tool | Signature | Operationalizes |
|---|---|---|---|
| 1 | `context_insert` | `InsertInput → ContextElement` | Avacchedaka insertion (Section 4.2) |
| 2 | `context_retrieve` | `RetrieveInput → list[ContextElement]` | Default avacchedaka query |
| 3 | `context_get` | `GetByIdInput → ContextElement` | Direct id-based fetch (audit) |
| 4 | `context_sublate` | `SublateInput → ContextElement` | Bādha (basic, by id) |
| 5 | `sublate_with_evidence` | `SublateWithEvidenceInput → ...` | Bādha with explicit pointer + reason |
| 6 | `detect_conflict` | `RetrieveInput → {conflicted: bool, items: ...}` | Bayesian-margin conflict check |
| 7 | `list_qualificands` | `() → list[str]` | Diagnostic surface |
| 8 | `compact` | `CompactInput → CompactionReport` | Adaptive forgetting (Section 5.7) |
| 9 | `boundary_compact` | `BoundaryCompactInput → ...` | Event-boundary compaction (Section 5.6) |
| 10 | `context_window` | `ContextWindowInput → WindowSnapshot` | Visible-context summary for Manas |
| 11 | `set_sakshi` | `SetSakshiInput → ()` | Pin a session-stable witness invariant |
| 12 | `get_sakshi` | `() → list[SakshiInvariant]` | Retrieve the witness invariants |
| 13 | `classify_khyativada` | `ClassifyInput → KhyatiResult` | 7-class hallucination classifier |
| 14 | `budget_status` | `BudgetStatusInput → BudgetStatus` | TokenBudgetWatchdog (Section 5.8) |
| 15 | `budget_record` | `BudgetRecordInput → BudgetStatus` | Record consumed tokens per turn |

Every tool is typed via Pydantic v2 input/output models, returns JSON-serializable structures, and is exercised by at least one unit test in `tests/test_v2/`. A complete tool reference with full schemas is in **Appendix B**.

## The 3 sub-agents

- **`agents/manas.md`** — defines the *attentional sense-organ* **sub-agent**. System prompt instructs the model to call `get_sakshi`, `context_retrieve`, `context_window`, `classify_khyativada`, and `budget_record`, and to return structured JSON of the form `{ "draft": "...", "grounding": ["<element_id>", ...], "uncertain_claims": ["..."], "needs_buddhi": true | false }` (verbatim contract in the repository file). Manas must *never* emit a user-visible final answer.
- **`agents/buddhi.md`** — the *determinative judging* sub-agent. May call `sublate_with_evidence`, `detect_conflict`, and `classify_khyativada`. Emits the user-visible answer and its self-classified `khyati_class` and `confidence`.
- **`agents/sakshi-keeper.md`** — the *witness keeper*. Read-only on the live store; append-only on the JSON-lines audit log at `~/.cache/pratyaksha/audit.jsonl`. Invoked from the lifecycle hooks (Section 6.5).

The choice to ship Manas and Buddhi as *agents* (system-prompt-only sub-agents driven by the host LLM) rather than as fine-tunes is deliberate: it preserves the harness's host-platform-agnosticism (Section 5.4) and lets users swap the underlying model without re-training anything.

## The 3 skills

Following the Cursor/Claude-Code skill format (an `SKILL.md` per directory):

- **`skills/context-discipline/SKILL.md`** — surfaces the harness's discipline as a triggered skill: *"every retrieved item must be entered with avacchedaka qualifiers; no free-text dumps"*. Activates when the host agent is about to call a search/RAG tool.
- **`skills/sublate-on-conflict/SKILL.md`** — triggers Buddhi's call to `sublate_with_evidence` when `detect_conflict` returns true.
- **`skills/witness-prefix/SKILL.md`** — wraps every Buddhi prompt with a stable `<sakshi_invariants>` system block via `get_sakshi`, so the witness frame is *literally a system message*, not inlined into the user turn.

## The 4 slash commands

- **`/context-status`** — pretty-prints the visible context window, by category, with budget gauge.
- **`/sublate <target_id> <by_id> <reason>`** — manual sublation override (audit / debug).
- **`/budget`** — shorthand for `budget_status` rendered as a one-line gauge.
- **`/compact-now [strategy]`** — manually trigger compaction; `strategy ∈ {adaptive, lru, none}` (default `adaptive`).

## The 3 lifecycle hooks

`hooks/hooks.json` registers three hooks against the standard Claude Code lifecycle events:

```json
{
  "hooks": {
    "SessionStart": [{ "matcher": "*", "hooks": [
      { "type": "command", "command": "${CLAUDE_PLUGIN_ROOT}/hooks/session-start.sh", "timeout": 10 }
    ]}],
    "PreToolUse": [{ "matcher": "mcp__pratyaksha_mcp__.*", "hooks": [
      { "type": "command", "command": "${CLAUDE_PLUGIN_ROOT}/hooks/pretooluse-budget.sh", "timeout": 5 }
    ]}],
    "Stop": [{ "matcher": "*", "hooks": [
      { "type": "command", "command": "${CLAUDE_PLUGIN_ROOT}/hooks/stop-compact.sh", "timeout": 5 }
    ]}]
  }
}
```

Their roles are:

- **`session-start.sh`** seeds the session-stable Sākṣī invariants (working directory, git SHA, model id, plugin version) via `set_sakshi`.
- **`pretooluse-budget.sh`** consults `budget_status` before any harness tool runs and **logs** over-threshold use by default (advisory hook). If `PRATYAKSHA_BUDGET_STRICT=1`, the same script can be configured to **deny** tool execution when the gauge exceeds a hard threshold (default 95% of `context_budget`).
- **`stop-compact.sh`** invokes `compact(strategy="adaptive")` at the end of every turn so memory pressure does not accumulate across turns.

## Worked example: a Redis-caching turn, end-to-end

To make the runtime contract concrete we trace a single user turn end-to-end through the deployed plugin. The user prompt is *"how do I cache a user session in Redis?"*; the agent's tool returns a mix of pre-Redis-7 blog snippets and the official Redis 7 documentation. Figure~\ref{fig:swimlane} visualises the swimlane across **User → Manas → Buddhi → Sublation → Sākṣī**, with the per-stage immutable JSON line written into `~/.cache/pratyaksha/audit.jsonl`.

```{=latex}
\input{figures_tikz/fig6_swimlane.tex}
```

The five host-visible artefacts are: (i) one `mcp__pratyaksha_mcp__manas_step` JSON-RPC call (Manas attends, $K=3$ items under a 1.2 K-token budget); (ii) one `detect_conflict` round-trip flagging a TYPE\_CLASH on the `(Redis-session, expiry-policy)` qualificand-qualifier slot; (iii) one `sublate_with_evidence` call that retires the blog post in favour of the Redis 7 docs (limitor precedence: official docs `prec=8` > blog post `prec=2`); (iv) one `mcp__pratyaksha_mcp__buddhi_step` call returning the answer with `khyati_class = "yathārtha"` (veridical) and `confidence = 0.91`; and (v) one append to the Sākṣī log per stage. The complete `/context-status` snapshot, the raw JSON-RPC payloads, and the four matching audit-log lines are reproduced verbatim in **Appendix C.8** so that the turn can be replayed byte-for-byte from the shipped plugin against the cached evidence trail.

This single turn exercises every load-bearing primitive of Sections 4–5 in a non-coding agentic context, and is the runtime template the L1 (§8) and L3 (§10) experiments instantiate at scale.

## What the plugin does *not* contain

We assert and audit two negative claims:

1. **Zero references to the dev-time orchestration tools** `attractor-flow`, `ralph-loop`, and `triz-engine`. Those tools were used to build the harness (Section 3) but are not runtime dependencies. A grep over the entire shipped plugin tree returns zero matches; the audit is preserved as `docs/no_dev_deps_audit.md` in the repository.
2. **Zero hard dependencies on heavyweight stacks** like `vllm`, `mlflow`, `chromadb`, or `huggingface-hub`. The harness's runtime requirements are `mcp`, `pydantic`, `tiktoken`, `numpy`, and `anthropic`; the optional Qwen3-1.7B surprise backend is loaded only when the user opts in.

This matters because the harness is a *discipline*, not an infrastructure. Anything heavier than that would defeat the purpose of a plugin you can install in 30 seconds.

## Smoke test and CI

`mcp/smoke_test.py` exercises every tool against the local MCP server: 15 round-trips, each verifying the input schema, the output schema, and one non-trivial behavioural assertion (e.g. `sublate_with_evidence` actually flips the target item's status). The smoke test runs in roughly 4 seconds wall-clock and is wired into the repo's CI as `pytest -m smoke`. We additionally exercise every command and hook via `claude --debug` in the developer's local Claude Code installation; the transcript of one such smoke run is preserved in `docs/plugin_smoke_transcript.md`.

## Hot-swappability across hosts

The same `marketplace.json` resolves in:

- **Cursor** — via the IDE's plugin browser, registered by URL.
- **Claude Code (CLI)** — via `/plugin install <url>`.
- **Claude Code (VS Code extension)** — via the same `/plugin install` UI.
- **Claude Desktop** — via the Desktop app's MCP-based plugin surface.

We have manually exercised the install path on all four hosts during development. The harness's runtime invariants (Sections 4 and 5) hold identically across all four because the only inter-process surface is MCP, and MCP is host-agnostic by construction \citep{anthropic2025mcp}.
