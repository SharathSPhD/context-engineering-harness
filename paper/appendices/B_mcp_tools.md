# Appendix B · Full MCP Tool Specifications

This appendix gives the full input / output schema and behavioural contract for each of the 15 MCP tools shipped by `pratyaksha-context-eng-harness` v1.0.0. The canonical source is `plugin/pratyaksha-context-eng-harness/mcp/server.py`.

All tools share three host-platform invariants:

- **Audit hook.** Every successful tool call appends a single JSON-line to `${CLAUDE_PLUGIN_ROOT}/witness_log.jsonl` via the `_audit(event, payload)` helper.
- **Budget hook.** Every tool call is preceded by the `pretooluse-budget.sh` lifecycle hook, which consults `budget_status` and refuses execution if the gauge exceeds the configured hard threshold.
- **Schema-strict I/O.** Every tool's input is validated by a Pydantic v2 model; every output is JSON-serialisable.

---

## B.1 `context_insert` — avacchedaka insertion

**Signature.** `context_insert(args: InsertInput) -> ContextElementOut`

```python
class InsertInput(BaseModel):
    qualificand: str
    qualifier: str
    condition: str = ""
    precision: float = Field(ge=0.0, le=1.0, default=0.5)
    source: str = "agent"
    stale: bool = False
    superseded_by_id: str | None = None
```

**Behaviour.** Generates a new `id` (UUIDv4), writes the row to the in-memory `ContextStore`, returns the serialised row.

---

## B.2 `context_retrieve` — default avacchedaka query

**Signature.** `context_retrieve(args: RetrieveInput) -> {items: list[ContextElement]}`

```python
class RetrieveInput(BaseModel):
    qualificand: str | None = None
    qualifier_substring: str | None = None
    condition: str | None = None
    include_bādhita: bool = False
    include_evicted: bool = False
    limit: int = 50
```

**Behaviour.** Walks the store, applies the rule engine (Section 5.2), returns matching live items by default.

---

## B.3 `context_get` — direct id-based fetch

**Signature.** `context_get(args: GetByIdInput) -> ContextElement | None`

```python
class GetByIdInput(BaseModel):
    id: str
```

**Behaviour.** Audit/debug surface; returns the element regardless of `status`.

---

## B.4 `context_sublate` — basic sublation

**Signature.** `context_sublate(args: SublateInput) -> ContextElement`

```python
class SublateInput(BaseModel):
    id: str
    reason: str = ""
```

**Behaviour.** Sets `status = "bādhita"` on the target. No `by_id` reference. Used when the agent has no specific superseding item to point to (e.g. policy-driven eviction).

---

## B.5 `sublate_with_evidence` — evidence-anchored sublation

**Signature.** `sublate_with_evidence(args: SublateWithEvidenceInput) -> {target: ContextElement, by: ContextElement}`

```python
class SublateWithEvidenceInput(BaseModel):
    target_id: str
    by_id: str
    reason: str
```

**Behaviour.** This is the *primary* bādha primitive (Section 4.3). Sets `target.status = "bādhita"` and `target.superseded_by_id = by_id`, retains the original for audit. Triggers the rule engine's *dominance check* (Section 4.3) automatically against any other live items sharing `(qualificand, condition)` whose precision is dominated by `by`.

---

## B.6 `detect_conflict` — Bayesian-margin conflict check

**Signature.** `detect_conflict(args: RetrieveInput) -> {conflicted: bool, margin: float, items: list[ContextElement]}`

**Behaviour.** Runs the Bayesian Beta-Bernoulli aggregator (Section 5.3) over the matching items. Returns `conflicted=True` when the posterior margin $|\bar{H} - 0.5|$ falls below the threshold $\tau$ (default 0.10).

---

## B.7 `list_qualificands` — diagnostic surface

**Signature.** `list_qualificands() -> {qualificands: list[str]}`

**Behaviour.** Returns the deduplicated list of `qualificand`s across the live store. Used by Manas to map "what topics do I currently know about?".

---

## B.8 `compact` — adaptive forgetting

**Signature.** `compact(args: CompactInput) -> CompactionReport`

```python
class CompactInput(BaseModel):
    strategy: Literal["adaptive", "lru", "none"] = "adaptive"
    max_keep: int | None = None
```

**Behaviour.** Applies the `AdaptiveForgetting` rules (Section 5.7). Evicted items are written to `evicted_log.jsonl`. The `none` strategy is a no-op for ablation studies.

---

## B.9 `boundary_compact` — event-boundary compaction

**Signature.** `boundary_compact(args: BoundaryCompactInput) -> {summarised_count: int, summary_id: str}`

```python
class BoundaryCompactInput(BaseModel):
    surprise_window_size: int = 12
    surprise_threshold_sigma: float = 2.0
    backend: Literal["vllm", "hf", "heuristic"] = "heuristic"
```

**Behaviour.** Implements Section 5.6. Runs the configured surprise backend, declares boundaries, summarises closed episodes into single `episode_summary` items.

---

## B.10 `context_window` — visible-context summary for Manas

**Signature.** `context_window(args: ContextWindowInput) -> WindowSnapshot`

```python
class ContextWindowInput(BaseModel):
    max_items: int = 50
    qualificand: str | None = None
```

**Behaviour.** Returns a token-budget-aware snapshot of the live store, sorted by `precision` descending, with each item rendered as `{id, qualificand, qualifier, condition, precision, source}`. Used by `ManasAgent`.

---

## B.11 `set_sakshi` — pin a session-stable witness invariant

**Signature.** `set_sakshi(args: SetSakshiInput) -> {ok: True}`

```python
class SetSakshiInput(BaseModel):
    key: str
    value: str
```

**Behaviour.** Pins one `(key, value)` pair into the witness invariant store. Invoked by `session-start.sh` to seed `cwd`, `git_sha`, `model_id`, `plugin_version`.

---

## B.12 `get_sakshi` — retrieve the witness invariants

**Signature.** `get_sakshi() -> {invariants: dict[str, str]}`

**Behaviour.** Returns the full invariant dict. Used by the `witness-prefix` skill to wrap every Buddhi prompt with a stable `<sakshi_invariants>` system block.

---

## B.13 `classify_khyativada` — 7-class hallucination classifier

**Signature.** `classify_khyativada(args: ClassifyInput) -> KhyatiResult`

```python
class ClassifyInput(BaseModel):
    claim: str
    ground_truth: str
    context: str = ""

class KhyatiResult(BaseModel):
    khyati_class: Literal[
        "anyathākhyāti", "ātmakhyāti", "anirvacanīyakhyāti",
        "asatkhyāti", "viparītakhyāti", "akhyāti", "none",
    ]
    rationale: str
    confidence: float
```

**Behaviour.** Section 5.5. Few-shot Claude-side classifier with structured JSON output and rule-based guardrails (force `asatkhyāti` for non-existent qualificands; force `viparītakhyāti` for inverted-boolean cited items).

---

## B.14 `budget_status` — TokenBudgetWatchdog gauge

**Signature.** `budget_status(args: BudgetStatusInput) -> BudgetStatus`

```python
class BudgetStatusInput(BaseModel):
    budget_total: int = 16384
```

**Behaviour.** Returns `{context_tokens_used, context_budget, fraction_used, by_category, compaction_recommended}`. Reads the running tally from the `_budget_gauge` file written by `budget_record`.

---

## B.15 `budget_record` — record consumed tokens per turn

**Signature.** `budget_record(args: BudgetRecordInput) -> BudgetStatus`

```python
class BudgetRecordInput(BaseModel):
    category: Literal["system", "retrieval", "tool", "completion"]
    n_tokens: int
```

**Behaviour.** Increments the per-category counter and re-emits the gauge. Called by lifecycle hooks at the boundaries of every host action.

---

## B.16 Negative claims — what the plugin does *not* do

We assert and audit:

- **No reference to `attractor-flow`, `ralph-loop`, or `triz-engine`** anywhere in the shipped plugin tree.
- **No hard dependency on `vllm`, `mlflow`, `chromadb`, or `huggingface-hub`.** The runtime requirements are `mcp`, `pydantic`, `tiktoken`, `numpy`, `anthropic`. The optional `vllm` / `transformers` path for the Qwen3 surprise backend (Section 5.6) is loaded only when the user opts in.
- **No model fine-tuning anywhere.** Every "agent" in this work is a system-prompted role over an unmodified frontier LLM.

The audit reproducer is `docs/no_dev_deps_audit.md` in the repository.
