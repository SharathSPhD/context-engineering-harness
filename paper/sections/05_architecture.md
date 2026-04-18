# 5 · The Pratyakṣa Harness: System Architecture

We describe the harness as a layered architecture: a Core ContextStore, three runtime modules (Avacchedaka query, Sublation, Bayesian aggregation), three reasoning modules (Buddhi/Manas, Sākṣī, Khyātivāda classifier), and three resource-management modules (EventBoundaryCompactor, AdaptiveForgetting, TokenBudgetWatchdog). Each module is implemented as one or more MCP tools (Section 6.2 / Appendix B) and is exercised by at least one of the seven preregistered hypotheses (H1–H7).

## 5.1 The ContextStore (core)

The `ContextStore` is a typed, append-mostly key-value store that lives outside the LLM and is exposed to it via MCP. Its row schema is exactly the *avacchedaka* tuple of Section 4.2:

```python
@dataclass(frozen=True)
class ContextItem:
    id: str
    qualificand: str         # the "thing" being asserted about
    qualifier: str           # what is asserted about it
    condition: str           # the limitor under which it holds
    precision: float         # source-supplied calibrated reliability ∈ [0, 1]
    source: str              # "user" | "rag" | "tool:foo" | "agent:buddhi" | ...
    inserted_at: float       # monotonic seconds
    stale: bool              # source-flagged staleness
    status: Literal["live", "bādhita", "evicted"]
    superseded_by_id: str | None
    witness_id: str          # set by SakshiKeeperAgent on first witness
```

Insertion is via a single MCP tool, `insert(...)`. Retrieval is via three MCP tools — `retrieve_by_id`, `retrieve_by_qualifier`, and `retrieve_under_condition` — none of which return `bādhita` or `evicted` rows by default. A debug flag exposes them for audit.

Two design commitments distinguish this from a vector store:

1. **Conditions are first-class.** Two rows that share `qualificand` and disagree on `qualifier` are *not* yet in conflict if their `condition` fields differ. This is the avacchedaka commitment.

2. **Provenance never disappears.** Sublation rewrites `status`, never deletes. Eviction (under hard budget pressure) writes to a sidecar `evicted_log.jsonl` so the witness can replay the eviction reason.

## 5.2 The AvacchedakaQuery module

`retrieve_under_condition(qualificand: str, condition: str)` walks the `ContextStore` and returns the *set* of items matching `qualificand` whose `condition` is *consistent* with the supplied condition. Consistency is decided by a small rule engine:

- **String equality** wins when conditions are identical.
- **Temporal subsumption** wins when one condition is a strictly tighter time-window of the other (e.g. `"Django ≥ 5.0"` subsumes `"Django 5.0.4"`).
- **Lexical-prefix subsumption** for versioned platforms (`"POSIX"` subsumes `"POSIX, Python ≥ 3.11"`).
- **Otherwise**, items are returned in *both* groups and the conflict is left to be resolved downstream by Buddhi, possibly via `sublate_with_evidence`.

This rule engine is intentionally simple. We are not building a description-logic reasoner; we are providing the agent with a typed surface on which to act.

## 5.3 The Sublation + Bayesian aggregation module

Conflict resolution proceeds in two steps. First, `sublate_with_evidence(target_id, by_id, reason)` performs the Vedānta operation of Section 4.3. Second, when the agent must commit to a single posterior probability for a contested claim — for example, "is the user's setting `enable_x = True`?" — the harness offers a **Bayesian Beta-Bernoulli aggregator** in place of the original *PrecisionWeightedRAG* of v0:

For a claim $H \in \{0, 1\}$ supported by $k$ source items with precisions $p_1, \dots, p_k$ and source-side votes $v_1, \dots, v_k \in \{0, 1\}$, the harness models each source as a Bernoulli with reliability $p_i$ and updates a Beta($\alpha_0, \beta_0$) prior:

$$
\alpha_n = \alpha_0 + \sum_{i: v_i=1} p_i,\qquad \beta_n = \beta_0 + \sum_{i: v_i=0} p_i.
$$

The posterior mean $\bar{H} = \alpha_n / (\alpha_n + \beta_n)$ is returned, alongside the posterior margin $|\bar{H} - 0.5|$. A claim is reported as *conflicted* when this margin falls below a threshold $\tau$ (default $0.10$). Calibration is evaluated by **Brier score** \citep{brier1950score} and **expected calibration error** (ECE) with equal-width binning \citep{naeini2015ece, guo2017calibration}; on H2 the Bayesian aggregator achieves Brier 0.094 vs. PrecisionWeightedRAG 0.176 and ECE 0.041 vs. 0.118 on a held-out 1,800-example validation slice (Section 8.2, T2 in Appendix C), corroborating the broader literature on Bayesian fusion under conflicting evidence \citep{singh2025bayesianfusion, ovadia2019can, gal2016dropout}.

## 5.4 Buddhi and Manas as plug-in sub-agents

The two-stage gate is implemented as two MCP-side prompt templates and a single orchestration policy in the host agent (Buddhi-Manas Orchestrator). The data-flow is:

1. The host agent enters the harness's `manas_step(query)` which returns the Manas JSON.
2. The host agent enters `buddhi_step(query, manas_output)` which returns the final answer + khyāti class.
3. The host agent enters `sakshi_record(...)` which appends the immutable witness log entry.

Both Manas and Buddhi are *prompted-only* sub-agents. We deliberately do *not* fine-tune. The two-stage gate is a discipline, not a model: a different LLM (Claude 3.5 Sonnet, Claude 4.x, GPT-4o, Qwen-3) can be plugged in as the *runner* of either prompt without breaking the harness's invariants. This is the "host-platform-agnostic" design commitment, and it makes the plugin hot-swappable across Cursor / Claude Code / Claude Desktop \citep{anthropic2025mcp, cursor2025plugins}.

## 5.5 The Khyativada classifier

The classifier is a single Claude prompt with structured JSON output. Inputs:

- The user query.
- The Buddhi answer.
- (Optional) the Manas-attended items.

Output:

```json
{
  "khyati_class": "anyathākhyāti" | "ātmakhyāti" | ... | "none",
  "rationale": "<≤2 sentence explanation citing item-ids and qualifiers>",
  "confidence": <float in [0,1]>
}
```

Sample exemplars are curated to maximize within-class lexical diversity (e.g., asatkhyāti includes both "fabricated CVE" and "fabricated stack-trace line", not just one). A heuristic guardrail overrides the LLM-side label when (a) the answer cites a `qualificand` not present in any `ContextStore` item (force `asatkhyāti`), or (b) the answer's `qualifier` is the negation of an item the answer cites by id (force `viparītakhyāti`). These guardrails are *additive* — they only flip `none` to a class, never flip one class to another.

## 5.6 The EventBoundaryCompactor

For long-running sessions we want to compact whole *episodes* of context, not individual items. Following event-segmentation theory \citep{zacks2007event, baldassano2017nested} and predictive-coding accounts of surprise \citep{rao1999predictive, friston2010fep, feldman2010precision}, we segment the rolling token stream by per-token **negative-log-probability surprise** computed by a small open-weights model — Qwen3-1.7B served via vLLM \citep{kwon2023vllm, vllm2023, qwen2024technical}, with an optional fallback to Qwen3-0.6B and a final heuristic-Zipf fallback \citep{piantadosi2014zipf, shannon1948mathematical, mackay2003information} when the GPU path is unavailable.

A boundary is declared when surprise exceeds a rolling-mean+2σ threshold sustained across $w=12$ tokens. At boundary, the EventBoundaryCompactor:

1. Summarises the just-closed episode into a single new `ContextItem` with `qualifier="episode_summary"` and `precision = mean(precisions of contained items)`.
2. Marks all contained items as `status="evicted"` (provenance retained).
3. Notifies the SakshiKeeperAgent of the boundary.

H4 (Section 8.4) tests this module against a no-compaction baseline.

## 5.7 AdaptiveForgetting

Rules are as in Section 4.7. Implemented as a periodic background MCP tool `compact_now(strategy: Literal["adaptive", "lru", "none"])` that the agent or a lifecycle hook can invoke. The default policy is `adaptive`; the H7 study (Section 8.7) compares all three.

## 5.8 TokenBudgetWatchdog

A small but indispensable module. The watchdog tracks input/output tokens by category (system, retrieval, tool, completion) against a configured budget (default 60% of model context window) and exposes one MCP tool, `budget_status()`, returning:

```json
{
  "context_tokens_used": 12345,
  "context_budget": 16384,
  "fraction_used": 0.753,
  "by_category": {"system": 1024, "retrieval": 7000, "tool": 3000, "completion": 1321},
  "compaction_recommended": false
}
```

It is also wired into a `pretooluse-budget` hook (Section 6.4) so the host can refuse expensive tool calls when over budget.

## 5.9 The full request lifecycle

A representative single-turn request now flows as:

1. **User query** lands in the host agent (Claude Code, Cursor, or Claude Desktop).
2. **Manas** is called: it inspects the `ContextStore` snapshot via `retrieve_by_qualifier` and returns its attended subset.
3. **Buddhi** is called with that subset. It may issue **`sublate_with_evidence`** for any conflicts it spots before answering.
4. **Khyativada classifier** is run on Buddhi's draft answer; if `khyati_class != "none"` and `confidence > τ_self_check` (default 0.7), Buddhi is asked to revise once.
5. **Sākṣī** writes the witness log entry.
6. The answer is returned to the user.
7. **AdaptiveForgetting** and **EventBoundaryCompactor** are invoked by lifecycle hooks if budget pressure is detected.

This lifecycle is the running invariant tested by every experiment in Sections 8–10.
