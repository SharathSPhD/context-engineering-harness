# TRIZ Analysis: Context Engineering Contradiction Resolution

> Session: `context-engineering-synthesis-2026-04-17` — logged at `.triz/session.jsonl`

---

## Problem Statement

Context engineering in LLM systems faces three interlocking contradictions that no current solution fully eliminates. All three have been independently identified across cognitive neuroscience and classical Indian epistemology, suggesting they are structural rather than incidental.

---

## Contradiction 1 — Technical (Primary)

**Improving:** P26 — Quantity of substance/matter *(context volume: more tokens, documents, retrieved passages in the window)*

**Worsening:** P28 — Measurement accuracy *(reasoning quality: accuracy, hallucination rate, coherence — all degrade with context volume)*

**Empirical evidence:** Liu et al. 2023 (lost-in-the-middle U-curve), Chroma 2025 (context rot across 18 frontier models), RULER 2024, NoCha 2024, HELMET 2024.

**Matrix lookup (P26 → P28):** Principles **13, 2, 28**

### Solution Sketch A — Principle 13: The Other Way Around (IFR: 3/4 ★★★)

> *Invert the context-loading direction from push to pull.*

**Classical TRIZ action:** Fix what moved; move what was fixed. Reverse producer/consumer roles.

**Domain application:** Instead of pre-populating the context window and hoping the attention mechanism retrieves correctly, invert the flow: the model **emits structured retrieval signals during generation** that specify exactly what context is needed at each reasoning step.

This extends Self-RAG's reflection tokens into a richer pull-based protocol:

```python
# Generation step emits an avacchedaka-tagged retrieval signal:
# <retrieve: condition="current_code_file=auth.py AND task_step=refactor_auth_flow"
#            limitor="function-level" precision_threshold=0.85>
```

**Vedic mapping:** This is *anuvyavasāya* (second-order cognition certifying the first) combined with *avacchedaka* limitors — the system knows WHAT it knows and requests ONLY that. The generation process enacts *svārthānumāna* (inference for itself) with explicit epistemic tagging.

**Neuroscience mapping:** Mirrors dorsal attention network's goal-directed top-down selection — PFC broadcasts the task template, posterior regions compete for access. Context is not passively received but actively assembled under task demand.

**Eliminates the contradiction because:** Active window at any moment contains only demanded context — quantity is minimized while accuracy-relevant content is maximized. No more O(n²) cost on irrelevant tokens.

**IFR criteria met:** minimal cost ✓ | no new problems ✓ | self-resolving ✓ | leverages existing ✗ (requires new retrieval protocol)

---

### Solution Sketch B — Principle 2: Taking Out (IFR: 3/4 ★★★)

> *Separate the disturbing element; extract only the necessary property.*

**Domain application:** At each agent decision boundary, extract only the *avacchedaka*-delimited context slice relevant to the next reasoning step. Full context lives in external memory; the active window receives only extracted-and-tagged subsets.

**Operationalization:**

```python
# Instead of: context = full_session_history (30K tokens)
# Do:
context = extract(
    store=session_store,
    limitor=avacchedaka(qualificand="current_task_step", 
                        condition="task_type=code_refactor",
                        precision_threshold=0.8),
    max_tokens=4096
)
```

**SLIMM mapping (neuroscience):** Van Kesteren et al.'s framework separates schema-congruent encoding (neocortical, fast) from schema-incongruent encoding (hippocampal, pattern-separated). "Taking out" operationalizes the SLIMM bifurcation as an architectural primitive.

**Nyāya mapping:** The five-limbed syllogism (*pratijñā → hetu → udāharaṇa → upanaya → nigamana*) contains only what is needed to establish *vyāpti*. No extraneous propositions are admitted. "Taking out" is the formal discipline of the anumāna structure.

---

### Solution Sketch C — Principle 28: Mechanics Substitution (IFR: 2/4)

> *Replace direct token concatenation ("mechanical") with field-like influence.*

**Domain application:** Instead of literally injecting text tokens into the context window, use indirect influence mechanisms — prefix KV-cache conditioning, continuous memory vectors (AutoCompressors), or LoRA adapters trained on the relevant corpus — so that context *influences* generation without consuming attention budget.

**Vedic mapping:** *Vāsanā* as sedimented prior-cognition — the conditioning is present but not explicitly re-articulated at each moment. Prefix caching is the nearest current implementation.

**Limitation:** Principle 28 manages rather than eliminates the contradiction; field-based influence is still subject to capacity limits. Ranked lower.

---

## Contradiction 2 — Technical (Secondary: Forgetting Policy)

**Improving:** P28 — Measurement accuracy *(achieved by selective forgetting of outdated/irrelevant context)*

**Worsening:** P26 — Quantity of substance *(any forgetting risks discarding critical information)*

**Matrix lookup (P28 → P26):** Principles **2, 6, 32**

### Solution Sketch D — Principles 2 + 6 (IFR: 3/4 ★★★)

> *Extract only the necessary property + make one mechanism perform multiple functions.*

**Domain application:** A single **precision-weighted relevance score** per context element serves three functions simultaneously (Principle 6 — Universality):
1. **Retrieval gate** — only elements above threshold enter the active window
2. **Compression trigger** — elements below threshold are summarized, not deleted
3. **Forgetting schedule** — elements contradicted by later cognitions are *sublated* (bādha), not erased

```python
class ContextElement:
    content: str
    precision: float          # expected reliability, 0-1
    timestamp: datetime
    sublated_by: str | None   # reference to the later cognition that cancelled this
    avacchedaka: dict         # limitor conditions under which this is valid

# Forgetting = sublation, not deletion
def sublate(element: ContextElement, by: ContextElement) -> ContextElement:
    return dataclasses.replace(element, sublated_by=by.id, precision=0.0)
```

**Richards & Frankland mapping:** Neurogenesis-driven clearance, AMPA-receptor removal, and synaptic scaling are not deletion but *precision-downweighting* — old traces persist in degraded form as interference protection for the new. The `sublated_by` pointer preserves the trace while zeroing its influence.

**Advaita mapping:** *Bādha* (sublation) explicitly does not destroy the earlier cognition — the rope-cognition cancels the snake-cognition but the snake experience is available for reflection (*anuvyavasāya*). Typed sublation with provenance satisfies this.

### Solution Sketch E — Principle 32: Color Changes (Informational Encoding)

> *Use observable indicators to signal state under selected conditions.*

**Domain application:** Tag every context element with a **salience heatmap** — a lightweight metadata overlay that makes relevance-to-current-task observable without requiring re-reading the element. Compaction agents use the heatmap to decide what to retain.

```python
# Each context element carries a salience vector:
# [task_relevance, temporal_recency, schema_congruence, conflict_flag]
salience = SalienceVector(task=0.92, recency=0.45, schema=0.78, conflict=False)
```

**Neuroscience mapping:** Precision-weighting in predictive coding (Feldman & Friston 2010) is exactly a salience overlay — each prediction error is weighted by expected reliability before propagating. The brain doesn't re-read memory; it re-weights it.

---

## Contradiction 3 — Physical (Core Impossibility)

**Statement:** The same parameter — context content — must simultaneously be **maximally inclusive** (complete, no relevant information lost) and **maximally exclusive** (selective, irrelevant information pruned).

**Resolution:** Separation in **Space** + **System Level**

| Level | Property | Implementation |
|---|---|---|
| System (whole) | **Complete** — all information preserved | External tiered memory store (episodic DB + semantic graph + procedural CLAUDE.md) |
| Component (part) | **Selective** — only relevant slice active | Active context window with precision-weighted top-k, avacchedaka-tagged |

**This is the Complementary Learning Systems principle in architectural form:** hippocampus (complete, sparse) and neocortex (selective, distributed) are not contradictory — they operate at different system levels. MemGPT/Letta already approximates this but without the formal typing.

**Principle 6 (Universality) applied:** The boundary between system-level and component-level is mediated by a single unified precision-weighting mechanism that handles retrieval, insertion, compression, and forgetting — four functions, one interface.

---

## IFR Statement

> **The Ideal Final Result:** The context window fills itself with exactly and only what the current reasoning step requires, at zero additional attention cost, by inverting from push-based pre-loading to pull-based demand assembly — with completeness guaranteed at the system level through typed external memory, and selectivity guaranteed at the component level through avacchedaka-tagged precision-weighted retrieval.

**IFR Score: 3/4 (Near-IFR)** for the combined TC1+physical solution.

The missing criterion (leverages existing infrastructure) is partially satisfied by building on existing RAG/MemGPT/Self-RAG patterns, but the avacchedaka notation layer is genuinely new.

---

## TRIZ → Research Hypothesis Mapping

| TRIZ Principle | Research Hypothesis | Core Claim |
|---|---|---|
| P13 (Other Way Around) | H2: Precision-weighted retrieval | Pull-based, demand-driven > push-based top-k |
| P2 (Taking Out) | H3: Buddhi/manas two-stage | Extract-then-commit architecture > single-pass |
| P2 + Spatial separation | H4: Event-boundary compaction | Boundary-triggered extraction > token-threshold |
| P6 (Universality) + P2 | H5: Avacchedaka notation | Single limitor algebra > ad-hoc caveats |
| P6 (Universality) + P32 | H6: Khyātivāda classifier | Typed hallucination diagnosis > untyped correction |
| Temporal separation | H7: Adaptive forgetting | Biologically-calibrated schedule > fixed compaction |
| System-level separation | H1: Schema-congruence | Congruence ratio > raw length as predictor |

---

## Recommended Strategy

1. **Start with H5 (avacchedaka notation)** — it is the enabling primitive for H2, H3, H4, H6, H7. Without a formal context algebra, all other hypotheses remain artisanal.
2. **H3 (buddhi/manas) next** — implements the pull-based inversion (P13) at the architectural level and is testable on existing benchmarks (NoCha, HELMET).
3. **H6 (khyātivāda classifier) in parallel with H3** — creates the evaluation vocabulary needed to measure hallucination outcomes of H3.
4. **H4, H2, H1, H7** in dependency order thereafter.
5. **Frontier directions** (sākṣī witness architecture, kośa-layered representations) after H3–H5 validate the primitives.
