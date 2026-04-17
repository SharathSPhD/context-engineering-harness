# Technical Specification
## Context Engineering Synthesis Framework

**Version:** 0.1 — 2026-04-17  
**Status:** Draft

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT ENGINEERING SYNTHESIS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │   External   │    │         ACTIVE CONTEXT WINDOW         │   │
│  │   Memory     │    │  ┌────────────┐  ┌─────────────────┐ │   │
│  │  (Complete)  │◄──►│  │   sākṣī    │  │  Avacchedaka-   │ │   │
│  │              │    │  │  (witness) │  │  tagged elements │ │   │
│  │ • Episodic   │    │  │  prefix    │  │  precision ≥ θ   │ │   │
│  │   vector DB  │    │  └────────────┘  └─────────────────┘ │   │
│  │ • Semantic   │    │                                        │   │
│  │   graph      │    │  ┌──────────────────────────────────┐ │   │
│  │ • Procedural │    │  │         ManasAgent               │ │   │
│  │   CLAUDE.md  │    │  │  (broad attention, no commit)    │ │   │
│  └──────────────┘    │  └──────────────┬───────────────────┘ │   │
│         ▲            │                  │ ManasOutput          │   │
│         │ retrieve   │  ┌──────────────▼───────────────────┐ │   │
│         │ (avacchedaka│  │         BuddhiAgent              │ │   │
│         │  query)    │  │  (narrow attention, commits)     │ │   │
│         └────────────┤  │  sublates candidates, withholds  │ │   │
│                      │  └──────────────────────────────────┘ │   │
│                      └──────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │   EVALUATION LAYER                                       │    │
│  │   KhyātivādaClassifier | EventBoundaryDetector          │    │
│  │   PrecisionWeightedRAG | AdaptiveForgettingSchedule      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- External memory is **complete** (separation in space — system level)
- Active window is **selective** (separation in space — component level)
- The avacchedaka layer is the typed boundary between them
- ManasAgent and BuddhiAgent use the SAME base model — architectural isolation only
- The sākṣī prefix is NEVER rewritten by context operations (witness-invariant)

---

## 2. Directory Structure

```
context/
├── docs/
│   ├── research.md              # Source synthesis (read-only)
│   ├── prd.md                   # Product requirements
│   ├── spec.md                  # This document
│   ├── triz.md                  # TRIZ contradiction analysis
│   └── superpowers/
│       └── plans/
│           └── 2026-04-17-context-engineering-synthesis.md
├── src/
│   ├── avacchedaka/             # F1: Core context algebra library
│   │   ├── __init__.py
│   │   ├── element.py           # ContextElement dataclass
│   │   ├── store.py             # ContextStore with typed operations
│   │   ├── query.py             # AvacchedakaQuery builder
│   │   └── schema.py            # JSON schema for API integration
│   ├── agents/                  # F2: Buddhi/manas architecture
│   │   ├── __init__.py
│   │   ├── manas.py             # ManasAgent
│   │   ├── buddhi.py            # BuddhiAgent
│   │   ├── orchestrator.py      # ManasOrchestratorBuddhi pipeline
│   │   └── sakshi.py            # Witness-invariant prefix manager
│   ├── rag/                     # F4: Precision-weighted RAG
│   │   ├── __init__.py
│   │   ├── precision_rag.py     # Bayesian precision-weighted retrieval
│   │   ├── conflicting_qa.py    # Conflicting-source QA benchmark builder
│   │   └── baselines.py         # Vanilla RAG, Self-RAG wrappers
│   ├── forgetting/              # F7: Adaptive forgetting schedules
│   │   ├── __init__.py
│   │   ├── schedules.py         # All 5 schedule variants
│   │   └── distribution_shift.py # Benchmark with controlled shifts
│   ├── compaction/              # F5: Event-boundary compaction
│   │   ├── __init__.py
│   │   ├── detector.py          # EventBoundaryDetector
│   │   └── compactor.py         # BoundaryTriggeredCompactor
│   └── evaluation/              # F3, F6: Evaluation systems
│       ├── __init__.py
│       ├── khyativada.py        # Hallucination classifier (6-class)
│       ├── schema_congruence.py # Congruence-controlled benchmark builder
│       ├── benchmarks.py        # RULER extension, withhold-or-answer
│       └── metrics.py           # ECE, conflict-rate, task-success-rate
├── tests/
│   ├── test_avacchedaka/
│   ├── test_agents/
│   ├── test_rag/
│   ├── test_forgetting/
│   ├── test_compaction/
│   └── test_evaluation/
├── experiments/
│   ├── h1_schema_congruence/
│   ├── h2_precision_rag/
│   ├── h3_buddhi_manas/
│   ├── h4_event_boundary/
│   ├── h5_avacchedaka_multiagent/
│   ├── h6_khyativada_classifier/
│   └── h7_adaptive_forgetting/
├── data/
│   ├── annotations/             # Khyātivāda annotation data
│   ├── benchmarks/              # Benchmark datasets
│   └── experiments/             # MLflow experiment artifacts
├── .triz/
│   └── session.jsonl            # TRIZ analysis log
├── pyproject.toml
├── Makefile
└── CLAUDE.md                    # Procedural memory for agents
```

---

## 3. Core Data Structures

### 3.1 ContextElement

```python
# src/avacchedaka/element.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class AvacchedakaConditions:
    """Limitor conditions specifying when this element is valid."""
    qualificand: str           # The entity this element describes
    qualifier: str             # The property being attributed
    condition: str             # Logical condition string (e.g., "task_type=code_refactor")
    relation: str = "inherence"  # samavāya / contact / conjunction

@dataclass
class ContextElement:
    id: str
    content: str
    precision: float           # Expected reliability, 0.0–1.0
    avacchedaka: AvacchedakaConditions
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance: str = ""       # Source: retrieved_doc, agent_observation, user_input, etc.
    sublated_by: str | None = None  # ID of the element that cancelled this
    salience: dict[str, float] = field(default_factory=dict)
    # salience keys: task_relevance, temporal_recency, schema_congruence, conflict_flag
```

### 3.2 ContextStore

```python
# src/avacchedaka/store.py
class ContextStore:
    def insert(self, element: ContextElement) -> None: ...
    
    def retrieve(
        self,
        query: AvacchedakaQuery,
        max_elements: int = 20,
        precision_threshold: float = 0.5,
    ) -> list[ContextElement]:
        """Returns elements sorted by precision * task_relevance_score,
        filtered by avacchedaka condition matching and precision threshold.
        Never returns sublated elements (sublated_by is not None)."""
        ...
    
    def sublate(self, element_id: str, by_element_id: str) -> None:
        """Typed sublation: sets element.sublated_by = by_element_id,
        element.precision = 0.0. Does NOT delete the element."""
        ...
    
    def compress(self, precision_threshold: float = 0.3) -> list[str]:
        """Summarize elements below threshold. Returns IDs of compressed elements."""
        ...
    
    def to_context_window(
        self,
        query: AvacchedakaQuery,
        max_tokens: int = 4096,
    ) -> str:
        """Assembles a context string from retrieved elements,
        formatted for injection into a Claude API message."""
        ...
```

### 3.3 ManasOutput / BuddhiOutput

```python
# src/agents/manas.py
@dataclass
class ManasOutput:
    candidates: list[ContextElement]       # Surfaced candidates, uncommitted
    uncertainty: float                      # 0–1
    recommended_retrieval: list[AvacchedakaQuery]  # Pull-based requests
    reasoning_sketch: str                   # Rough draft, not committed

# src/agents/buddhi.py
@dataclass
class BuddhiOutput:
    answer: str | None                     # None = withhold (confidence too low)
    confidence: float                       # 0–1
    sublated: list[str]                    # IDs of candidates buddhi rejected
    reasoning_trace: str                   # Full reasoning, for audit
    khyativada_flags: list[str]            # Any detected hallucination type
```

---

## 4. API Design: Avacchedaka Query Format

The avacchedaka notation is also exposed as a JSON schema compatible with Claude API tool_use / structured output:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema",
  "title": "AvacchedakaQuery",
  "type": "object",
  "required": ["qualificand", "condition"],
  "properties": {
    "qualificand": {
      "type": "string",
      "description": "The entity or topic this retrieval concerns"
    },
    "qualifier": {
      "type": "string",
      "description": "The property being sought"
    },
    "condition": {
      "type": "string",
      "description": "Logical condition string, e.g. 'task_type=code_refactor AND file=auth.py'"
    },
    "precision_threshold": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.5
    },
    "max_elements": {
      "type": "integer",
      "default": 10
    }
  }
}
```

---

## 5. Khyātivāda Hallucination Ontology

| Class | Sanskrit Term | Description | Signature | Primary Mitigation |
|---|---|---|---|---|
| `anyathakhyati` | anyathākhyāti | Misidentifying one real entity as another | Factual confusion between real entities | Retrieval (correct entity grounding) |
| `atmakhyati` | ātmakhyāti | Projecting internal pattern as external fact | Confident confabulation not in source | Calibration training (uncertainty) |
| `anirvacaniyakhyati` | anirvacanīyakhyāti | Novel confabulation — neither real nor unreal | Plausible but invented content | Constrained decoding, grounding |
| `asatkhyati` | asatkhyāti | Hallucinating pure non-being | Asserting nonexistent entities/events | Existence verification retrieval |
| `viparitakhyati` | viparītakhyāti | Inverted identification | Systematic reversal (A↔B confusion) | Contrastive retrieval |
| `akhyati` | akhyāti | Two true cognitions combined wrongly | True components, false combination | Structural/relational grounding |

### Classifier Architecture

```python
# src/evaluation/khyativada.py
class KhyativadaClassifier:
    """6-class classifier over hallucination types.
    Uses Claude claude-sonnet-4-6 with structured output for inference;
    lightweight sklearn LogReg over embeddings for fast batch scoring."""
    
    CLASSES = [
        "anyathakhyati",
        "atmakhyati", 
        "anirvacaniyakhyati",
        "asatkhyati",
        "viparitakhyati",
        "akhyati",
    ]
    
    def classify(self, claim: str, context: str, ground_truth: str) -> dict:
        """Returns {'class': str, 'confidence': float, 'rationale': str}"""
        ...
    
    def batch_classify(self, examples: list[dict]) -> list[dict]: ...
```

---

## 6. Experimental Protocol

### Shared Infrastructure

All 7 hypothesis experiments use:
- **MLflow** for experiment tracking (run `mlflow ui` to browse)
- **Reproducibility seed:** `RANDOM_SEED=42` in all experiments
- **Model:** `claude-sonnet-4-6` for all Claude API calls unless noted
- **Prompt caching:** prefix caching enabled for shared system prompts
- **Cost tracking:** `anthropic.usage` logged per run

### Experiment Entry Point Pattern

```python
# experiments/h{n}_*/run.py
import mlflow

def main():
    mlflow.set_experiment("hypothesis-{n}")
    with mlflow.start_run():
        mlflow.log_params({
            "model": MODEL,
            "context_length": CONTEXT_LENGTH,
            "hypothesis": "H{n}",
        })
        results = run_experiment()
        mlflow.log_metrics(results)
        mlflow.log_artifact("results.json")

if __name__ == "__main__":
    main()
```

### Make Targets

```makefile
# Makefile
reproduce-h1:  ## Reproduce schema-congruence context rot experiment
reproduce-h2:  ## Reproduce precision-weighted RAG experiment
reproduce-h3:  ## Reproduce buddhi/manas architecture experiment
reproduce-h4:  ## Reproduce event-boundary compaction experiment
reproduce-h5:  ## Reproduce avacchedaka multi-agent experiment
reproduce-h6:  ## Reproduce khyātivāda classifier experiment
reproduce-h7:  ## Reproduce adaptive forgetting experiment

test:          ## Run full test suite
test-unit:     ## Unit tests only
test-integration:  ## Integration tests (requires ANTHROPIC_API_KEY)
coverage:      ## Run tests with coverage report

annotate-h6:   ## Launch khyātivāda annotation interface
build-benchmarks:  ## Build all benchmark datasets
```

---

## 7. CLAUDE.md (Procedural Memory for Agents)

```markdown
# Context Engineering Synthesis — Agent Instructions

## Project context
This is a research project validating 7 hypotheses about context engineering
at the intersection of LLM systems, neuroscience, and Vedic epistemology.

## Key invariants (sākṣī — never overwrite)
- All experiments use RANDOM_SEED=42
- Never delete annotated data in data/annotations/
- Never modify benchmark splits after they are generated
- Always log to MLflow before writing results to disk

## Memory operations
- Use avacchedaka queries (see docs/spec.md §4) for context retrieval
- When contradicting stored memory, call sublate() — do NOT overwrite
- Compress elements below precision=0.3 at end of each session

## Current hypotheses under test
See docs/prd.md §4 for full feature requirements per hypothesis.
```

---

## 8. Testing Strategy

### Unit tests
Every public function in `src/avacchedaka/` has:
- Happy-path test
- Precision-threshold boundary test
- Sublation preserves element (does not delete) test
- Avacchedaka condition matching test

### Integration tests
- `test_manas_buddhi_pipeline`: full ManasAgent → BuddhiAgent run on 5-hop QA
- `test_avacchedaka_multiagent`: 2-agent coordination using avacchedaka queries, measure conflict rate
- `test_khyativada_end_to_end`: inject a known `anyathakhyati` hallucination, verify classifier labels it correctly

### Benchmark tests
- `test_h3_noCha`: buddhi/manas vs. single-stage on NoCha global narrative comprehension
- `test_h2_conflicting_sources`: precision RAG vs. top-k on 50 conflicting-source examples
- `test_h4_compaction_budget`: boundary-triggered vs. threshold-triggered at 80% token budget

---

## 9. Neuroscience ↔ Vedic ↔ LLM Mapping (Formal)

This table is the ground truth for naming conventions across the codebase:

| LLM Component | Neuroscience | Vedic | Code Name |
|---|---|---|---|
| Active context window | Episodic buffer + WM persistent activity (dlPFC) | *Antaḥkaraṇa* as single *manas*-gated buffer | `active_window` |
| External memory store | Hippocampal episodic + semantic memory | *Smṛti* (memory store) | `ContextStore` |
| System prompt / persona | Task set in dlPFC (Miller & Cohen 2001) | *Saṃskāra / Vāsanā* | `sakshi_prefix` |
| Pull-based retrieval | Biased competition top-down signal | *Pratyabhijñā* (recognition-retrieval) | `AvacchedakaQuery` |
| Fast broad-attention stage | Construction phase (Kintsch 1988) | *Manas* (indecisive, sensory) | `ManasAgent` |
| Slow narrow-commit stage | Integration phase (Kintsch 1988) | *Buddhi* (discriminative, decisive) | `BuddhiAgent` |
| Typed sublation | Active forgetting via precision-downweighting | *Bādha* (sublation) | `store.sublate()` |
| Hallucination type | Confabulation under weak priors / schema intrusion | *Adhyāsa / Khyātivāda* | `KhyativadaClass` |
| Precision weight | Precision-weighted prediction error (Friston 2010) | *Avacchedaka* (limitor) | `element.precision` |
| Witness-invariant summary | Stable PFC schema (Bartlett 1932; Ghosh & Gilboa 2014) | *Sākṣī* (witness consciousness) | `SakshiPrefix` |
| Event-boundary detection | Prediction failure = segment boundary (Zacks 2007) | *Tarka* (hypothetical reasoning to defend vyāpti) | `EventBoundaryDetector` |
| Compaction at boundaries | Hippocampal SWR replay (Buzsáki 2015) | *Viveka* (discrimination, selecting what to retain) | `BoundaryTriggeredCompactor` |
