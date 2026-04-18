# Context Engineering Harness — User Guide

## 1. Getting Started

### Prerequisites

- Python 3.11+
- `uv` package manager (`pip install uv`)
- `claude` CLI authenticated — run `claude --version` to confirm

### Installation

```bash
git clone https://github.com/SharathSPhD/context-engineering-harness
cd context-engineering-harness
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Verify installation

```bash
pytest tests/ -m "not integration" -q   # 100+ tests should pass
python -c "from src.cli_bridge import ClaudeCLIClient; print('bridge OK')"
```

---

## 2. Configuration

All tunable parameters live in `config.toml` at the repo root. The file ships with defaults — edit only the values you want to override.

```toml
[models]
fast  = "claude-haiku-4-5"    # ManasAgent, H1/H2/H3/H7 validation calls
smart = "claude-sonnet-4-6"   # BuddhiAgent, KhyativadaClassifier

[tokens]
fast_max  = 256
smart_max = 1024

[avacchedaka]
compress_threshold          = 0.3
default_precision_threshold = 0.3

[compaction]
surprise_threshold = 0.75
token_threshold    = 500

[forgetting]
decay_factor   = 0.9
keep_threshold = 0.3
keep_newest    = 4

random_seed = 42   # invariant — do not change
```

Access config values in code:

```python
from src.config import config

model = config.fast_model          # "claude-haiku-4-5"
threshold = config.compress_threshold  # 0.3
```

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  External ContextStore (complete — "hippocampus")               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ element  │  │ element  │  │ element  │  │ sublated │       │
│  │precision │  │precision │  │precision │  │precision │       │
│  │  = 0.9  │  │  = 0.85 │  │  = 0.7  │  │  = 0.0  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└──────────────────────┬──────────────────────────────────────────┘
                       │ AvacchedakaQuery (typed boundary)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Active Context Window (selective — "neocortex")                │
│                                                                 │
│  [SAKSHI INVARIANT]  ←── frozen, never rewritten               │
│  This system withholds when evidence is insufficient.           │
│  [/SAKSHI INVARIANT]                                            │
│                                                                 │
│  ManasAgent (broad, uncommitted) ────► surfaces candidates      │
│       │                                                         │
│       ▼                                                         │
│  BuddhiAgent (narrow, decisive) ─────► answer | withhold       │
└─────────────────────────────────────────────────────────────────┘
```

**Key invariants (from `CLAUDE.md`):**
- `RANDOM_SEED=42` for all experiments
- `store.sublate()` never deletes — sets `precision=0.0` and `sublated_by`
- Log to MLflow before writing results to disk
- Default model: `claude-sonnet-4-6` (smart), `claude-haiku-4-5` (fast)

---

## 4. Core API

### ContextStore

```python
from src.avacchedaka.store import ContextStore
from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.query import AvacchedakaQuery

store = ContextStore()

# Insert a typed element
store.insert(ContextElement(
    id="doc-001",
    content="JWT tokens expire after 1 hour.",
    precision=0.95,
    avacchedaka=AvacchedakaConditions(
        qualificand="auth",       # what this is about
        qualifier="expiry",       # what property
        condition="task_type=code_review",  # when it applies
    ),
    provenance="agent2",
))

# Retrieve by avacchedaka conditions (excludes sublated elements)
query = AvacchedakaQuery(
    qualificand="auth",
    condition="task_type=code_review",
    precision_threshold=0.5,
)
results = store.retrieve(query)   # sorted by precision desc

# Sublate (bādha — invalidate without deleting)
store.sublate("doc-000", by_element_id="doc-001")
# doc-000 still exists: precision=0.0, sublated_by="doc-001"

# Compress low-precision elements at session end
store.compress(precision_threshold=0.3)
```

### Two-Stage Agent Pipeline

```python
from src.agents.orchestrator import ManusBuddhiOrchestrator

# No API key needed — uses claude CLI (config.toml controls model selection)
orch = ManusBuddhiOrchestrator(store=store)

result = orch.run(
    question="How long are JWT tokens valid?",
    task_context="task_type=code_review",
    qualificand="auth",
)

print(result.answer)           # "1 hour" or None (withheld)
print(result.confidence)       # 0.0–1.0
print(result.reasoning_trace)  # chain-of-thought
print(result.khyativada_flags) # ["asatkhyati"] if hallucination detected
```

### Hallucination Classifier

```python
from src.evaluation.khyativada import KhyativadaClassifier

clf = KhyativadaClassifier()

# Heuristic (fast, no LLM)
result = clf.classify_heuristic(
    claim="Python's GIL was removed in version 3.10",
    ground_truth="Python's GIL was removed in version 3.13",
)
# {"class": "anyathakhyati", "confidence": 0.8, "rationale": "..."}

# LLM-based (uses claude CLI, smart model)
result = clf.classify(claim="...", context="...", ground_truth="...")
```

### Precision-Weighted RAG

```python
from src.rag.precision_rag import PrecisionWeightedRAG
from src.rag.baselines import VanillaRAG

sources = [
    {"content": "24 hours", "precision": 0.3, "answer": "24 hours"},
    {"content": "1 hour",   "precision": 0.9, "answer": "1 hour"},
]

rag = PrecisionWeightedRAG()
selected = rag.select_sources(sources, top_k=1)
# selected[0]["answer"] == "1 hour"  (high precision wins)
```

---

## 5. Running the Validation Suite

### Full suite (all H1-H7)

```bash
make validate         # runs all, saves docs/validation_report.md
make validate-fast    # skips LLM calls — algorithmic tests only (H4, H5, H6)
make report           # regenerate report from last results
```

### Individual hypotheses

```bash
make validate-h1   # ~6 claude CLI calls (haiku)
make validate-h2   # ~6 claude CLI calls (haiku)
make validate-h3   # ~6 claude CLI calls (haiku + sonnet)
make validate-h4   # 0 CLI calls (library test)
make validate-h5   # 0 CLI calls (library test)
make validate-h6   # 0 CLI calls (heuristic classifier)
make validate-h7   # ~3 claude CLI calls (haiku)
```

### Reading the report

After `make validate`, open `docs/validation_report.md`:

```
| H1 | congruent=60%, incongruent=100%, delta=-40% | ✅ PASS |
| H2 | precision_rag=100%, vanilla=50%              | ✅ PASS |
...
```

---

## 6. Using Claude CLI vs API Key

By default, all agents use the CLI bridge (no API key):

```python
orch = ManusBuddhiOrchestrator()          # uses claude CLI
orch = ManusBuddhiOrchestrator(api_key="sk-ant-...") # uses Anthropic SDK
```

For the original MLflow experiment runners (`make reproduce-h{1-7}`), set:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
make reproduce-h1
```

---

## 7. Extending: Adding a New Hypothesis

1. **Write benchmark data** in `experiments/validate/data.py`
2. **Create the validation module** `experiments/validate/h8_mytest.py`:
   ```python
   def run_h8() -> dict:
       return {
           "hypothesis": "H8",
           "description": "My new test",
           "my_metric": 0.85,
           "target_met": True,
           "target_description": "my_metric >= 0.80",
       }
   ```
3. **Register in runner** `experiments/validate/runner.py`
4. **Add Makefile target**: `validate-h8: .venv/bin/python experiments/validate/h8_mytest.py`
5. **Write tests** in `tests/test_validate/test_h8.py`

---

## 8. Avacchedaka Notation Reference

Every `ContextElement` carries `AvacchedakaConditions`:

| Field | Meaning | Example |
|---|---|---|
| `qualificand` | What entity this describes | `"auth"`, `"database"` |
| `qualifier` | What property is asserted | `"expiry"`, `"version"` |
| `condition` | When this assertion applies | `"task_type=code_review"` |
| `relation` | How qualifier relates to qualificand | `"inherence"` (default) |

`AvacchedakaQuery` matches elements where:
1. `element.avacchedaka.qualificand == query.qualificand`
2. All AND-tokens in `query.condition` appear in `element.avacchedaka.condition`
3. `element.precision >= query.precision_threshold`
4. `element.sublated_by is None`
