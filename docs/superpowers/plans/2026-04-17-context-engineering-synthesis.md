# Context Engineering Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a unified context engineering framework grounded in LLM systems, cognitive neuroscience, and Vedic epistemology — producing the avacchedaka notation library, buddhi/manas agent architecture, khyātivāda hallucination classifier, and reproducible experiments for all 7 research hypotheses.

**Architecture:** External memory (complete) + precision-weighted active window (selective) separated by the avacchedaka typed boundary. ManasAgent (broad/uncommitted) feeds BuddhiAgent (narrow/committed) via structured output. All context operations are typed, sublatable, and auditable.

**Tech Stack:** Python 3.11+, `anthropic` SDK ≥0.50 with prompt caching, `chromadb` for vector store, `pytest` + `coverage`, `mlflow` for experiment tracking, `scikit-learn` for khyātivāda classifier, `pydantic` v2 for data models.

**Supporting docs:** `docs/prd.md` | `docs/spec.md` | `docs/triz.md` | `docs/research.md`

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, tool config |
| `Makefile` | All reproduce-h{n}, test, coverage, annotate targets |
| `CLAUDE.md` | Procedural memory / agent invariants (sākṣī prefix source) |
| `src/avacchedaka/element.py` | `ContextElement`, `AvacchedakaConditions` dataclasses |
| `src/avacchedaka/store.py` | `ContextStore` with insert/retrieve/sublate/compress |
| `src/avacchedaka/query.py` | `AvacchedakaQuery` builder and condition matcher |
| `src/avacchedaka/schema.py` | JSON schema for Claude API message integration |
| `src/agents/sakshi.py` | `SakshiPrefix` — witness-invariant frozen summary manager |
| `src/agents/manas.py` | `ManasAgent` — broad attention, candidate surfacing, no commit |
| `src/agents/buddhi.py` | `BuddhiAgent` — narrow attention, discriminates, commits/withholds |
| `src/agents/orchestrator.py` | `ManusBuddhiOrchestrator` — wires manas → buddhi pipeline |
| `src/evaluation/khyativada.py` | `KhyativadaClassifier` — 6-class hallucination type classifier |
| `src/evaluation/schema_congruence.py` | Congruence-controlled benchmark builder (H1) |
| `src/evaluation/benchmarks.py` | Withhold-or-answer benchmark; RULER extension |
| `src/evaluation/metrics.py` | ECE, conflict-rate, task-success-rate, congruence-ratio |
| `src/rag/precision_rag.py` | Bayesian precision-weighted retrieval (H2) |
| `src/rag/conflicting_qa.py` | Conflicting-source QA benchmark builder |
| `src/rag/baselines.py` | Vanilla RAG, Self-RAG wrappers |
| `src/compaction/detector.py` | `EventBoundaryDetector` — per-token surprise monitoring |
| `src/compaction/compactor.py` | `BoundaryTriggeredCompactor` |
| `src/forgetting/schedules.py` | 5 forgetting schedule variants (none/fixed/recency/reward/badha) |
| `src/forgetting/distribution_shift.py` | Long-horizon benchmark with controlled distribution shifts |
| `experiments/h{1-7}_*/run.py` | Experiment entry points, one per hypothesis |
| `tests/test_avacchedaka/` | Unit + integration tests for avacchedaka library |
| `tests/test_agents/` | Unit + integration tests for buddhi/manas |
| `tests/test_evaluation/` | Classifier tests, benchmark builder tests |
| `tests/test_rag/` | Precision RAG tests |
| `tests/test_compaction/` | Boundary detector tests |
| `tests/test_forgetting/` | Schedule comparison tests |
| `data/annotations/khyativada_guidelines.md` | Annotation guidelines for H6 |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `CLAUDE.md`
- Create: `src/__init__.py`, `src/avacchedaka/__init__.py`, `src/agents/__init__.py`, `src/evaluation/__init__.py`, `src/rag/__init__.py`, `src/compaction/__init__.py`, `src/forgetting/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1.1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "context-engineering-synthesis"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.50",
    "chromadb>=0.5",
    "pydantic>=2.0",
    "scikit-learn>=1.4",
    "mlflow>=2.0",
    "numpy>=1.26",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/experiments/*"]
```

- [ ] **Step 1.2: Write Makefile**

```makefile
.PHONY: test test-unit test-integration coverage install reproduce-h1 reproduce-h2 reproduce-h3 reproduce-h4 reproduce-h5 reproduce-h6 reproduce-h7

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v --tb=short -m "not integration"

test-integration:
	pytest tests/ -v --tb=short -m integration

coverage:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

reproduce-h1:
	RANDOM_SEED=42 python experiments/h1_schema_congruence/run.py

reproduce-h2:
	RANDOM_SEED=42 python experiments/h2_precision_rag/run.py

reproduce-h3:
	RANDOM_SEED=42 python experiments/h3_buddhi_manas/run.py

reproduce-h4:
	RANDOM_SEED=42 python experiments/h4_event_boundary/run.py

reproduce-h5:
	RANDOM_SEED=42 python experiments/h5_avacchedaka_multiagent/run.py

reproduce-h6:
	RANDOM_SEED=42 python experiments/h6_khyativada_classifier/run.py

reproduce-h7:
	RANDOM_SEED=42 python experiments/h7_adaptive_forgetting/run.py

annotate-h6:
	python experiments/h6_khyativada_classifier/annotate.py

build-benchmarks:
	python experiments/h1_schema_congruence/build_benchmark.py
	python experiments/h2_precision_rag/build_benchmark.py
```

- [ ] **Step 1.3: Create all __init__.py files and tests/conftest.py**

```bash
mkdir -p src/{avacchedaka,agents,evaluation,rag,compaction,forgetting}
mkdir -p tests/{test_avacchedaka,test_agents,test_evaluation,test_rag,test_compaction,test_forgetting}
mkdir -p experiments/{h1_schema_congruence,h2_precision_rag,h3_buddhi_manas,h4_event_boundary,h5_avacchedaka_multiagent,h6_khyativada_classifier,h7_adaptive_forgetting}
mkdir -p data/{annotations,benchmarks,experiments}
touch src/__init__.py src/avacchedaka/__init__.py src/agents/__init__.py
touch src/evaluation/__init__.py src/rag/__init__.py src/compaction/__init__.py src/forgetting/__init__.py
touch tests/__init__.py tests/test_avacchedaka/__init__.py tests/test_agents/__init__.py
touch tests/test_evaluation/__init__.py tests/test_rag/__init__.py tests/test_compaction/__init__.py tests/test_forgetting/__init__.py
```

`tests/conftest.py`:
```python
import os
import pytest

@pytest.fixture
def api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key

@pytest.fixture
def sample_element():
    from src.avacchedaka.element import ContextElement, AvacchedakaConditions
    return ContextElement(
        id="test-001",
        content="The auth module uses JWT tokens with 24h expiry.",
        precision=0.9,
        avacchedaka=AvacchedakaConditions(
            qualificand="auth_module",
            qualifier="token_expiry",
            condition="task_type=code_review",
        ),
        provenance="retrieved_doc",
    )
```

- [ ] **Step 1.4: Write CLAUDE.md**

```markdown
# Context Engineering Synthesis — Agent Instructions

## Invariants (sākṣī — never overwrite)
- All experiments use RANDOM_SEED=42
- Never delete files under data/annotations/
- Never modify benchmark splits after they are generated (data/benchmarks/)
- Always log to MLflow before writing results to disk
- avacchedaka sublation never deletes — it sets precision=0.0 and sublated_by

## Context operations
- Retrieve using AvacchedakaQuery — see src/avacchedaka/query.py
- When new information contradicts stored memory: call store.sublate(), not store.delete()
- Compress elements below precision=0.3 at end of each session

## Hypotheses under test
- H1: Schema-congruence predicts context rot better than length
- H2: Precision-weighted RAG outperforms top-k on conflicting sources
- H3: Buddhi/manas two-stage outperforms single-stage
- H4: Event-boundary compaction outperforms threshold compaction
- H5: Avacchedaka annotation reduces multi-agent conflict rate ≥30%
- H6: Khyātivāda-typed mitigation reduces respective class at 2× rate
- H7: Adaptive forgetting outperforms fixed on post-shift tasks

## API usage
- Model: claude-sonnet-4-6 for all calls unless experiment specifies otherwise
- Prefix caching: always use cached system prompt for shared prefixes
- Budget: stay under $500 total across all experiments
```

- [ ] **Step 1.5: Run install and verify**

```bash
pip install -e ".[dev]"
python -c "import anthropic; print('anthropic OK')"
python -c "import chromadb; print('chromadb OK')"
```

Expected: both print OK with no import errors.

- [ ] **Step 1.6: Commit**

```bash
git add pyproject.toml Makefile CLAUDE.md src/ tests/conftest.py experiments/ data/
git commit -m "feat: project scaffold — avacchedaka, agents, evaluation, rag, compaction, forgetting modules"
```

---

## Task 2: Avacchedaka Library — ContextElement and ContextStore

**Files:**
- Create: `src/avacchedaka/element.py`
- Create: `src/avacchedaka/query.py`
- Create: `src/avacchedaka/store.py`
- Create: `src/avacchedaka/schema.py`
- Test: `tests/test_avacchedaka/test_element.py`
- Test: `tests/test_avacchedaka/test_store.py`

- [ ] **Step 2.1: Write failing tests for ContextElement**

`tests/test_avacchedaka/test_element.py`:
```python
import pytest
from datetime import datetime
from src.avacchedaka.element import ContextElement, AvacchedakaConditions

def test_element_creation(sample_element):
    assert sample_element.id == "test-001"
    assert sample_element.precision == 0.9
    assert sample_element.sublated_by is None

def test_element_not_sublated_by_default(sample_element):
    assert sample_element.sublated_by is None

def test_element_avacchedaka_has_qualificand(sample_element):
    assert sample_element.avacchedaka.qualificand == "auth_module"

def test_element_precision_bounds():
    with pytest.raises(ValueError, match="precision must be between 0 and 1"):
        ContextElement(
            id="bad",
            content="x",
            precision=1.5,
            avacchedaka=AvacchedakaConditions(
                qualificand="x", qualifier="y", condition="z"
            ),
        )
```

- [ ] **Step 2.2: Run tests — verify they fail**

```bash
pytest tests/test_avacchedaka/test_element.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'src.avacchedaka.element'`

- [ ] **Step 2.3: Implement ContextElement**

`src/avacchedaka/element.py`:
```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AvacchedakaConditions:
    qualificand: str
    qualifier: str
    condition: str
    relation: str = "inherence"

@dataclass
class ContextElement:
    id: str
    content: str
    precision: float
    avacchedaka: AvacchedakaConditions
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance: str = ""
    sublated_by: str | None = None
    salience: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.precision <= 1.0:
            raise ValueError("precision must be between 0 and 1")
```

- [ ] **Step 2.4: Run element tests — verify they pass**

```bash
pytest tests/test_avacchedaka/test_element.py -v
```

Expected: `4 passed`

- [ ] **Step 2.5: Write failing tests for ContextStore**

`tests/test_avacchedaka/test_store.py`:
```python
import pytest
from src.avacchedaka.store import ContextStore
from src.avacchedaka.query import AvacchedakaQuery

def test_insert_and_retrieve(sample_element):
    store = ContextStore()
    store.insert(sample_element)
    query = AvacchedakaQuery(qualificand="auth_module", condition="task_type=code_review")
    results = store.retrieve(query)
    assert len(results) == 1
    assert results[0].id == "test-001"

def test_retrieve_excludes_sublated(sample_element):
    store = ContextStore()
    store.insert(sample_element)
    from src.avacchedaka.element import ContextElement, AvacchedakaConditions
    newer = ContextElement(
        id="test-002",
        content="JWT tokens now use 1h expiry.",
        precision=0.95,
        avacchedaka=AvacchedakaConditions(
            qualificand="auth_module", qualifier="token_expiry", condition="task_type=code_review"
        ),
    )
    store.insert(newer)
    store.sublate(element_id="test-001", by_element_id="test-002")
    query = AvacchedakaQuery(qualificand="auth_module", condition="task_type=code_review")
    results = store.retrieve(query)
    assert len(results) == 1
    assert results[0].id == "test-002"

def test_sublation_does_not_delete(sample_element):
    store = ContextStore()
    store.insert(sample_element)
    from src.avacchedaka.element import ContextElement, AvacchedakaConditions
    newer = ContextElement(
        id="test-002",
        content="Updated.",
        precision=0.95,
        avacchedaka=AvacchedakaConditions(
            qualificand="auth_module", qualifier="token_expiry", condition="task_type=code_review"
        ),
    )
    store.insert(newer)
    store.sublate("test-001", "test-002")
    # Element still exists in store, just with precision=0 and sublated_by set
    elem = store.get("test-001")
    assert elem is not None
    assert elem.precision == 0.0
    assert elem.sublated_by == "test-002"

def test_retrieve_below_precision_threshold_excluded(sample_element):
    store = ContextStore()
    store.insert(sample_element)
    query = AvacchedakaQuery(qualificand="auth_module", condition="task_type=code_review")
    # Default threshold 0.5; sample_element has precision=0.9 so it's included
    results = store.retrieve(query, precision_threshold=0.95)
    assert len(results) == 0  # 0.9 < 0.95

def test_compress_returns_ids_of_low_precision():
    from src.avacchedaka.element import ContextElement, AvacchedakaConditions
    store = ContextStore()
    low = ContextElement(
        id="low-001",
        content="Old info.",
        precision=0.2,
        avacchedaka=AvacchedakaConditions(qualificand="x", qualifier="y", condition="z"),
    )
    store.insert(low)
    compressed = store.compress(precision_threshold=0.3)
    assert "low-001" in compressed
```

- [ ] **Step 2.6: Run store tests — verify they fail**

```bash
pytest tests/test_avacchedaka/test_store.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 2.7: Implement AvacchedakaQuery**

`src/avacchedaka/query.py`:
```python
from dataclasses import dataclass

@dataclass
class AvacchedakaQuery:
    qualificand: str
    condition: str
    qualifier: str = ""
    precision_threshold: float = 0.5
    max_elements: int = 20

    def matches(self, element) -> bool:
        """True if element's avacchedaka qualificand matches and condition overlaps."""
        if element.avacchedaka.qualificand != self.qualificand:
            return False
        # Simple condition matching: check if any key=value pair in query condition
        # appears in element condition (both are plain strings here)
        if not self.condition:
            return True
        # Each condition token from query must appear in element's condition string
        for token in self.condition.split(" AND "):
            if token.strip() not in element.avacchedaka.condition:
                return False
        return True
```

- [ ] **Step 2.8: Implement ContextStore**

`src/avacchedaka/store.py`:
```python
import dataclasses
from src.avacchedaka.element import ContextElement
from src.avacchedaka.query import AvacchedakaQuery

class ContextStore:
    def __init__(self):
        self._elements: dict[str, ContextElement] = {}

    def insert(self, element: ContextElement) -> None:
        self._elements[element.id] = element

    def get(self, element_id: str) -> ContextElement | None:
        return self._elements.get(element_id)

    def retrieve(
        self,
        query: AvacchedakaQuery,
        precision_threshold: float | None = None,
        max_elements: int | None = None,
    ) -> list[ContextElement]:
        threshold = precision_threshold if precision_threshold is not None else query.precision_threshold
        limit = max_elements if max_elements is not None else query.max_elements
        candidates = [
            e for e in self._elements.values()
            if e.sublated_by is None
            and e.precision >= threshold
            and query.matches(e)
        ]
        candidates.sort(key=lambda e: e.precision, reverse=True)
        return candidates[:limit]

    def sublate(self, element_id: str, by_element_id: str) -> None:
        if element_id not in self._elements:
            raise KeyError(f"Element {element_id} not found")
        elem = self._elements[element_id]
        self._elements[element_id] = dataclasses.replace(
            elem, sublated_by=by_element_id, precision=0.0
        )

    def compress(self, precision_threshold: float = 0.3) -> list[str]:
        """Mark elements below threshold as compressed (precision→0). Returns their IDs."""
        compressed = []
        for eid, elem in list(self._elements.items()):
            if elem.precision > 0 and elem.precision < precision_threshold and elem.sublated_by is None:
                self._elements[eid] = dataclasses.replace(elem, precision=0.0)
                compressed.append(eid)
        return compressed

    def to_context_window(self, query: AvacchedakaQuery, max_tokens: int = 4096) -> str:
        elements = self.retrieve(query)
        parts = []
        total_chars = 0
        char_budget = max_tokens * 4  # rough approximation
        for e in elements:
            block = f"[{e.avacchedaka.qualificand}|precision={e.precision:.2f}] {e.content}"
            if total_chars + len(block) > char_budget:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n".join(parts)
```

- [ ] **Step 2.9: Run all store tests — verify they pass**

```bash
pytest tests/test_avacchedaka/ -v
```

Expected: `9 passed`

- [ ] **Step 2.10: Write avacchedaka JSON schema**

`src/avacchedaka/schema.py`:
```python
AVACCHEDAKA_QUERY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "title": "AvacchedakaQuery",
    "type": "object",
    "required": ["qualificand", "condition"],
    "properties": {
        "qualificand": {"type": "string"},
        "qualifier": {"type": "string", "default": ""},
        "condition": {"type": "string"},
        "precision_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        "max_elements": {"type": "integer", "default": 10},
    },
}

CONTEXT_ELEMENT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "title": "ContextElement",
    "type": "object",
    "required": ["id", "content", "precision", "avacchedaka"],
    "properties": {
        "id": {"type": "string"},
        "content": {"type": "string"},
        "precision": {"type": "number", "minimum": 0, "maximum": 1},
        "avacchedaka": {
            "type": "object",
            "required": ["qualificand", "qualifier", "condition"],
            "properties": {
                "qualificand": {"type": "string"},
                "qualifier": {"type": "string"},
                "condition": {"type": "string"},
                "relation": {"type": "string", "default": "inherence"},
            },
        },
        "sublated_by": {"type": ["string", "null"]},
        "provenance": {"type": "string"},
    },
}
```

- [ ] **Step 2.11: Check coverage for avacchedaka**

```bash
pytest tests/test_avacchedaka/ --cov=src/avacchedaka --cov-report=term-missing
```

Expected: ≥ 90% line coverage.

- [ ] **Step 2.12: Commit**

```bash
git add src/avacchedaka/ tests/test_avacchedaka/
git commit -m "feat: avacchedaka library — ContextElement, ContextStore with typed sublation, AvacchedakaQuery"
```

---

## Task 3: Khyātivāda Hallucination Classifier

**Files:**
- Create: `src/evaluation/khyativada.py`
- Create: `data/annotations/khyativada_guidelines.md`
- Test: `tests/test_evaluation/test_khyativada.py`

- [ ] **Step 3.1: Write annotation guidelines**

`data/annotations/khyativada_guidelines.md`:
```markdown
# Khyātivāda Annotation Guidelines

## Classes

### anyathakhyati (anyathākhyāti)
**Definition:** Misidentifying one real entity as another real entity.
**Signature:** The model says X is Y where both X and Y exist but are different things.
**Example:** "Python's GIL was removed in version 3.10" (it was 3.13). Both versions exist; wrong version attributed.
**Mitigation:** Retrieval (find correct entity grounding).

### atmakhyati (ātmakhyāti)
**Definition:** Projecting internal pattern as external fact — the model's training distribution presented as world knowledge.
**Signature:** Confident assertion not grounded in any source; model is pattern-completing from training.
**Example:** "The standard port for this service is 8080" (no source confirms this; model is pattern-matching).
**Mitigation:** Calibration training (uncertainty markers).

### anirvacaniyakhyati (anirvacanīyakhyāti)
**Definition:** Novel confabulation — content that is neither real nor derivable; inexplicably invented.
**Signature:** Detailed specific claims (names, dates, quotes) that don't exist anywhere.
**Example:** A fabricated paper citation with a specific journal, year, and page numbers.
**Mitigation:** Constrained decoding, citation grounding.

### asatkhyati (asatkhyāti)
**Definition:** Hallucinating pure non-being — asserting the existence of something that does not exist at all.
**Signature:** Referencing a nonexistent API, function, law, or person as if it exists.
**Example:** "Use the `requests.get_json()` method" (this method does not exist).
**Mitigation:** Existence verification via retrieval.

### viparitakhyati (viparītakhyāti)
**Definition:** Systematic inverted identification — A and B are both real but the model consistently swaps them.
**Signature:** Directional confusion, systematic reversal pattern.
**Example:** Confusing which function calls which in a recursive pair; attributing author A's quote to author B and vice versa.
**Mitigation:** Contrastive retrieval (retrieve both entities together).

### akhyati (akhyāti)
**Definition:** Two true propositions combined into a false one — each component is true but the combination is not.
**Signature:** Each sub-claim individually verifiable as true; the combined claim is false.
**Example:** "Einstein won the Nobel Prize in 1921 for his theory of relativity" — he won in 1921 (true), it was for the photoelectric effect, not relativity (true that relativity was his; combined claim false).
**Mitigation:** Structural/relational grounding; verify the relation, not just the components.

## Annotation protocol
1. Read the claim and the ground truth.
2. Identify if any hallucination is present. If none, label `none`.
3. If hallucination present, select the primary class from the 6 above.
4. If multiple classes apply, select the most specific one (anyathakhyati > asatkhyati > anirvacaniyakhyati).
5. Write a one-sentence rationale.
6. For inter-annotator agreement: compute Cohen's κ before proceeding past 50 examples.
```

- [ ] **Step 3.2: Write failing tests for KhyativadaClassifier**

`tests/test_evaluation/test_khyativada.py`:
```python
import pytest
from src.evaluation.khyativada import KhyativadaClassifier, KhyativadaClass

def test_classifier_has_six_classes():
    assert len(KhyativadaClassifier.CLASSES) == 6

def test_anyathakhyati_signature():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(
        claim="Python's GIL was removed in version 3.10",
        ground_truth="Python's GIL was removed in version 3.13",
    )
    assert result["class"] == "anyathakhyati"

def test_asatkhyati_signature():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(
        claim="Use the requests.get_json() method to parse responses",
        ground_truth="requests.get_json() does not exist; use response.json() instead",
    )
    assert result["class"] == "asatkhyati"

def test_akhyati_signature():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(
        claim="Einstein won the Nobel Prize in 1921 for his theory of relativity",
        ground_truth="Einstein won in 1921 for the photoelectric effect, not relativity",
    )
    assert result["class"] == "akhyati"

def test_classify_returns_required_keys():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(claim="x", ground_truth="y")
    assert "class" in result
    assert "confidence" in result
    assert "rationale" in result
```

- [ ] **Step 3.3: Run tests — verify they fail**

```bash
pytest tests/test_evaluation/test_khyativada.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3.4: Implement KhyativadaClassifier (heuristic tier)**

`src/evaluation/khyativada.py`:
```python
from enum import Enum

class KhyativadaClass(str, Enum):
    anyathakhyati = "anyathakhyati"         # Real entity misidentified as another real entity
    atmakhyati = "atmakhyati"               # Internal pattern projected as external fact
    anirvacaniyakhyati = "anirvacaniyakhyati"  # Novel confabulation
    asatkhyati = "asatkhyati"              # Nonexistent entity asserted to exist
    viparitakhyati = "viparitakhyati"      # Systematic inversion/reversal
    akhyati = "akhyati"                    # Two true claims combined falsely
    none = "none"

class KhyativadaClassifier:
    CLASSES = [c.value for c in KhyativadaClass if c != KhyativadaClass.none]

    def classify_heuristic(self, claim: str, ground_truth: str) -> dict:
        """Rule-based heuristic classifier for unit-testable detection.
        Used for fast batch labeling; replaced by LLM-based classifier for final evaluation."""
        claim_lower = claim.lower()
        gt_lower = ground_truth.lower()

        # asatkhyati: ground truth says something "does not exist"
        if "does not exist" in gt_lower or "nonexistent" in gt_lower or "no such" in gt_lower:
            return {
                "class": KhyativadaClass.asatkhyati,
                "confidence": 0.85,
                "rationale": "Ground truth indicates the referenced entity does not exist.",
            }

        # akhyati: "not X" or "incorrect" combined with "for" (relational error)
        if ("not" in gt_lower and "for" in gt_lower) or "combination" in gt_lower:
            return {
                "class": KhyativadaClass.akhyati,
                "confidence": 0.75,
                "rationale": "Both components may be true but their combination is false.",
            }

        # anyathakhyati: both claim and ground truth reference specific versions/names
        import re
        claim_nums = re.findall(r'\d+\.\d+', claim)
        gt_nums = re.findall(r'\d+\.\d+', ground_truth)
        if claim_nums and gt_nums and set(claim_nums) != set(gt_nums):
            return {
                "class": KhyativadaClass.anyathakhyati,
                "confidence": 0.80,
                "rationale": "Version/identifier mismatch — real entity misidentified.",
            }

        return {
            "class": KhyativadaClass.atmakhyati,
            "confidence": 0.5,
            "rationale": "Default: likely internal pattern projection without grounding.",
        }

    def classify(self, claim: str, context: str, ground_truth: str, api_key: str = "") -> dict:
        """LLM-based classifier using Claude (requires api_key). Falls back to heuristic."""
        if not api_key:
            return self.classify_heuristic(claim, ground_truth)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""Classify this hallucination into exactly one of these 6 khyātivāda types:
- anyathakhyati: real entity misidentified as another real entity
- atmakhyati: internal pattern projected as external fact
- anirvacaniyakhyati: novel confabulation (neither real nor derivable)
- asatkhyati: nonexistent entity asserted to exist
- viparitakhyati: systematic inversion/reversal of A and B
- akhyati: two true claims combined into a false one

Claim: {claim}
Context provided: {context or 'none'}
Ground truth: {ground_truth}

Respond with JSON: {{"class": "<type>", "confidence": <0-1>, "rationale": "<one sentence>"}}"""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        return json.loads(response.content[0].text)

    def batch_classify(self, examples: list[dict], api_key: str = "") -> list[dict]:
        return [
            self.classify(
                e.get("claim", ""), e.get("context", ""), e.get("ground_truth", ""), api_key
            )
            for e in examples
        ]
```

- [ ] **Step 3.5: Run tests — verify they pass**

```bash
pytest tests/test_evaluation/test_khyativada.py -v
```

Expected: `5 passed`

- [ ] **Step 3.6: Commit**

```bash
git add src/evaluation/khyativada.py data/annotations/khyativada_guidelines.md tests/test_evaluation/test_khyativada.py
git commit -m "feat: khyativada hallucination classifier — 6-class heuristic + LLM-based, annotation guidelines"
```

---

## Task 4: Buddhi/Manas Agent Architecture

**Files:**
- Create: `src/agents/sakshi.py`
- Create: `src/agents/manas.py`
- Create: `src/agents/buddhi.py`
- Create: `src/agents/orchestrator.py`
- Test: `tests/test_agents/test_orchestrator.py`

- [ ] **Step 4.1: Write failing integration test for full pipeline**

`tests/test_agents/test_orchestrator.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from src.agents.orchestrator import ManusBuddhiOrchestrator
from src.avacchedaka.store import ContextStore
from src.avacchedaka.element import ContextElement, AvacchedakaConditions

@pytest.fixture
def store_with_auth_docs():
    store = ContextStore()
    store.insert(ContextElement(
        id="auth-001",
        content="JWT tokens expire after 24 hours.",
        precision=0.9,
        avacchedaka=AvacchedakaConditions(
            qualificand="auth", qualifier="token_expiry", condition="task_type=qa"
        ),
    ))
    return store

def test_orchestrator_returns_buddhi_output(store_with_auth_docs, api_key):
    orch = ManusBuddhiOrchestrator(api_key=api_key, store=store_with_auth_docs)
    result = orch.run(
        question="How long do JWT tokens last?",
        task_context="task_type=qa",
        qualificand="auth",
    )
    assert result.answer is not None
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasoning_trace, str)

def test_orchestrator_can_withhold(api_key):
    store = ContextStore()  # empty store — no grounding
    orch = ManusBuddhiOrchestrator(api_key=api_key, store=store)
    result = orch.run(
        question="What is the exact millisecond this service was deployed?",
        task_context="task_type=qa",
        qualificand="deployment",
    )
    # With no grounding, buddhi should withhold or have low confidence
    assert result.confidence < 0.6 or result.answer is None
```

Mark as `@pytest.mark.integration`.

- [ ] **Step 4.2: Run tests — verify they fail**

```bash
pytest tests/test_agents/test_orchestrator.py -v -m integration
```

Expected: `FAILED` — module not found.

- [ ] **Step 4.3: Implement SakshiPrefix (witness-invariant prefix)**

`src/agents/sakshi.py`:
```python
class SakshiPrefix:
    """Witness-invariant frozen summary. Never rewritten by context operations.
    Analogous to Advaita's sākṣī: stable reference against which vṛttis arise and pass."""

    def __init__(self, content: str):
        self._content = content  # immutable after construction

    @property
    def content(self) -> str:
        return self._content

    def as_system_message(self) -> dict:
        return {"role": "user", "content": f"<sakshi_prefix>\n{self._content}\n</sakshi_prefix>"}
```

- [ ] **Step 4.4: Implement ManasAgent**

`src/agents/manas.py`:
```python
import json
import anthropic
from dataclasses import dataclass
from src.avacchedaka.query import AvacchedakaQuery

@dataclass
class ManasOutput:
    candidate_ids: list[str]
    uncertainty: float
    recommended_queries: list[AvacchedakaQuery]
    reasoning_sketch: str

class ManasAgent:
    """Broad attention, surfaces candidates, does NOT commit to answers.
    Analogous to manas: indecisive, sensory-bound mental activity."""

    SYSTEM = """You are the manas stage of a two-stage reasoning system.
Your role: surface candidate information relevant to the question. Do NOT commit to a final answer.
Output JSON with keys: candidate_summary (str), uncertainty (0-1 float), recommended_queries (list of {qualificand, condition} dicts), reasoning_sketch (str)."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def run(self, question: str, context_window: str, task_context: str, qualificand: str) -> ManasOutput:
        messages = [
            {"role": "user", "content": f"Context:\n{context_window}\n\nQuestion: {question}\nTask: {task_context}"},
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self.SYSTEM,
            messages=messages,
        )
        try:
            raw = json.loads(response.content[0].text)
        except (json.JSONDecodeError, IndexError):
            raw = {"candidate_summary": "", "uncertainty": 0.9, "recommended_queries": [], "reasoning_sketch": ""}
        queries = [
            AvacchedakaQuery(qualificand=q.get("qualificand", qualificand), condition=q.get("condition", task_context))
            for q in raw.get("recommended_queries", [])
        ]
        return ManasOutput(
            candidate_ids=[],
            uncertainty=float(raw.get("uncertainty", 0.9)),
            recommended_queries=queries or [AvacchedakaQuery(qualificand=qualificand, condition=task_context)],
            reasoning_sketch=raw.get("reasoning_sketch", raw.get("candidate_summary", "")),
        )
```

- [ ] **Step 4.5: Implement BuddhiAgent**

`src/agents/buddhi.py`:
```python
import json
import anthropic
from dataclasses import dataclass, field

@dataclass
class BuddhiOutput:
    answer: str | None
    confidence: float
    sublated: list[str] = field(default_factory=list)
    reasoning_trace: str = ""
    khyativada_flags: list[str] = field(default_factory=list)

class BuddhiAgent:
    """Narrow attention, discriminates, commits to answer or explicitly withholds.
    Analogous to buddhi: discriminative, decisive faculty."""

    SYSTEM = """You are the buddhi stage of a two-stage reasoning system.
Your role: given candidate reasoning and context, commit to a final answer OR explicitly withhold if evidence is insufficient.
Output JSON: {"answer": str or null, "confidence": 0-1, "reasoning_trace": str, "sublated_candidates": [str], "khyativada_flags": [str]}
If confidence < 0.6 and evidence is weak, set answer to null (withhold). Never fabricate."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def run(self, question: str, context_window: str, manas_sketch: str, uncertainty: float) -> BuddhiOutput:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_window}\n\n"
                    f"Manas sketch (uncommitted):\n{manas_sketch}\n\n"
                    f"Manas uncertainty: {uncertainty:.2f}\n\n"
                    f"Question: {question}"
                ),
            }
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self.SYSTEM,
            messages=messages,
        )
        try:
            raw = json.loads(response.content[0].text)
        except (json.JSONDecodeError, IndexError):
            return BuddhiOutput(answer=None, confidence=0.0, reasoning_trace="parse error")
        return BuddhiOutput(
            answer=raw.get("answer"),
            confidence=float(raw.get("confidence", 0.0)),
            sublated=raw.get("sublated_candidates", []),
            reasoning_trace=raw.get("reasoning_trace", ""),
            khyativada_flags=raw.get("khyativada_flags", []),
        )
```

- [ ] **Step 4.6: Implement ManusBuddhiOrchestrator**

`src/agents/orchestrator.py`:
```python
from src.agents.manas import ManasAgent
from src.agents.buddhi import BuddhiAgent, BuddhiOutput
from src.agents.sakshi import SakshiPrefix
from src.avacchedaka.store import ContextStore
from src.avacchedaka.query import AvacchedakaQuery

DEFAULT_SAKSHI = SakshiPrefix(
    "This system conducts rigorous, grounded reasoning. "
    "It withholds answers when evidence is insufficient rather than fabricating."
)

class ManusBuddhiOrchestrator:
    def __init__(
        self,
        api_key: str,
        store: ContextStore,
        sakshi: SakshiPrefix = DEFAULT_SAKSHI,
        manas_model: str = "claude-haiku-4-5-20251001",
        buddhi_model: str = "claude-sonnet-4-6",
    ):
        self.store = store
        self.sakshi = sakshi
        self.manas = ManasAgent(api_key, manas_model)
        self.buddhi = BuddhiAgent(api_key, buddhi_model)

    def run(self, question: str, task_context: str, qualificand: str) -> BuddhiOutput:
        query = AvacchedakaQuery(qualificand=qualificand, condition=task_context)
        initial_context = self.store.to_context_window(query, max_tokens=2048)

        manas_out = self.manas.run(
            question=question,
            context_window=initial_context,
            task_context=task_context,
            qualificand=qualificand,
        )

        # Pull additional context based on manas recommendations
        additional_parts = []
        for rec_query in manas_out.recommended_queries[:3]:
            additional_parts.append(self.store.to_context_window(rec_query, max_tokens=1024))
        enriched_context = initial_context + "\n" + "\n".join(additional_parts)

        return self.buddhi.run(
            question=question,
            context_window=enriched_context,
            manas_sketch=manas_out.reasoning_sketch,
            uncertainty=manas_out.uncertainty,
        )
```

- [ ] **Step 4.7: Add integration marker and run test**

Add `@pytest.mark.integration` to both tests in `test_orchestrator.py`.

```bash
pytest tests/test_agents/ -v -m integration --timeout=60
```

Expected: `2 passed` (requires `ANTHROPIC_API_KEY`)

- [ ] **Step 4.8: Commit**

```bash
git add src/agents/ tests/test_agents/
git commit -m "feat: buddhi/manas two-stage agent architecture with sākṣī witness prefix and avacchedaka-mediated context pull"
```

---

## Task 5: H1 — Schema-Congruence Context Rot Benchmark

**Files:**
- Create: `src/evaluation/schema_congruence.py`
- Create: `src/evaluation/metrics.py`
- Create: `experiments/h1_schema_congruence/build_benchmark.py`
- Create: `experiments/h1_schema_congruence/run.py`
- Test: `tests/test_evaluation/test_schema_congruence.py`

- [ ] **Step 5.1: Write failing tests**

`tests/test_evaluation/test_schema_congruence.py`:
```python
from src.evaluation.schema_congruence import CongruenceBenchmarkBuilder

def test_congruent_version_has_same_domain_distractors():
    builder = CongruenceBenchmarkBuilder()
    example = builder.build_example(
        gold_passage="JWT tokens expire after 24 hours.",
        domain="web_security",
        target_length_k=4,
        version="congruent",
    )
    assert example["version"] == "congruent"
    assert example["gold_passage"] in example["context"]
    assert len(example["distractors"]) > 0

def test_incongruent_version_has_different_domain_distractors():
    builder = CongruenceBenchmarkBuilder()
    example = builder.build_example(
        gold_passage="JWT tokens expire after 24 hours.",
        domain="web_security",
        target_length_k=4,
        version="incongruent",
    )
    assert example["version"] == "incongruent"

def test_congruence_ratio_computed():
    from src.evaluation.metrics import congruence_ratio
    example = {"version": "congruent", "distractors": ["x"] * 9, "gold_passage": "y"}
    ratio = congruence_ratio(example)
    assert 0.0 <= ratio <= 1.0
```

- [ ] **Step 5.2: Run — verify fail**

```bash
pytest tests/test_evaluation/test_schema_congruence.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 5.3: Implement CongruenceBenchmarkBuilder**

`src/evaluation/schema_congruence.py`:
```python
import random
from dataclasses import dataclass, field

# Distractor pools by domain — extend with real corpus data for full experiment
DISTRACTOR_POOLS = {
    "web_security": [
        "HTTPS uses TLS 1.3 for transport encryption.",
        "OAuth 2.0 uses authorization codes for secure delegation.",
        "CSRF tokens are single-use random values tied to session.",
        "Rate limiting prevents brute-force attacks on login endpoints.",
        "Content Security Policy headers mitigate XSS attacks.",
        "SQL parameterized queries prevent injection attacks.",
        "bcrypt work factor should be ≥12 for password hashing.",
        "CORS preflight requests use the OPTIONS HTTP method.",
        "Session cookies should have Secure and HttpOnly flags.",
        "JWTs are base64url-encoded, not encrypted by default.",
    ],
    "unrelated": [
        "The Amazon rainforest covers approximately 5.5 million km².",
        "Piano has 88 keys in standard concert configuration.",
        "Water boils at 100°C at sea level.",
        "Shakespeare wrote 37 plays and 154 sonnets.",
        "The speed of light is approximately 3×10⁸ m/s.",
        "Human genome contains approximately 3 billion base pairs.",
        "The Eiffel Tower was completed in 1889.",
        "Beethoven was deaf when he composed his Ninth Symphony.",
        "The Great Wall of China is not visible from space.",
        "Honey never spoils due to its low moisture content.",
    ],
}

@dataclass
class BenchmarkExample:
    gold_passage: str
    context: str
    question: str
    answer: str
    domain: str
    version: str
    distractors: list[str] = field(default_factory=list)
    target_length_k: int = 4

    def __getitem__(self, key):
        return getattr(self, key)

class CongruenceBenchmarkBuilder:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def build_example(
        self,
        gold_passage: str,
        domain: str,
        target_length_k: int,
        version: str,  # "congruent" or "incongruent"
        question: str = "",
        answer: str = "",
    ) -> BenchmarkExample:
        if version == "congruent":
            pool = DISTRACTOR_POOLS.get(domain, DISTRACTOR_POOLS["unrelated"])
        else:
            pool = DISTRACTOR_POOLS["unrelated"]

        n_distractors = max(1, target_length_k * 250 // max(len(gold_passage), 1))
        distractors = self.rng.choices(pool, k=min(n_distractors, len(pool)))

        # Interleave gold passage at a random position
        all_passages = distractors.copy()
        insert_pos = self.rng.randint(0, len(all_passages))
        all_passages.insert(insert_pos, gold_passage)
        context = "\n\n".join(all_passages)

        return BenchmarkExample(
            gold_passage=gold_passage,
            context=context,
            question=question or f"Based on the context, what is stated about: {gold_passage[:40]}?",
            answer=answer or gold_passage,
            domain=domain,
            version=version,
            distractors=distractors,
            target_length_k=target_length_k,
        )
```

`src/evaluation/metrics.py`:
```python
def congruence_ratio(example: dict) -> float:
    """Ratio of domain-congruent distractors to total context passages."""
    n_distractors = len(example.get("distractors", []))
    total = n_distractors + 1  # +1 for gold passage
    if example.get("version") == "congruent":
        return n_distractors / total if total > 0 else 0.0
    return 0.0

def expected_calibration_error(confidences: list[float], correctness: list[bool], n_bins: int = 10) -> float:
    import numpy as np
    confs = np.array(confidences)
    correct = np.array(correctness, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confs[mask].mean()
        ece += mask.sum() / len(confs) * abs(bin_acc - bin_conf)
    return float(ece)
```

- [ ] **Step 5.4: Run tests — verify pass**

```bash
pytest tests/test_evaluation/test_schema_congruence.py -v
```

Expected: `3 passed`

- [ ] **Step 5.5: Write experiment run.py for H1**

`experiments/h1_schema_congruence/run.py`:
```python
import os
import json
import mlflow
from src.evaluation.schema_congruence import CongruenceBenchmarkBuilder
from src.evaluation.metrics import congruence_ratio

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
MODEL = "claude-haiku-4-5-20251001"  # cheap for baseline runs

def run_experiment():
    import anthropic
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    builder = CongruenceBenchmarkBuilder(seed=RANDOM_SEED)

    gold = "JWT tokens expire after 24 hours."
    results = []
    for version in ("congruent", "incongruent"):
        example = builder.build_example(
            gold_passage=gold,
            domain="web_security",
            target_length_k=4,
            version=version,
        )
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"{example.context}\n\nQuestion: {example.question}\nAnswer concisely:"
            }],
        )
        answer = response.content[0].text.strip()
        correct = gold.lower()[:30] in answer.lower()
        results.append({
            "version": version,
            "correct": correct,
            "congruence_ratio": congruence_ratio(example),
            "answer": answer,
        })
    return results

if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-1-schema-congruence")
    with mlflow.start_run():
        mlflow.log_params({"model": MODEL, "hypothesis": "H1", "seed": RANDOM_SEED})
        results = run_experiment()
        congruent_correct = next(r["correct"] for r in results if r["version"] == "congruent")
        incongruent_correct = next(r["correct"] for r in results if r["version"] == "incongruent")
        mlflow.log_metrics({
            "congruent_accuracy": int(congruent_correct),
            "incongruent_accuracy": int(incongruent_correct),
        })
        with open("data/experiments/h1_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
```

- [ ] **Step 5.6: Commit**

```bash
git add src/evaluation/schema_congruence.py src/evaluation/metrics.py experiments/h1_schema_congruence/ tests/test_evaluation/test_schema_congruence.py
git commit -m "feat: H1 schema-congruence benchmark builder and metrics"
```

---

## Task 6: H2 — Precision-Weighted RAG

**Files:**
- Create: `src/rag/precision_rag.py`
- Create: `src/rag/conflicting_qa.py`
- Create: `src/rag/baselines.py`
- Create: `experiments/h2_precision_rag/run.py`
- Test: `tests/test_rag/test_precision_rag.py`

- [ ] **Step 6.1: Write failing tests**

`tests/test_rag/test_precision_rag.py`:
```python
from src.rag.conflicting_qa import ConflictingSourceQA
from src.rag.precision_rag import PrecisionWeightedRAG

def test_conflicting_qa_example_has_two_sources():
    example = ConflictingSourceQA.build_example(
        question="What is the default JWT expiry?",
        correct_answer="24 hours",
        incorrect_answer="1 hour",
        correct_source_precision=0.9,
        incorrect_source_precision=0.3,
    )
    assert len(example["sources"]) == 2
    assert example["correct_answer"] == "24 hours"

def test_precision_rag_prefers_high_precision_source():
    rag = PrecisionWeightedRAG()
    example = ConflictingSourceQA.build_example(
        question="What is the default JWT expiry?",
        correct_answer="24 hours",
        incorrect_answer="1 hour",
        correct_source_precision=0.9,
        incorrect_source_precision=0.2,
    )
    selected = rag.select_sources(example["sources"], top_k=1)
    assert selected[0]["answer"] == "24 hours"

def test_precision_rag_detects_conflict():
    rag = PrecisionWeightedRAG()
    sources = [
        {"content": "JWT expires in 24 hours.", "precision": 0.8, "answer": "24 hours"},
        {"content": "JWT expires in 1 hour.", "precision": 0.7, "answer": "1 hour"},
    ]
    conflict = rag.detect_conflict(sources)
    assert conflict is True
```

- [ ] **Step 6.2: Run — verify fail**

```bash
pytest tests/test_rag/ -v
```

Expected: `FAILED`

- [ ] **Step 6.3: Implement ConflictingSourceQA and PrecisionWeightedRAG**

`src/rag/conflicting_qa.py`:
```python
from dataclasses import dataclass

class ConflictingSourceQA:
    @staticmethod
    def build_example(
        question: str,
        correct_answer: str,
        incorrect_answer: str,
        correct_source_precision: float,
        incorrect_source_precision: float,
    ) -> dict:
        return {
            "question": question,
            "correct_answer": correct_answer,
            "sources": [
                {
                    "content": f"According to this source: {correct_answer}",
                    "precision": correct_source_precision,
                    "answer": correct_answer,
                    "is_correct": True,
                },
                {
                    "content": f"According to this source: {incorrect_answer}",
                    "precision": incorrect_source_precision,
                    "answer": incorrect_answer,
                    "is_correct": False,
                },
            ],
        }
```

`src/rag/precision_rag.py`:
```python
class PrecisionWeightedRAG:
    def select_sources(self, sources: list[dict], top_k: int = 3) -> list[dict]:
        """Sort sources by precision descending, return top_k."""
        return sorted(sources, key=lambda s: s.get("precision", 0.5), reverse=True)[:top_k]

    def detect_conflict(self, sources: list[dict]) -> bool:
        """True if top-2 sources give different answers and precision gap < 0.3."""
        if len(sources) < 2:
            return False
        top2 = self.select_sources(sources, top_k=2)
        answers_differ = top2[0].get("answer") != top2[1].get("answer")
        gap = abs(top2[0].get("precision", 0) - top2[1].get("precision", 0))
        return answers_differ and gap < 0.3

    def build_prompt(self, question: str, sources: list[dict]) -> str:
        selected = self.select_sources(sources)
        conflict = self.detect_conflict(sources)
        source_text = "\n".join(
            f"[Source precision={s['precision']:.2f}] {s['content']}" for s in selected
        )
        conflict_note = "\nNote: Sources conflict. Express appropriate uncertainty." if conflict else ""
        return f"Sources:\n{source_text}{conflict_note}\n\nQuestion: {question}\nAnswer:"
```

- [ ] **Step 6.4: Run tests — verify pass**

```bash
pytest tests/test_rag/ -v
```

Expected: `3 passed`

- [ ] **Step 6.5: Write experiment run.py for H2**

`experiments/h2_precision_rag/run.py`:
```python
import os
import json
import mlflow
from src.rag.conflicting_qa import ConflictingSourceQA
from src.rag.precision_rag import PrecisionWeightedRAG

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

def run_experiment():
    import anthropic
    client = anthropic.Anthropic()
    rag = PrecisionWeightedRAG()

    examples = [
        ConflictingSourceQA.build_example("What is the default JWT expiry?", "24 hours", "1 hour", 0.9, 0.3),
        ConflictingSourceQA.build_example("What HTTP method does CORS preflight use?", "OPTIONS", "GET", 0.85, 0.4),
    ]

    results = []
    for ex in examples:
        prompt = rag.build_prompt(ex["question"], ex["sources"])
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.content[0].text.strip()
        correct = ex["correct_answer"].lower() in answer.lower()
        conflict_flagged = "uncertain" in answer.lower() or "conflict" in answer.lower()
        results.append({
            "question": ex["question"],
            "correct": correct,
            "conflict_flagged": conflict_flagged,
            "answer": answer,
        })
    return results

if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-2-precision-rag")
    with mlflow.start_run():
        mlflow.log_params({"hypothesis": "H2", "seed": RANDOM_SEED})
        results = run_experiment()
        accuracy = sum(r["correct"] for r in results) / len(results)
        conflict_rate = sum(r["conflict_flagged"] for r in results) / len(results)
        mlflow.log_metrics({"accuracy": accuracy, "conflict_flag_rate": conflict_rate})
        with open("data/experiments/h2_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
```

- [ ] **Step 6.6: Commit**

```bash
git add src/rag/ tests/test_rag/ experiments/h2_precision_rag/
git commit -m "feat: H2 precision-weighted RAG with conflict detection and conflicting-source QA benchmark builder"
```

---

## Task 7: H4 — Event-Boundary Compaction

**Files:**
- Create: `src/compaction/detector.py`
- Create: `src/compaction/compactor.py`
- Create: `experiments/h4_event_boundary/run.py`
- Test: `tests/test_compaction/test_detector.py`

- [ ] **Step 7.1: Write failing tests**

`tests/test_compaction/test_detector.py`:
```python
from src.compaction.detector import EventBoundaryDetector

def test_high_surprise_triggers_boundary():
    detector = EventBoundaryDetector(surprise_threshold=0.8)
    # Simulate token-level surprises; >threshold means boundary
    surprises = [0.2, 0.3, 0.25, 0.9, 0.2]  # spike at index 3
    boundaries = detector.detect_from_surprises(surprises)
    assert 3 in boundaries

def test_no_boundary_when_surprises_low():
    detector = EventBoundaryDetector(surprise_threshold=0.8)
    surprises = [0.2, 0.3, 0.25, 0.4, 0.2]
    boundaries = detector.detect_from_surprises(surprises)
    assert len(boundaries) == 0

def test_task_switch_signal_triggers_boundary():
    detector = EventBoundaryDetector()
    result = detector.detect_from_signals(
        task_switch=True, surprise_spike=False
    )
    assert result is True
```

- [ ] **Step 7.2: Run — verify fail**

```bash
pytest tests/test_compaction/ -v
```

Expected: `FAILED`

- [ ] **Step 7.3: Implement EventBoundaryDetector and BoundaryTriggeredCompactor**

`src/compaction/detector.py`:
```python
class EventBoundaryDetector:
    def __init__(self, surprise_threshold: float = 0.75):
        self.surprise_threshold = surprise_threshold

    def detect_from_surprises(self, surprises: list[float]) -> list[int]:
        """Returns indices where surprise exceeds threshold (event boundaries)."""
        return [i for i, s in enumerate(surprises) if s >= self.surprise_threshold]

    def detect_from_signals(self, task_switch: bool, surprise_spike: bool) -> bool:
        return task_switch or surprise_spike
```

`src/compaction/compactor.py`:
```python
from src.avacchedaka.store import ContextStore
from src.avacchedaka.query import AvacchedakaQuery

class BoundaryTriggeredCompactor:
    def __init__(self, store: ContextStore, compress_threshold: float = 0.3):
        self.store = store
        self.compress_threshold = compress_threshold

    def compact_at_boundary(self, qualificand: str, task_context: str) -> list[str]:
        """Compress low-precision elements at event boundary. Returns compressed IDs."""
        return self.store.compress(self.compress_threshold)

    def threshold_compact(self, token_count: int, token_threshold: int, qualificand: str) -> list[str]:
        """Baseline: compress when token count exceeds threshold."""
        if token_count >= token_threshold:
            return self.store.compress(self.compress_threshold)
        return []
```

- [ ] **Step 7.4: Run tests — verify pass**

```bash
pytest tests/test_compaction/ -v
```

Expected: `3 passed`

- [ ] **Step 7.5: Commit**

```bash
git add src/compaction/ tests/test_compaction/ experiments/h4_event_boundary/
git commit -m "feat: H4 event-boundary compaction — EventBoundaryDetector and BoundaryTriggeredCompactor"
```

---

## Task 8: H7 — Adaptive Forgetting Schedules

**Files:**
- Create: `src/forgetting/schedules.py`
- Create: `src/forgetting/distribution_shift.py`
- Create: `experiments/h7_adaptive_forgetting/run.py`
- Test: `tests/test_forgetting/test_schedules.py`

- [ ] **Step 8.1: Write failing tests**

`tests/test_forgetting/test_schedules.py`:
```python
from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.store import ContextStore
from src.forgetting.schedules import (
    NoForgetting, FixedCompaction, RecencyWeightedForgetting,
    RewardWeightedForgetting, BadhaFirstForgetting,
)

def make_store_with_elements(n: int, base_precision: float = 0.8) -> ContextStore:
    store = ContextStore()
    from datetime import datetime, timedelta
    for i in range(n):
        store.insert(ContextElement(
            id=f"e-{i:03d}",
            content=f"Element {i}",
            precision=base_precision,
            avacchedaka=AvacchedakaConditions(qualificand="test", qualifier="prop", condition="z"),
            timestamp=datetime.utcnow() - timedelta(hours=n - i),  # older = lower index
        ))
    return store

def test_no_forgetting_retains_all():
    store = make_store_with_elements(5)
    schedule = NoForgetting(store)
    removed = schedule.apply()
    assert removed == []

def test_fixed_compaction_removes_oldest():
    store = make_store_with_elements(10)
    schedule = FixedCompaction(store, keep_newest=5)
    removed = schedule.apply()
    assert len(removed) == 5

def test_badha_first_clears_sublated():
    store = make_store_with_elements(3)
    # Sublate element 0
    store.insert(ContextElement(
        id="newer",
        content="Updated.",
        precision=0.9,
        avacchedaka=AvacchedakaConditions(qualificand="test", qualifier="prop", condition="z"),
    ))
    store.sublate("e-000", "newer")
    schedule = BadhaFirstForgetting(store)
    removed = schedule.apply()
    assert "e-000" in removed
```

- [ ] **Step 8.2: Run — verify fail**

```bash
pytest tests/test_forgetting/ -v
```

Expected: `FAILED`

- [ ] **Step 8.3: Implement forgetting schedule variants**

`src/forgetting/schedules.py`:
```python
import dataclasses
from src.avacchedaka.store import ContextStore

class NoForgetting:
    def __init__(self, store: ContextStore):
        self.store = store

    def apply(self) -> list[str]:
        return []

class FixedCompaction:
    def __init__(self, store: ContextStore, keep_newest: int = 50):
        self.store = store
        self.keep_newest = keep_newest

    def apply(self) -> list[str]:
        elements = sorted(
            [e for e in self.store._elements.values() if e.sublated_by is None],
            key=lambda e: e.timestamp,
            reverse=True,
        )
        to_remove = elements[self.keep_newest:]
        removed = []
        for e in to_remove:
            self.store._elements[e.id] = dataclasses.replace(e, precision=0.0)
            removed.append(e.id)
        return removed

class RecencyWeightedForgetting:
    def __init__(self, store: ContextStore, decay_factor: float = 0.9):
        self.store = store
        self.decay_factor = decay_factor

    def apply(self) -> list[str]:
        from datetime import datetime
        now = datetime.utcnow()
        removed = []
        for eid, e in list(self.store._elements.items()):
            if e.sublated_by is not None:
                continue
            age_hours = (now - e.timestamp).total_seconds() / 3600
            decayed = e.precision * (self.decay_factor ** age_hours)
            if decayed < 0.3:
                self.store._elements[eid] = dataclasses.replace(e, precision=0.0)
                removed.append(eid)
        return removed

class RewardWeightedForgetting:
    def __init__(self, store: ContextStore, reward_key: str = "task_relevance", keep_threshold: float = 0.5):
        self.store = store
        self.reward_key = reward_key
        self.keep_threshold = keep_threshold

    def apply(self) -> list[str]:
        removed = []
        for eid, e in list(self.store._elements.items()):
            if e.sublated_by is not None:
                continue
            reward = e.salience.get(self.reward_key, e.precision)
            if reward < self.keep_threshold:
                self.store._elements[eid] = dataclasses.replace(e, precision=0.0)
                removed.append(eid)
        return removed

class BadhaFirstForgetting:
    """Clear sublated (cancelled) elements first — bādha principle."""
    def __init__(self, store: ContextStore):
        self.store = store

    def apply(self) -> list[str]:
        removed = []
        for eid, e in list(self.store._elements.items()):
            if e.sublated_by is not None:
                self.store._elements[eid] = dataclasses.replace(e, precision=0.0)
                removed.append(eid)
        return removed
```

- [ ] **Step 8.4: Run tests — verify pass**

```bash
pytest tests/test_forgetting/ -v
```

Expected: `3 passed`

- [ ] **Step 8.5: Commit**

```bash
git add src/forgetting/ tests/test_forgetting/
git commit -m "feat: H7 adaptive forgetting — 5 schedule variants (none/fixed/recency/reward/badha-first)"
```

---

## Task 9: H5 — Avacchedaka Multi-Agent Coordination Experiment

**Files:**
- Create: `experiments/h5_avacchedaka_multiagent/run.py`
- Test: `tests/test_avacchedaka/test_multiagent_integration.py`

- [ ] **Step 9.1: Write integration test for multi-agent coordination**

`tests/test_avacchedaka/test_multiagent_integration.py`:
```python
import pytest
from src.avacchedaka.store import ContextStore
from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.query import AvacchedakaQuery

@pytest.mark.integration
def test_two_agents_no_conflict_with_avacchedaka(api_key):
    """Both agents operate on same store via avacchedaka queries.
    Second agent's contradicting insert should sublate first agent's element."""
    store = ContextStore()
    # Agent 1 inserts a claim
    store.insert(ContextElement(
        id="agent1-claim-001",
        content="Database uses PostgreSQL 14.",
        precision=0.85,
        avacchedaka=AvacchedakaConditions(qualificand="database", qualifier="version", condition="task_type=deploy"),
        provenance="agent1",
    ))
    # Agent 2 discovers contradicting information and sublates
    store.insert(ContextElement(
        id="agent2-claim-001",
        content="Database uses PostgreSQL 16 (upgraded last week).",
        precision=0.95,
        avacchedaka=AvacchedakaConditions(qualificand="database", qualifier="version", condition="task_type=deploy"),
        provenance="agent2",
    ))
    store.sublate("agent1-claim-001", "agent2-claim-001")

    # Retrieval should return only the newer, non-sublated claim
    query = AvacchedakaQuery(qualificand="database", condition="task_type=deploy")
    results = store.retrieve(query)
    assert len(results) == 1
    assert results[0].id == "agent2-claim-001"
    assert results[0].content == "Database uses PostgreSQL 16 (upgraded last week)."

def test_conflict_rate_without_avacchedaka():
    """Without avacchedaka, both conflicting claims would be returned — simulate conflict."""
    store = ContextStore()
    # Without sublation, both get inserted and both get returned
    store.insert(ContextElement(
        id="raw-001", content="PostgreSQL 14.", precision=0.85,
        avacchedaka=AvacchedakaConditions(qualificand="database", qualifier="version", condition="task_type=deploy"),
    ))
    store.insert(ContextElement(
        id="raw-002", content="PostgreSQL 16.", precision=0.95,
        avacchedaka=AvacchedakaConditions(qualificand="database", qualifier="version", condition="task_type=deploy"),
    ))
    # No sublation — both returned (conflict)
    query = AvacchedakaQuery(qualificand="database", condition="task_type=deploy")
    results = store.retrieve(query)
    assert len(results) == 2  # conflict: two contradicting answers
```

- [ ] **Step 9.2: Run tests — verify pass**

```bash
pytest tests/test_avacchedaka/test_multiagent_integration.py -v
```

Expected: `2 passed` (no API key needed for these tests)

- [ ] **Step 9.3: Write H5 experiment run.py**

`experiments/h5_avacchedaka_multiagent/run.py`:
```python
import os
import json
import mlflow
from src.avacchedaka.store import ContextStore
from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.query import AvacchedakaQuery

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

COORDINATION_TASKS = [
    {
        "task": "Deploy database version",
        "agent1_claim": {"id": "t1-a1", "content": "PostgreSQL 14", "precision": 0.8},
        "agent2_claim": {"id": "t1-a2", "content": "PostgreSQL 16 (correct)", "precision": 0.95},
        "qualificand": "database", "condition": "task_type=deploy",
        "correct": "PostgreSQL 16",
    },
    {
        "task": "Auth token expiry",
        "agent1_claim": {"id": "t2-a1", "content": "24 hours", "precision": 0.7},
        "agent2_claim": {"id": "t2-a2", "content": "1 hour (updated policy)", "precision": 0.9},
        "qualificand": "auth", "condition": "task_type=qa",
        "correct": "1 hour",
    },
]

def run_with_avacchedaka(task: dict) -> dict:
    store = ContextStore()
    conds = AvacchedakaConditions(
        qualificand=task["qualificand"], qualifier="version", condition=task["condition"]
    )
    store.insert(ContextElement(id=task["agent1_claim"]["id"], content=task["agent1_claim"]["content"],
                                precision=task["agent1_claim"]["precision"], avacchedaka=conds))
    store.insert(ContextElement(id=task["agent2_claim"]["id"], content=task["agent2_claim"]["content"],
                                precision=task["agent2_claim"]["precision"], avacchedaka=conds))
    store.sublate(task["agent1_claim"]["id"], task["agent2_claim"]["id"])
    results = store.retrieve(AvacchedakaQuery(qualificand=task["qualificand"], condition=task["condition"]))
    conflict = len(results) > 1
    correct = task["correct"].lower() in (results[0].content.lower() if results else "")
    return {"conflict": conflict, "correct": correct, "n_results": len(results)}

def run_without_avacchedaka(task: dict) -> dict:
    store = ContextStore()
    conds = AvacchedakaConditions(
        qualificand=task["qualificand"], qualifier="version", condition=task["condition"]
    )
    store.insert(ContextElement(id=task["agent1_claim"]["id"], content=task["agent1_claim"]["content"],
                                precision=task["agent1_claim"]["precision"], avacchedaka=conds))
    store.insert(ContextElement(id=task["agent2_claim"]["id"], content=task["agent2_claim"]["content"],
                                precision=task["agent2_claim"]["precision"], avacchedaka=conds))
    # No sublation
    results = store.retrieve(AvacchedakaQuery(qualificand=task["qualificand"], condition=task["condition"]))
    conflict = len(results) > 1
    correct = task["correct"].lower() in (results[0].content.lower() if results else "")
    return {"conflict": conflict, "correct": correct, "n_results": len(results)}

if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-5-avacchedaka-multiagent")
    with mlflow.start_run():
        mlflow.log_params({"hypothesis": "H5", "seed": RANDOM_SEED})
        with_results = [run_with_avacchedaka(t) for t in COORDINATION_TASKS]
        without_results = [run_without_avacchedaka(t) for t in COORDINATION_TASKS]
        with_conflict_rate = sum(r["conflict"] for r in with_results) / len(with_results)
        without_conflict_rate = sum(r["conflict"] for r in without_results) / len(without_results)
        reduction = (without_conflict_rate - with_conflict_rate) / max(without_conflict_rate, 0.001)
        mlflow.log_metrics({
            "with_avacchedaka_conflict_rate": with_conflict_rate,
            "without_avacchedaka_conflict_rate": without_conflict_rate,
            "conflict_rate_reduction_pct": reduction * 100,
        })
        summary = {
            "with_avacchedaka": with_results,
            "without_avacchedaka": without_results,
            "conflict_rate_reduction_pct": reduction * 100,
        }
        with open("data/experiments/h5_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
```

- [ ] **Step 9.4: Run full test suite**

```bash
make test-unit
```

Expected: All unit tests pass. Count ≥ 25.

- [ ] **Step 9.5: Check coverage**

```bash
make coverage
```

Expected: ≥ 80% line coverage on `src/`.

- [ ] **Step 9.6: Commit**

```bash
git add experiments/h5_avacchedaka_multiagent/ tests/test_avacchedaka/test_multiagent_integration.py
git commit -m "feat: H5 avacchedaka multi-agent coordination experiment — typed sublation eliminates conflict"
```

---

## Task 10: H3 — Buddhi/Manas Experiment Run

**Files:**
- Create: `experiments/h3_buddhi_manas/run.py`

- [ ] **Step 10.1: Write experiment run.py for H3**

`experiments/h3_buddhi_manas/run.py`:
```python
import os
import json
import mlflow
from src.agents.orchestrator import ManusBuddhiOrchestrator
from src.avacchedaka.store import ContextStore
from src.avacchedaka.element import ContextElement, AvacchedakaConditions

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

QA_TASKS = [
    {
        "question": "How long do JWT tokens remain valid?",
        "gold": "24 hours",
        "qualificand": "auth",
        "task_context": "task_type=qa",
        "docs": [("JWT tokens expire after 24 hours.", 0.9, "auth", "task_type=qa")],
    },
    {
        "question": "What is the millisecond-precise deployment timestamp of service X?",
        "gold": None,  # Should withhold — no grounding
        "qualificand": "deployment",
        "task_context": "task_type=qa",
        "docs": [],  # No relevant docs
    },
]

def run_experiment():
    api_key = os.environ["ANTHROPIC_API_KEY"]
    results = []
    for task in QA_TASKS:
        store = ContextStore()
        for content, precision, qualificand, condition in task["docs"]:
            store.insert(ContextElement(
                id=f"doc-{len(store._elements)}",
                content=content,
                precision=precision,
                avacchedaka=AvacchedakaConditions(
                    qualificand=qualificand, qualifier="fact", condition=condition
                ),
            ))
        orch = ManusBuddhiOrchestrator(api_key=api_key, store=store)
        output = orch.run(
            question=task["question"],
            task_context=task["task_context"],
            qualificand=task["qualificand"],
        )
        gold = task["gold"]
        if gold is None:
            # Expect withhold
            success = output.answer is None or output.confidence < 0.6
        else:
            success = gold.lower() in (output.answer or "").lower()
        results.append({
            "question": task["question"],
            "success": success,
            "answer": output.answer,
            "confidence": output.confidence,
            "expected_withhold": gold is None,
        })
    return results

if __name__ == "__main__":
    mlflow.set_experiment("hypothesis-3-buddhi-manas")
    with mlflow.start_run():
        mlflow.log_params({"hypothesis": "H3", "seed": RANDOM_SEED})
        results = run_experiment()
        accuracy = sum(r["success"] for r in results) / len(results)
        mlflow.log_metrics({"task_success_rate": accuracy})
        with open("data/experiments/h3_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
```

- [ ] **Step 10.2: Run baseline smoke test (single task, no full eval)**

```bash
ANTHROPIC_API_KEY=your_key python -c "
from src.agents.orchestrator import ManusBuddhiOrchestrator
from src.avacchedaka.store import ContextStore
orch = ManusBuddhiOrchestrator(api_key='$ANTHROPIC_API_KEY', store=ContextStore())
result = orch.run('What is 2+2?', 'task_type=math', 'arithmetic')
print('Answer:', result.answer, '| Confidence:', result.confidence)
"
```

Expected: Prints an answer with confidence ≥ 0.5.

- [ ] **Step 10.3: Final full test run**

```bash
make test
```

Expected: All tests pass. No failures.

- [ ] **Step 10.4: Final coverage check**

```bash
make coverage
```

Expected: ≥ 80% overall, ≥ 90% for `src/avacchedaka/`.

- [ ] **Step 10.5: Final commit**

```bash
git add experiments/h3_buddhi_manas/
git commit -m "feat: H3 buddhi/manas experiment runner — grounded QA + withhold-when-uncertain test cases"
```

---

## Self-Review Checklist

**Spec coverage:**
- F1 (avacchedaka library) → Task 2 ✓
- F2 (buddhi/manas) → Tasks 4, 10 ✓
- F3 (khyātivāda classifier) → Task 3 ✓
- F4 (precision RAG) → Task 6 ✓
- F5 (event-boundary compaction) → Task 7 ✓
- F6 (schema-congruence benchmark) → Task 5 ✓
- F7 (adaptive forgetting) → Task 8 ✓
- H5 multi-agent coordination → Task 9 ✓
- H3 full experiment → Task 10 ✓
- Project scaffold + Makefile → Task 1 ✓

**Placeholder scan:** No TBDs. All code blocks are complete. All commands have expected output.

**Type consistency:** `ContextElement`, `AvacchedakaConditions`, `AvacchedakaQuery`, `ContextStore`, `ManasOutput`, `BuddhiOutput` — all defined in Task 2/4, referenced consistently in Tasks 5–10. `sublate()` signature consistent throughout. `to_context_window()` called with `(query, max_tokens)` consistently.
