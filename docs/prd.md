# Product Requirements Document
## Context Engineering Synthesis: LLM × Neuroscience × Vedic Epistemology

**Version:** 0.1 — 2026-04-17  
**Status:** Draft  
**Owner:** Research Initiative  

---

## 1. Vision

Build and validate a unified **context engineering framework** grounded in three independent traditions — transformer/LLM engineering, cognitive neuroscience, and classical Indian epistemology — that resolves the fundamental contradiction: *context must be simultaneously complete and selective*.

The project produces:
- A **formal context algebra** (the avacchedaka notation system) that makes context engineering principled rather than artisanal
- A **two-stage agent architecture** (buddhi/manas) that reduces context rot and hallucination
- A **typed hallucination ontology** (khyātivāda-grounded) for diagnosis and targeted mitigation
- **Seven validated research hypotheses** with reproducible benchmarks and falsifiable experimental designs
- An **adaptive forgetting system** inspired by hippocampal replay prioritization

---

## 2. Scope

### In Scope
- Designing and running experiments for H1–H7 from `docs/research.md`
- Building the avacchedaka notation layer (H5) as a reusable Python library
- Implementing the buddhi/manas two-stage architecture (H3) using the Anthropic SDK
- Implementing the khyātivāda hallucination classifier (H6) with training data from HaluEval + TruthfulQA
- Building adaptive forgetting (H7) for long-horizon agent sessions
- Producing benchmark datasets for each hypothesis test
- Writing technical papers or documentation for each validated hypothesis

### Out of Scope
- Pre-training or fine-tuning foundation models
- Building production consumer applications
- Full Sanskrit scholarship — the Vedic frameworks are used as engineering inspiration, not theological claims
- Replicating all cited papers from scratch; we build *on top of* existing implementations

---

## 3. Researcher / User Personas

### P1 — ML Researcher (primary)
Needs rigorous experimental infrastructure: reproducible baselines, controlled benchmark splits, ablation harnesses. Wants to run a hypothesis test in < 1 day setup time.

### P2 — Agent Systems Engineer
Needs the avacchedaka notation system as a drop-in library for their existing LangChain/LlamaIndex/Claude Code agent. Wants a clear API, type annotations, and documentation.

### P3 — AI Safety Researcher
Needs the khyātivāda hallucination ontology for richer evaluation of model truthfulness. Wants annotated datasets and classifier weights.

### P4 — Cognitive Scientist / Interdisciplinary Scholar
Needs clear mappings between computational and biological/philosophical constructs. Wants the conceptual mappings from `docs/research.md` formalized with precision and cited correctly.

---

## 4. Feature Requirements

### F1: Avacchedaka Notation Library (`avacchedaka`)

**Priority:** P0 (enabling primitive)

| Requirement | Details |
|---|---|
| F1.1 | Define `ContextElement` with fields: `content`, `precision` (float 0–1), `avacchedaka` (dict of limitor conditions), `sublated_by` (reference or None), `timestamp`, `provenance` |
| F1.2 | Define `ContextStore` with operations: `insert`, `retrieve(limitor)`, `sublate(element_id, by_id)`, `compress(below_precision_threshold)` |
| F1.3 | Implement precision-weighted retrieval: `retrieve` returns elements sorted by `precision * task_relevance_score` |
| F1.4 | Implement typed sublation: `sublate(A, B)` sets A.sublated_by = B.id, A.precision = 0.0, logs the event — does NOT delete A |
| F1.5 | Provide JSON schema for avacchedaka annotation format compatible with Claude API messages |
| F1.6 | Unit test coverage ≥ 90% |
| F1.7 | Integration test: complete a multi-hop QA task using only avacchedaka-mediated context assembly |

### F2: Buddhi/Manas Two-Stage Agent Architecture

**Priority:** P0

| Requirement | Details |
|---|---|
| F2.1 | `ManasAgent`: fast, broad-attention stage — surfaces candidate context elements, does NOT commit to answers, emits structured candidate list |
| F2.2 | `BuddhiAgent`: slow, narrow-attention stage — receives candidate list from manas, discriminates, commits to answer, may sublate candidates |
| F2.3 | Both agents use same base model (Claude claude-sonnet-4-6) — architectural isolation, not model isolation |
| F2.4 | Manas emits structured output: `ManasOutput(candidates: list[ContextElement], uncertainty: float, recommended_retrieval: list[AvacchedakaQuery])` |
| F2.5 | Buddhi receives manas output + avacchedaka-retrieved context, emits `BuddhiOutput(answer: str, confidence: float, sublated: list[str], reasoning_trace: str)` |
| F2.6 | Architecture must expose a "withhold" option: buddhi can return `answer=None, confidence_below_threshold=True` |
| F2.7 | Evaluate on HELMET, NoCha, RULER, and a custom "withhold-or-answer" benchmark |

### F3: Khyātivāda Hallucination Classifier

**Priority:** P1

| Requirement | Details |
|---|---|
| F3.1 | Implement 6-class classifier over the khyātivāda ontology: `anyathakhyati`, `atmakhyati`, `anirvacaniyakhyati`, `asatkhyati`, `viparitakhyati`, `akhyati` |
| F3.2 | Annotation schema for labeling hallucinations in HaluEval, TruthfulQA, FACTS-grounding |
| F3.3 | Annotate minimum 500 hallucination examples per class (3,000 total) |
| F3.4 | Classifier achieves ≥ 70% macro-F1 on held-out test set |
| F3.5 | Demonstrate that retrieval-heavy mitigation disproportionately reduces `anyathakhyati`; calibration-heavy reduces `atmakhyati`; constrained decoding reduces `anirvacaniyakhyati` |
| F3.6 | Publish annotation guidelines as a standalone document |

### F4: Precision-Weighted RAG (H2)

**Priority:** P1

| Requirement | Details |
|---|---|
| F4.1 | Build "conflicting-sources QA" benchmark: questions where retrieved corpora contain ≥2 contradictory answers at varying confidence levels |
| F4.2 | Implement baseline: vanilla top-k RAG |
| F4.3 | Implement comparison: Self-RAG (existing) |
| F4.4 | Implement experimental: Bayesian RAG with per-source precision scores (from metadata: recency, author authority, internal consistency) |
| F4.5 | Measure: accuracy, ECE (expected calibration error), "correctly flagged conflict" rate |
| F4.6 | Minimum 200 conflicting-source QA examples per domain (3 domains: science, code, history) |

### F5: Event-Boundary Compaction (H4)

**Priority:** P1

| Requirement | Details |
|---|---|
| F5.1 | Implement event-boundary detector: monitors per-token surprise (cross-entropy), task-switch signals, prediction-failure signatures |
| F5.2 | Trigger compaction at detected boundaries rather than fixed token thresholds |
| F5.3 | Instrument an open agent (Aider or OpenHands) to use boundary-triggered compaction |
| F5.4 | Evaluate on SWE-bench Verified and synthetic multi-task suites |
| F5.5 | Metric: task success rate at equal total token budget vs. threshold-triggered baseline |

### F6: Schema-Congruence Context Rot Benchmark (H1)

**Priority:** P2

| Requirement | Details |
|---|---|
| F6.1 | Construct paired long-context prompts at 32K / 64K / 128K tokens |
| F6.2 | Version A: distractors from same topical domain (congruent) |
| F6.3 | Version B: distractors from unrelated topics (incongruent) |
| F6.4 | Evaluate on Claude, GPT-4.1, Gemini 2.5, Qwen3, Jamba |
| F6.5 | Measure: error rate vs. congruence ratio (not just length) |
| F6.6 | Extend RULER benchmark with congruence-controlled distractors |

### F7: Adaptive Forgetting Schedules (H7)

**Priority:** P2

| Requirement | Details |
|---|---|
| F7.1 | Implement 5 forgetting schedule variants: none, fixed, recency-weighted, reward-weighted, bādha-first (contradicted-memory-cleared-first) |
| F7.2 | Build long-horizon benchmark with controlled distribution shifts (codebase API convention change midstream; user preference reversal) |
| F7.3 | Metric: performance on post-shift tasks relative to pre-shift tasks |

---

## 5. Non-Functional Requirements

| NFR | Target |
|---|---|
| Reproducibility | All experiments must be reproducible from a single `make reproduce-h{n}` command |
| Latency | avacchedaka library retrieval: < 50ms for stores up to 10K elements |
| Cost | All hypothesis tests runnable under $500 total API spend |
| Documentation | Every public API function has a one-line docstring; architectural decisions documented in `docs/spec.md` |
| Testing | Minimum 80% line coverage on all library code; integration tests for each hypothesis |

---

## 6. Success Metrics

### Research Outcomes (6–12 months)
- H1: Congruence-ratio model predicts context rot better than length-only model (p < 0.05, ΔF1 ≥ 5pp)
- H2: Bayesian RAG outperforms Self-RAG on conflicting-source QA (accuracy +5pp, ECE −10%)
- H3: Buddhi/manas outperforms single-stage on NoCha global comprehension (+5pp) and "withhold" calibration (+10% ECE improvement)
- H4: Boundary-triggered compaction matches or beats threshold compaction at 80% token budget
- H5: Avacchedaka-annotated multi-agent reduces conflict rate by ≥ 30% vs. no-annotation baseline
- H6: Targeted mitigation reduces respective khyātivāda type at 2× rate of untargeted mitigation
- H7: Adaptive forgetting shows ≥ 10pp advantage on post-shift tasks vs. fixed compaction

### Engineering Outcomes
- `avacchedaka` library published as open-source Python package
- Annotated khyātivāda hallucination dataset (3,000 examples) released publicly
- Benchmarks (conflicting-source QA, schema-congruence context rot, withhold-or-answer) released

---

## 7. Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| `anthropic` SDK | ≥ 0.50 | Claude API access for all experiments |
| `langchain` / `llama-index` | latest | RAG baselines |
| `chromadb` or `qdrant` | latest | Vector store for context elements |
| `pytest` | ≥ 8.0 | Test harness |
| `mlflow` | ≥ 2.0 | Experiment tracking |
| `pandas` / `numpy` | latest | Data manipulation |
| `scikit-learn` | latest | Classifier training (H6) |

---

## 8. Timeline

| Phase | Milestone | Target |
|---|---|---|
| 0 | Project setup, avacchedaka library skeleton, test harness | Week 1–2 |
| 1 | H5 avacchedaka notation v1 with full tests | Week 3–4 |
| 2 | H3 buddhi/manas architecture + H6 classifier annotation | Week 5–8 |
| 3 | H2 precision-weighted RAG + H4 event-boundary compaction | Week 9–12 |
| 4 | H1 schema-congruence benchmark + H7 adaptive forgetting | Week 13–16 |
| 5 | Cross-hypothesis integration, frontier-direction prototypes | Week 17–20 |
| 6 | Write-up, dataset release, library packaging | Week 21–24 |

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| API costs exceed budget | Medium | Start with Claude Haiku for baseline runs; use cache prefix for shared prompts |
| H3 buddhi/manas shows no advantage over single-stage | Medium | Pre-registered with expected failure modes; null result is publishable |
| Annotation quality for khyātivāda classes is poor | High | Two-annotator agreement protocol; start with 50-example pilot before full annotation |
| avacchedaka notation is too complex for adoption | Low | Design Python-idiomatic API first; formal notation is secondary representation |
| Benchmark contamination (models trained on RULER/NoCha) | High | Use H1's novel congruence-controlled extension; use custom "withhold-or-answer" |
