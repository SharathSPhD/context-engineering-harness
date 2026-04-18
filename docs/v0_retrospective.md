# v0 Retrospective — Frozen Baseline for the v2 Rebuild

**Frozen at:** commit `a354ca5` on branch `main` (2026-04-18)
**Rebuild branch:** `rebuild/v2`
**Worktree:** `.worktrees/v2`

This document is the **honest baseline** of the original Context Engineering Harness as it existed before the v2 rebuild. It records what was claimed, what was actually delivered, and the 12 hard gaps that motivated the rebuild. The v2 paper's "Methodology critique" chapter cites this file directly.

---

## 1. v0 headline numbers

From [`docs/validation_report.md`](validation_report.md) (generated 2026-04-18 17:13):

| Hypothesis | Claim (one-line) | v0 metric | v0 verdict | Real-world status |
|---|---|---|---|---|
| H1 | Schema-congruence predicts context rot better than length | congruent=0.667, incongruent=0.667 (no delta) | FAIL | Inconclusive: target inverted; N=3 per arm; identical answers across arms |
| H2 | Precision-weighted RAG > top-k on conflicting sources | both 1.0 | FAIL | Inconclusive: ceiling effect, N=3, correct source always first in distractor list |
| H3 | Buddhi/manas two-stage >= single-stage | both 0.5 | PASS (tie) | Inconclusive: N=2 questions, both got null answers |
| H4 | Event-boundary compaction >= threshold compaction | both 1.0 | PASS (tie) | Vacuous: 0 LLM calls; both methods compress noise-only by construction |
| H5 | Avacchedaka annotation reduces multi-agent conflict >= 30% | reduction=100% | PASS | Vacuous: 0 LLM calls; "without" path is `except KeyError: pass`, "with" path is the absence of that exception |
| H6 | Khyātivāda heuristic accuracy >= 60% | 9/9 | PASS | Heavily label-leaked: heuristic keys off literal substrings present in expected labels |
| H7 | Adaptive forgetting >= no-forgetting on post-shift | both 1 | PASS (tie) | Inconclusive: 1 element, 1 question, hand-curated to pass |

**Net honest read:** Of 7 hypotheses, **0 are validly supported by v0**. 4 PASS verdicts are due to ties on tiny samples or vacuous algorithmic constructions; 2 FAILs are due to inverted targets / ceiling effects; 1 PASS (H6) is due to label leakage.

This is **not** a rebuke of the framework — the framework's primitives (Avacchedaka, Sublate, Sākṣī, Buddhi/Manas, Khyātivāda taxonomy) are intellectually defensible and worth keeping. The v0 *evaluation* is what is broken, not the underlying ideas. The v2 rebuild keeps the primitives and replaces the evaluation.

---

## 2. v0 architecture in one diagram

```
src/
  cli_bridge.py            # claude CLI subprocess wrapper (drops assistant turns, ignores max_tokens)
  config.py                # TOML loader, env overrides
  agents/
    sakshi.py              # SakshiPrefix (witness invariants)
    manas.py               # Fast/intuitive draft agent (max_tokens=512 hardcoded)
    buddhi.py              # Slow/deliberate verifier (max_tokens=512 hardcoded)
    orchestrator.py        # Manas -> Buddhi pipeline (sakshi inlined into user context)
  avacchedaka/
    schema.py              # Static JSON schema dicts (no runtime validation)
    query.py               # AvacchedakaQuery + matches() (qualifier field unused)
    store.py               # ContextStore (char-based budget, silent overwrite)
  compaction/
    detector.py            # EventBoundaryDetector (consumes pre-supplied surprise scalars)
    compactor.py           # BoundaryTriggeredCompactor (ignores qualificand/task_context)
  rag/
    precision_rag.py       # 4-line sort by precision (NOT Bayesian)
    conflicting_qa.py      # Synthetic Q&A (correct source ALWAYS first)
  forgetting/
    schedules.py           # NoForgetting/BadhaFirst/FixedCompaction (operates on .salience only)
  evaluation/
    schema_congruence.py   # CongruenceBenchmarkBuilder (target_length_k IGNORED, distractor pool <=10)
    khyativada.py          # KhyativadaClassifier (heuristic emits 4 of 6 classes; LLM path hardcodes claude-sonnet-4-6)

experiments/
  validate/                # CLI-callable validation suite (the one that produced the v0 report)
  h{1..7}_*/               # Older MLflow-instrumented runners (different metrics, mostly stale)

data/
  experiments/validation_results.json    # raw JSON of v0 verdicts
  annotations/khyativada_guidelines.md   # 6-class annotation protocol (no annotated data yet)
```

---

## 3. The 12 hard gaps (the rebuild charter)

| # | Gap | Where | TRIZ principle | v2 resolution |
|---|---|---|---|---|
| G1 | Synthetic NexusAPI corpus (10 docs, 3 shifts) | `experiments/validate/data.py` | P28 substitution | Real benchmark adapters: HELMET, RULER, NoCha, HaluEval, TruthfulQA, FACTS-Grounding, SWE-bench Verified |
| G2 | Tiny N (often 3 or 1 per arm) | all `experiments/validate/h*.py` | P38 strong stats | Multi-seed ≥5, multi-model ≥3, bootstrap CI, paired permutation |
| G3 | H1 target direction inverted | `experiments/validate/h1_*.py` | P13 inversion | Fix target; recompute |
| G4 | PrecisionWeightedRAG is just a sort | [`src/rag/precision_rag.py`](../src/rag/precision_rag.py) | P24 intermediary | Bayesian Beta-posterior aggregation + Brier/ECE |
| G5 | EventBoundaryDetector consumes pre-supplied surprise scalars | [`src/compaction/detector.py`](../src/compaction/detector.py) | P23 feedback | Real per-token surprise via local Qwen3 (vllm or HF transformers) |
| G6 | KhyativadaClassifier heuristic emits 4 of 6 classes; LLM path hardcoded | [`src/evaluation/khyativada.py`](../src/evaluation/khyativada.py) | P3 local quality | Few-shot Claude with structured output + 3000 annotated examples + Cohen's κ ≥ 0.6 |
| G7 | `compact_at_boundary` ignores `qualificand`/`task_context` | [`src/compaction/compactor.py`](../src/compaction/compactor.py) | P40 composite | Honor both end-to-end |
| G8 | Schema-congruence distractor pool ≤10 strings; `target_length_k` ignored | [`src/evaluation/schema_congruence.py`](../src/evaluation/schema_congruence.py) | P17 another dimension | Wikipedia + arXiv distractors, tiktoken-exact 8K–1M tokens |
| G9 | Sakshi text inlined into Buddhi user context, not system | [`src/agents/orchestrator.py`](../src/agents/orchestrator.py) | P2 taking out | Real `system` field both stages |
| G10 | `cli_bridge` drops assistant turns; ignores `max_tokens` | [`src/cli_bridge.py`](../src/cli_bridge.py) | P32 parameter changes | Preserve full role turns; forward `max_tokens` |
| G11 | Forgetting schedules operate purely on `.salience`/`.precision` | [`src/forgetting/schedules.py`](../src/forgetting/schedules.py) | P35 parameter changes | Replay-statistics-aware decay |
| G12 | `ContextStore.to_context_window` uses char count (chars≈max_tokens*4) | [`src/avacchedaka/store.py`](../src/avacchedaka/store.py) | P32 | Tokenizer-exact (tiktoken / claude tokenizer) |

---

## 4. What we keep from v0

- **The seven hypotheses**: their *claims* are intellectually defensible; only the v0 *evaluations* fail them.
- **The Vedic-epistemology vocabulary**: Avacchedaka (typed limitor), Sākṣī (witness invariants), Sublation (`bādha`, never deletes), Buddhi/Manas (slow/fast), Khyātivāda (6-class hallucination taxonomy).
- **The TRIZ origin trail**: `.triz/session.jsonl` records the contradictions and inventive principles that seeded each hypothesis. The v2 paper's "Origin" chapter cites this file directly.
- **The CLI-strict invariant**: zero paid API; all LLM calls go through the user's `claude` CLI subscription.

## 5. What we replace

- The synthetic NexusAPI corpus → real public benchmarks
- Hardcoded models in agents/classifier → fully `config.toml`-driven
- Heuristic-only classifier → few-shot LLM with structured output + annotated training set
- The "validate" suite → a `BenchmarkAdapter` ABC + `HypothesisSpec` + a stats harness that gates every claim with bootstrap CI + paired permutation p-values
- The character-counted context window → tokenizer-exact
- The ad-hoc forgetting schedules → replay-statistics-aware adaptive forgetting

## 6. What we *add*

- A **standalone Claude Code plugin** at `github.com/SharathSPhD/pratyaksha-context-eng-harness` (root `marketplace.json` mirroring `SharathSPhD/attractor-flow`) with MCP server, skills, agents, commands, hooks.
- A **token-aware CLI scheduler** (`tools/dev/scheduler/cli_budget.py`) that detects rate limits, sleeps to next 5h window, caches exhaustively, and resumes durably.
- **Three validation tracks**: (A) H1–H7 on real benchmarks, (B) live case study on `pallets/click`, (C) SWE-bench Verified A/B at N≥100.
- An **arXiv preprint** with 12 sections + 6 appendices, 13 figures, 7 tables, ≥150 citations.

---

## 7. Reproducing the v0 baseline

```bash
git checkout a354ca5
make validate
make report
diff data/experiments/validation_results.json docs/v0_retrospective_data_snapshot.json
```

The v2 worktree carries a copy of the v0 results JSON at `docs/v0_retrospective_data_snapshot.json` so the baseline cannot drift.
