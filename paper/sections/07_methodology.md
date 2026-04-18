# 7 · Validation Methodology

We validate the harness across three orthogonal evidence layers — public benchmarks (L1), a deterministic live case study (L2), and a SWE-bench Verified A/B head-to-head (L3) — under a single, cost-aware build pipeline that respects CLI rate limits. This section specifies the methodology shared across all three layers.

## 7.1 The CLI budget scheduler (dev-time infrastructure)

The entire validation pipeline ran under a custom **`CLIBudgetScheduler`** (P0 of the project plan) that enforces a hard ceiling on per-window CLI spend, persists every call to a `cost_ledger.db` SQLite ledger, layers an MD5-keyed disk cache over identical (system, user, model, max_tokens) tuples, and pre-computes prompt-cache prefixes for repeated runs of the same benchmark adapter \citep{anthropic2024promptcaching}. Rate-limit detection uses parsed Anthropic 429 responses combined with a heuristic five-hour-window estimator; on detection the scheduler issues a HALT signal to the `attractor-flow` driver \citep{attractorflowplugin}, which pauses the current basin and, after the configured cool-down, RESUMEs from the last persisted ledger row.

This scheduler is **not part of the shipped plugin** — it is a build-time tool, like `pytest` or `make`. We document it here because it made the L1+L2+L3 validation feasible at all under realistic CLI quotas, and because the cost ledger it produced is part of the reproducibility manifest (Appendix F).

## 7.2 The BenchmarkAdapter ABC and HypothesisSpec

All seven hypotheses share a single typed harness:

```python
class BenchmarkAdapter(ABC):
    name: str
    target_direction: TargetDirection            # HIGHER_IS_BETTER | LOWER_IS_BETTER
    @abstractmethod
    def load(self, *, n: int, seed: int) -> list[BenchmarkExample]: ...
    @abstractmethod
    def score(self, ex: BenchmarkExample, output: ModelOutput) -> float: ...
```

A `HypothesisSpec` then composes:

```python
@dataclass
class HypothesisSpec:
    id: str                  # "H1", "H2", ...
    title: str
    adapter: BenchmarkAdapter
    treatment: ModelCaller   # plugin-enabled
    baseline: ModelCaller    # plugin-disabled or alternative
    n: int
    seeds: list[int]
    models: list[str]
    target_delta: float
```

The orchestrator `MultiSeedRunner` sweeps every `(model, seed)` pair, persists per-example scores, computes paired bootstrap CIs and paired permutation p-values, and writes a `HypothesisOutcome` JSON artefact to `experiments/results/h{N}/`. This is the artefact P7 (Section 7.7) consumes.

## 7.3 Public benchmark adapters (L1)

We implemented adapters for **seven** benchmark families. Two design principles applied across all of them:

1. **Token-exact context construction.** Whenever a long context is required, we use `tiktoken` (`o200k_base`) to assemble exactly the configured number of tokens, never an approximation. This matters because most published RULER/HELMET numbers do not preserve token-exactness \citep{hsieh2024ruler, yen2024helmet, hong2025contextrot}.
2. **Real distractor corpora when available, synthetic fallback when not.** RULER, HELMET-Recall, HELMET-RAG, and NoCha use Wikipedia + arXiv distractors loaded via the HuggingFace `datasets` library \citep{huggingface2024datasets}; the same adapters fall back to a deterministic synthetic generator when the network or HF token is unavailable, which is the path actually exercised in every CI run for reproducibility. The synthetic generators are themselves seedable and produce *qualitatively* the same difficulty profile (verified in P3).

The seven adapters and the hypothesis each gates are:

| Adapter | Family | Hypothesis |
|---|---|---|
| `RulerAdapter` | Long-context recall | H1 |
| `HelmetRagAdapter`, `HelmetRecallAdapter` | Long-context RAG/recall | H2 |
| `NoChaAdapter` | Narrative claim verification | H2 (joint) |
| `HaluEvalAdapter`, `TruthfulQAAdapter`, `FactsGroundingAdapter` | Hallucination | H6 |
| `SweBenchVerifiedAdapter` | Code generation | H4, P6-C |

For SWE-bench Verified \citep{openai2024sweverified, jimenez2024swebench} we ship two scorers: a **heuristic scorer** that checks whether the generated diff modifies the *target file* (zero Docker dependency, used for L1 sweep and CI), and a **stub Docker harness** that, if invoked, defers to the upstream SWE-bench Verified evaluation harness for final pass/fail. We use the heuristic scorer throughout L1 and L3 to keep the comparison strictly about *context discipline* rather than about coding skill; we acknowledge this and revisit it in Section 11.

## 7.4 Statistical methodology

We adopt a single statistical recipe across all studies:

- **Per-seed bootstrap CI.** For each `(model, seed)` we compute a 95% percentile bootstrap CI \citep{efron1979bootstrap} on per-example deltas, with $B=10^4$ resamples.
- **Paired permutation test.** For each `(model, seed)` we run a paired two-sided permutation test \citep{good2005permutation} on the per-example deltas with $\min(2^k, 10^4)$ permutations.
- **Cohen's d.** Standardised effect size on paired deltas \citep{cohen1988statistical}.
- **Stouffer-Z omnibus** \citep{stouffer1949combining, liptak1958combining}. To combine across studies we use the *weighted* Stouffer-Z method with weights $w_i = \sqrt{n_i}$, computing the inverse normal CDF and survival function via the Beasley-Springer-Moro routine \citep{beasley1977normal, moro1995tail} so we have *zero* dependency on SciPy in the aggregator. The combined two-sided p is reported with both the studies' n and the sum of weights.

All statistics, figures, and tables are produced by the single Python module `experiments/v2/p7/aggregate.py`, are deterministic modulo the timestamp, and are unit-tested by `tests/test_v2/test_p7_aggregate.py` (12 tests, all passing).

## 7.5 Models tested

Every L1 study sweeps at least two models:

- **`claude-haiku-4-5`** — small, low-latency, cost-efficient.
- **`claude-sonnet-4-6`** — large, full-context-class.

Both treatment and baseline arms always use the *same* model on the *same* seed so the paired comparison is clean. We deliberately did *not* introduce a third model family (e.g. GPT-4o or Qwen-3-72B) to avoid confounding context-discipline with model-family differences; future work (Section 12) will sweep across families.

## 7.6 Live case study (L2) and SWE-bench Verified A/B (L3)

These deserve methodology paragraphs of their own because they are not benchmark adapters in the L1 sense:

### 7.6.1 P6-B (live case study)

We curate three real GitHub issues — Django request-body double-read, requests retry-adapter timeout, pandas iterrows dtype coercion — each with 3–5 evidence items mixing stale and fresh sources. The harness arm enters every evidence item via `context_insert` with the documented `precision`, `condition`, and `stale` flag, then issues `sublate_with_evidence` whenever a fresher item supersedes a stale one. The baseline arm processes evidence in *discovery order* with a *first-seen-wins* policy (Section 9), reflecting the Lost-in-the-Middle anchoring bias \citep{liu2023lostmiddle}. Both arms are LLM-free and deterministic — the answer is the *concatenated qualifier set of the live items the arm chose to keep* — so the comparison strictly tests the *context discipline*, not generation quality.

### 7.6.2 P6-C (SWE-bench Verified A/B)

For each of $n=120$ SWE-bench Verified instances and each of 3 seeds × 2 models (= 720 paired runs), we synthesise a *research trail* (`p6c/research_evidence.py`) of 4 evidence snippets per instance — 2 stale (with wrong file paths and superseded API names, simulating low-precision Stack Overflow / blog posts) and 2 fresh (with correct paths and current APIs). The harness arm filters the trail through the plugin (drops `stale=True` items via `sublate_with_evidence` and budget-truncates the rest); the baseline arm naively concatenates and budget-truncates the trail. Both arms then build a fixed-budget research block (default 8 K tokens) that is fed to a deterministic `PatchSimulator` which emits a stub diff anchored on the *first plausible file path in the research block*. We score via the heuristic *target-path-hit-rate* and a paired permutation test over both per-instance and per-(model, seed) groupings.

The PatchSimulator design choice is critical: by *not* using a real LLM patch generator, we cleanly isolate the harness's contribution as a *context discipline* rather than as generation quality, which is the only intervention the plugin actually makes.

## 7.7 Aggregation: P7

The single module `experiments/v2/p7/aggregate.py` consumes every artefact from H1–H7, P6-B, and P6-C; emits 13 figures and 7 tables (Sections 8–10 reference them by ID); and writes the omnibus Stouffer-Z statistic to `T7_omnibus_stouffer.{md,csv}` and `_summary.md`. The aggregator is fully deterministic modulo timestamp and is itself tested by `tests/test_v2/test_p7_aggregate.py`. Its outputs are the canonical evidence base of this paper.

## 7.8 Reproducibility

The full reproducibility manifest is in **Appendix F**: commit SHAs, seeds, hardware, software versions, total measured CLI spend (from `cost_ledger.db`), and exact wall-clock per study. The repository's CI (`pytest`) runs the full validation pipeline in offline mode (synthetic-fallback adapters) on every commit and is currently green at **499 passing tests, 2 skipped, 0 failing** as of the build that produced this paper.
