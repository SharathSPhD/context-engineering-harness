# Negative and Null Results

A faithful empirical paper must publish what *did not work* with the same rigour as what did. This appendix records four substantive negative results and one null result encountered during the v2 iteration of the harness. These are documented in raw form in `docs/v0_retrospective.md` (the v0 baseline retrospective) and in the per-iteration journals under `attractor-flow-state/journals/`; this appendix is the consolidated reviewer-facing summary.

## Negative — `PrecisionWeightedRAG` (v0)

**What we tried.** The v0 harness had a single `PrecisionWeightedRAG` aggregator that combined multiple sources by precision-weighted majority vote.

**Why it failed.** Three reasons surfaced empirically:

1. **No calibration.** It reported binary "agree/disagree" without a calibrated posterior. ECE ran at $\sim 0.18$ on H2.
2. **No conflict detection.** Conflicts that should have triggered sublation merely returned the higher-precision answer silently. The agent had no signal to *audit* the disagreement.
3. **No witness.** Decisions were not traceable: the agent could not later explain *why* one source had been preferred.

**Replacement.** The Bayesian Beta-Bernoulli aggregator with explicit posterior margin and conflict detection (Section 5.3). H2 ECE dropped to $0.07$.

## Negative — single-stage `BuddhiAgent`

**What we tried.** A single-stage agent (one Claude call) that both attended to the store *and* emitted the user-visible answer.

**Why it failed.** It conflated the attentional and judging steps. In ablation runs with a single-stage agent, the H3 grounding score dropped by 0.18 vs. the two-stage Manas/Buddhi pipeline. The single-stage agent kept "remembering" items that the rule engine had marked as `bādhita` because they were syntactically near-by in its prompt.

**Replacement.** The two-stage Manas-then-Buddhi orchestration (Section 5.4), where Manas's only output is a list of `selected_ids` and Buddhi sees only those.

## Negative — naive LRU compaction

**What we tried.** A simple LRU eviction policy in `compact`.

**Why it failed.** It evicted high-precision items that had not been *recently* touched but were still load-bearing for later turns (e.g. the user's stated goal). H5 plummeted by $-0.22$.

**Replacement.** The `AdaptiveForgetting` rule set (Section 5.7), which is precision-weighted, witness-aware, and bias-aware against eviction of `source=user` items.

## Negative — single-source-of-truth Khyātivāda annotator

**What we tried.** A single LLM-as-judge annotator over 3 000 examples for the H6 evaluation.

**Why it failed.** It artificially inflated agreement: the *same* LLM was scoring its own taxonomy. Cohen's kappa against an independent rubric was unmeasurable in this design.

**Replacement.** Two independent annotators (heuristic + LLM-as-judge) with rule-based guardrails on top, reported as Cohen's kappa $\kappa = 0.736$ (H6, Section 8). This is "substantial agreement" by Landis-Koch convention but is *not* the inflated single-annotator number we initially computed ($0.91$, which we now treat as a methodological warning sign).

## Null — vLLM Qwen3-1.7B vs. heuristic surprise backend

**What we tried.** Replacing the lightweight Zipf-style heuristic surprise scorer (used in Section 5.6) with a real per-token negative-log-likelihood from Qwen3-1.7B served via vLLM.

**What we found.** The two backends agreed on event boundary placement to within $\pm 1$ token on $94\%$ of test sessions. The H4 EventBoundaryCompactor metric *did not change to within noise* ($\Delta = +0.003$, $p = 0.71$ on a 200-session sample).

**Implication.** For event-boundary detection in realistic agent transcripts, the heuristic surprise backend is sufficient and saves $\sim 700$ ms/turn plus a $\sim 1.7$ GB GPU footprint. We ship the heuristic as default and document the vLLM path as opt-in, with the explicit note that *we have not found a workload where it pays for itself*. We treat this as a healthy null result that informs deployment guidance.

## Negative — early plugin tried to inline ContextStore into prompt

**What we tried.** In an early plugin draft, every Buddhi prompt was prefixed with a JSON dump of the *entire* live `ContextStore` (capped at 4K tokens). This was meant to make the model "see everything".

**Why it failed.** It re-introduced exactly the lost-in-the-middle pathology the harness exists to defeat: items in the middle of the dump were systematically ignored, and the prompt's token budget exploded under multi-turn workloads.

**Replacement.** Manas explicitly *selects* which items to surface, and only the selected subset is inlined into Buddhi's prompt. The full store is queried only via tool calls, never inlined wholesale. This is the reason the harness's contract is "selected items, not the store".

These negative results are not failures of the underlying epistemology — they are failures of *first* implementations. Each pushed us toward a more disciplined operationalization, which is the value the harness now codifies.
