# Results — Layer 3: SWE-bench Verified A/B Head-to-Head (P6-C)

Layer 3 supplies the most ecologically valid *coding-context* evidence in the paper, but it is not the headline claim. The mechanisms validated in Layers 1 (general long-context and hallucination benchmarks, §8) and 2 (live multi-domain case study, §9) are agent- and domain-agnostic; SWE-bench Verified is included here because it is *one challenging coding-specific instantiation* of the same mechanisms under a fixed token budget. We run a head-to-head A/B test on **120 SWE-bench Verified \citep{openai2024sweverified, jimenez2024swebench}**-style instances, **3 seeds × 2 models** (= **720 paired runs**). Both arms see the same instances, the same **synthetic** evidence trail, and the same model; the only difference is whether the system is in the loop. Patch generation is deterministically anchored on the first plausible file path of the research block. The headline run reported here uses the **deterministic patch-simulator** path (no Docker harness) with the fast research-block budget **`--research-block-budget 512`** as recorded in `experiments/results/p6c/_summary.json` (`spec.research_budget_tokens = 512`, `spec.use_docker_harness = false`). A separate Docker-harness sub-sample at 30 instances reports scorer agreement $\kappa = 0.97$ between the heuristic and the official Docker grader; the larger budget (`--research-block-budget 8192`) is supported by the runner but is not the source of the numbers below. The numbers come from `experiments/results/p6c/swebench_ab.json` and are aggregated in Table~\ref{tab:t4_p6c_headline} (headline) and Table~\ref{tab:t5_p6c_per_seed} (per-seed breakdown) and Figures~\ref{fig:f10}--\ref{fig:f11}.

## Design

For each SWE-bench Verified instance we:

1. **Generate a research trail of four snippets.** Two are *stale* (low precision in $[0.20, 0.45]$, point at *wrong* file paths constructed by perturbing the correct path with a synthetic typo, and reference superseded API names typical of pre-issue Stack Overflow / blog answers). Two are *fresh* (precision in $[0.80, 0.97]$, point at the *correct* file path, reference the current API). Stale items carry `superseded_by_id` pointers to fresh items; the four are then *shuffled* so the agent cannot use ordering as a signal. (Implementation: `experiments/v2/p6c/research_evidence.py`, deterministic per-seed.)

2. **Build a research block under a fixed token budget.** The headline run reported here uses the deterministic patch-simulator path with **`--research-block-budget 512`** (the run-card value `spec.research_budget_tokens = 512` recorded in `_summary.json`). The runner also supports **`--research-block-budget 8192`**; that larger budget would compress the gain (the system's contribution shifts from truncation to sublation), and the headline numbers below should not be interpreted as a 8\,K-budget claim.
   - **`with_harness` arm.** Insert every snippet via `context_insert`; trigger `sublate_with_evidence` for every stale-vs-fresh pair via the dominance rule; then assemble the block from `live` items only, truncating from the end if the budget is exceeded.
   - **`without_harness` arm.** Concatenate raw snippets in shuffle order and truncate to the same token budget.

3. **Feed the research block to a deterministic `PatchSimulator`** that scans the block for the *first* file path mentioned in backtick-quoted form and emits a stub diff anchored on that path. The simulator is *not* an LLM; it is a deterministic anchoring function. This is intentional — it makes the experiment a clean A/B on the *research-block contents*, not on the model's coding skill, which the system does not address.

4. **Score the resulting diff** by a binary *target-path-hit* metric: 1.0 if the diff modifies the SWE-bench Verified ground-truth file, 0.0 otherwise. We additionally record `n_target_path_hit`, `n_research_sublations`, and `mean_score` per (model, seed) cell.

5. **Statistics.** Two paired permutation tests are computed: (i) per-instance, $n=720$ pairs (every (instance, model, seed) triple is its own pair); (ii) per-(model, seed), $n=6$ pairs (every cell's mean score is a pair).

## Headline result

| metric | value |
|---|---|
| treatment_metric_mean | **0.5000** |
| baseline_metric_mean | 0.2514 |
| **per_instance_delta** | **+0.2486** |
| per_instance_CI 95% | [0.2299, 0.2667] |
| per_instance_p_value (paired permutation) | **0.0005** |
| per_instance_Cohen's d | **0.994** |
| per_instance_n_pairs | 720 |
| per_(model, seed)_delta | +0.2486 |
| per_(model, seed)_CI 95% | [0.2361, 0.2611] |
| per_(model, seed)_p_value | **0.03125** (= $1/32$, the discrete test floor) |
| per_(model, seed)_n_pairs | 6 |
| **treatment_target_path_hit_rate** | **1.0000 — 100% in all 6 (model × seed) cells, 720 / 720 paired runs** |
| baseline_target_path_hit_rate | 0.5028 (362 / 720) |
| total_sublations_fired | **1,440** |
| target_met | **True** |

Source: `experiments/results/p6c/_summary.json` and Table~\ref{tab:t4_p6c_headline}. Visualised by Figures~\ref{fig:f10} (paired-difference histogram, all positive) and \ref{fig:f11} (target-path-hit-rate bars). The per-instance permutation $p = 0.0005$ and the per-(model, seed) $p = 0.03125$ are both load-bearing: the latter is the **floor** of the exact permutation distribution on six cells, so we supplement it with bootstrap CIs as in Section 7.4.

```{=latex}
\begin{table}[!tbp]
\centering
\caption{\textbf{T4 --- P6-C SWE-bench Verified A/B (headline).} Per-instance and per-(model, seed) deltas with paired permutation $p$-values, 95\% bootstrap CIs, and Cohen's $d$. Source: \texttt{experiments/results/p6c/swebench\_ab.json}.}
\label{tab:t4_p6c_headline}
\small
\input{tables/T4_p6c_swebench_ab_headline.tex}
\end{table}

\begin{figure}[!tbp]
\centering
\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth]{F10_P6C_paired_diff_histogram.png}
  \caption{Paired-difference histogram, $n = 720$.}
  \label{fig:f10}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth]{F11_P6C_target_path_hitrate.png}
  \caption{Target-path-hit rate per (model, seed) cell.}
  \label{fig:f11}
\end{subfigure}
\caption{\textbf{P6-C SWE-bench Verified A/B head-to-head.} Treatment hits the target file path in 720/720 paired runs (100\% in every (model, seed) cell); baseline hits 362/720 (50.3\%) under an identical 512-token research-block budget (the headline run; \texttt{spec.research\_budget\_tokens = 512}).}
\label{fig:f10f11}
\end{figure}
```

## Per-(model, seed) breakdown

The system's behaviour is *identical* across `claude-haiku-4-5` and `claude-sonnet-4-6` (because the patch simulator is deterministic, by design — see Section 10.1). The baseline's behaviour varies slightly across seeds (because the seed determines the shuffle order of the four snippets, and the baseline's `first-seen` anchoring is order-sensitive). Concretely, treatment hits target-path **120/120** in every cell; baseline hits 57, 66, 58, 57, 66, 58 across the six cells — a near-coin-flip outcome, exactly as the *Lost-in-the-Middle*-style anchoring \citep{liu2023lostmiddle, kazemnejad2024impact} predicts when the wrong-path snippet appears first. Table~\ref{tab:t5_p6c_per_seed} reports the full breakdown.

```{=latex}
\begin{table}[!tbp]
\centering
\caption{\textbf{T5 --- P6-C per-(model, seed) breakdown.} Treatment hits 120/120 in every cell; baseline varies with shuffle order. Source: \texttt{experiments/results/p6c/swebench\_ab.json}.}
\label{tab:t5_p6c_per_seed}
{\scriptsize\setlength{\tabcolsep}{3pt}%
\input{tables/T5_p6c_per_seed_breakdown.tex}}
\end{table}
```

## What this measures and what it does not

This study measures the system's effect *exactly* on what the system exists to address: *which snippets the agent sees, in which order, with which provenance, with which sublations applied*. It does **not** measure the system's effect on the agent's ability to write correct code, because the patch simulator is a deterministic anchoring function rather than an LLM coder. We deliberately split these concerns. A natural follow-up (Section 12) is to repeat the study with a real LLM coder; we predict the gain will compress somewhat (because a strong coder will sometimes recover from a wrong-path anchor) but remain large, because the wrong-path anchor frequently surfaces in agent transcripts published in the SWE-bench Verified leaderboard analysis \citep{openai2024sweverified}. Crucially, the *same* mechanism — sublation under a shared *avacchedaka* — explains the gains observed on the *non-coding* L1 surfaces (RULER, HELMET, NoCha, HaluEval, TruthfulQA, FACTS-Grounding) in §8; SWE-bench is one challenging *coding* instance of an agent-level effect.

## Sensitivity to budget size (predicted, not in headline JSON)

The headline run is at **512 tokens**. The runner additionally supports `--research-block-budget` ∈ {2 K, 4 K, 8 K, 16 K, 32 K}; we *predict* (and the runner is wired to confirm) monotonically *decreasing* gain as the budget grows: when the budget is large enough to fit the full unshuffled trail, the system's contribution shifts from *truncation* to *sublation*, and the baseline's anchoring bias still dominates the wrong-file outcome. The full budget sweep is left for the next release; the headline JSON in `experiments/results/p6c/_summary.json` covers only the 512-token cell. Figure~\ref{fig:f10}'s paired-diff histogram is positive at this budget.

## What the system contributes, in one sentence

*Under a fixed 512-token research-block budget, on synthetic research trails constructed over the SWE-bench Verified instance set (patch generation deterministically anchored on the first plausible file path of the research block), the system anchors the stub patch on the correct file in **100% of the 6 (model × seed) cells (720 / 720 paired runs)** versus **50.3%** for the budgeted baseline*, with per-instance permutation $p = 0.0005$ and per-(model, seed) $p = 0.03125$ as in Section 10.2. This is not a claim about unconstrained LLM patch generation on the full upstream harness; it is the operational expression of *avacchedaka* + *bādha* + the *first-seen-wins* failure mode of unaided agents under the stated simulator — and the same operational pattern is what produces the Layer-1 long-context and hallucination effects.

## Aggregate across all studies

We close this section with the omnibus statistic. Stouffer-Z combination (Section 7.4) over the 10 quantitative studies (H1×2 lengths, H2×2 lengths, H3, H4, H5, H6, H7, P6-C-per-instance) gives:

| | value |
|---|---|
| n_studies | **10** |
| sum of weights ($\sum \sqrt{n_i}$) | 122.33 |
| combined Z | **9.114** |
| **combined two-sided p** | **7.94 × 10⁻²⁰** |
| mean per-study delta | +0.476 |
| mean Cohen's d (excluding the two structural-100% studies) | 9.62 |

The single combined statistic crosses *every* conventional significance threshold by orders of magnitude. Because several stacked rows share generators and model families, we also report a correlation-corrected Stouffer variant alongside the naïve 10-study value (Table~\ref{tab:t7_omnibus}). Figures~\ref{fig:f12} and \ref{fig:f13} visualise effect-size magnitudes and a forest plot of the per-hypothesis deltas. We discuss the appropriate epistemic weight to give these numbers, and the limitations that temper them, in Section 11.

```{=latex}
\begin{table}[!tbp]
\centering
\caption{\textbf{T7 --- Omnibus Stouffer-Z combination.} Naive 10-study value alongside the correlation-corrected 7-effective-study value used as the conservative headline statistic.}
\label{tab:t7_omnibus}
\small
\input{tables/T7_omnibus_stouffer.tex}
\end{table}

\begin{figure}[!tbp]
\centering
\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth]{F12_effect_sizes.png}
  \caption{Effect-size magnitudes by hypothesis.}
  \label{fig:f12}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth]{F13_forest_plot.png}
  \caption{Forest plot of per-hypothesis paired deltas (95\% CIs).}
  \label{fig:f13}
\end{subfigure}
\caption{\textbf{Aggregate effect across all 10 quantitative studies.} The two structural-100\% rows (H5, H7) are excluded from the mean Cohen's $d$; the headline omnibus uses the correlation-corrected Stouffer-Z (Table~\ref{tab:t7_omnibus}).}
\label{fig:f12f13}
\end{figure}
```
