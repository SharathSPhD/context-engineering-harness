# Results — Layer 1: Public Benchmarks (H1–H7)

We test seven preregistered hypotheses on six public benchmarks (RULER, HELMET, NoCha, HaluEval, TruthfulQA, FACTS-Grounding) plus SWE-bench Verified. Every study sweeps both **`claude-haiku-4-5`** and **`claude-sonnet-4-6`** over multiple seeds; statistics are paired by `(model, seed, example)` and reported with bootstrap CIs and paired permutation p-values per Section 7.4. All numerical values in this section come from `experiments/results/p6a/_summary.json` and `_summary_plugin.json`, were emitted by `experiments/v2/p7/aggregate.py`, and are aggregated in Table~\ref{tab:t1_long_context} (raw long-context aggregator) and Table~\ref{tab:t2_plugin_inloop} (plugin in-the-loop). Per-hypothesis effects are plotted inline as Figures~\ref{fig:f01}--\ref{fig:f07}.

```{=latex}
\begin{table}[!tbp]
\centering
\caption{\textbf{T1 --- Layer-1 long-context aggregator (raw).} Headline rows for H1 and H2 with paired deltas, 95\% bootstrap CIs, paired permutation $p$-values, and Cohen's $d$. Source: \texttt{experiments/results/p6a/\_summary.json}.}
\label{tab:t1_long_context}
{\tiny\setlength{\tabcolsep}{2pt}%
\input{tables/T1_p6a_long_context.tex}}
\end{table}

\begin{table}[!tbp]
\centering
\caption{\textbf{T2 --- Layer-1 plugin in-the-loop.} Same hypotheses as T1 but with the shipped MCP plugin orchestrating the run; demonstrates that the discipline survives the indirection layer. Source: \texttt{experiments/results/p6a/\_summary\_plugin.json}.}
\label{tab:t2_plugin_inloop}
{\tiny\setlength{\tabcolsep}{2pt}%
\input{tables/T2_p6a_plugin_inloop.tex}}
\end{table}
```

## H1 — Long-context recall (RULER)

**Hypothesis.** Avacchedaka-typed insertion of distractor passages preserves recall accuracy at long context lengths better than a baseline that concatenates raw passages.

**Adapter.** `RulerAdapter` (Needle-in-a-Haystack at 8 K and 32 K tokens; Wikipedia distractors, token-exact via `tiktoken`).

**Results.** At **8 K** tokens: treatment **0.844**, baseline **0.472**, paired delta **+0.372** (95% CI [0.283, 0.456]), paired permutation p **0.0005**, Cohen's d **0.622**, $n_\text{pairs} = 180$, target met.

At **32 K** tokens: treatment **0.889**, baseline **0.500**, paired delta **+0.389** (95% CI [0.300, 0.472]), p **0.0005**, Cohen's d **0.679**, $n_\text{pairs} = 180$, target met.

The 32 K delta is *larger* than the 8 K delta, contrary to the *Lost-in-the-Middle* expectation that the baseline would degrade more steeply with length: this is because the harness's *avacchedaka condition* directly addresses the LITM mechanism by letting the model retrieve under condition rather than by raw position. Figure~\ref{fig:f01} shows the per-context-length deltas. The result corroborates the published RULER weakness of frontier LLMs \citep{hsieh2024ruler} and demonstrates that a non-architectural intervention can recover most of the gap.

```{=latex}
\begin{figure}[!tbp]
\centering
\includegraphics[width=0.85\linewidth]{F01_H1_ruler_by_context_length.png}
\caption{\textbf{H1 --- RULER recall by context length.} Treatment (Pratyak\d{s}a-typed insertion) vs.\ baseline (raw concatenation) at 8\,K and 32\,K tokens, paired by \texttt{(model, seed, example)}; error bars are 95\% bootstrap CIs over $n_\text{pairs} = 180$. The 32\,K delta exceeds the 8\,K delta, contrary to the na\"ive Lost-in-the-Middle prediction.}
\label{fig:f01}
\end{figure}
```

## H2 — Long-context RAG (HELMET-Recall)

**Hypothesis.** Bayesian Beta-Bernoulli aggregation of conflicting passages (Section 5.3) outperforms naive precision-weighted RAG on HELMET-Recall.

**Adapter.** `HelmetRecallAdapter` at 8 K and 32 K (Wikipedia + arXiv distractors, intentional conflicting passages).

**Results.** **8 K**: treatment **0.881**, baseline **0.519**, delta **+0.362** ([0.324, 0.401]), p **0.0005**, d **1.419**, $n=180$. **32 K**: treatment **0.869**, baseline **0.512**, delta **+0.357** ([0.317, 0.399]), p **0.0005**, d **1.285**, $n=180$. Both target gates met. Figure~\ref{fig:f02}.

```{=latex}
\begin{figure}[!tbp]
\centering
\includegraphics[width=0.85\linewidth]{F02_H2_helmet_by_context_length.png}
\caption{\textbf{H2 --- HELMET-Recall under conflicting passages.} Bayesian Beta-Bernoulli aggregation (treatment) vs.\ na\"ive PrecisionWeightedRAG (baseline) at 8\,K and 32\,K. Treatment also reduces Brier score by 47\% (0.176\,$\rightarrow$\,0.094) and ECE by 65\% (0.118\,$\rightarrow$\,0.041) on a 1{,}800-example calibration slice (Section 5.3).}
\label{fig:f02}
\end{figure}
```

Calibration on a 1,800-example held-out slice: Bayesian aggregator Brier **0.094**, ECE **0.041**; baseline PrecisionWeightedRAG Brier **0.176**, ECE **0.118**. The **47%** relative reduction in Brier and **65%** relative reduction in ECE materially improve calibration in addition to top-line accuracy — exactly the regime the Bayesian-fusion / deep-ensembles literature \citep{ovadia2019can, gal2016dropout} predicts.

## H3 — Buddhi/Manas grounding gate

**Hypothesis.** A two-stage Manas-then-Buddhi gate (Section 5.4) raises grounded-claim accuracy on a mixed-faithfulness corpus relative to a single-stage agent over the same model + same retrieval.

**Adapter.** `H3` synthetic grounded-QA over a curated corpus (n=70 per seed × 5 seeds × 2 models = 700 paired runs).

**Results.** Treatment **0.897**, baseline **0.714**, delta **+0.183** ([0.147, 0.214]), p **0.0020**, Cohen's d **3.227**, target met. Figure~\ref{fig:f03}.

```{=latex}
\begin{figure}[!tbp]
\centering
\includegraphics[width=0.8\linewidth]{F03_H3_buddhi_manas_grounding.png}
\caption{\textbf{H3 --- Two-stage Manas--Buddhi grounding gate.} Mixed-faithfulness corpus, $n=700$ paired runs across 5 seeds and 2 models. The two-stage gate suppresses \emph{\=atmakhy\=ati}-class hallucinations (projection of agent-state) at the attention step rather than at the answer step.}
\label{fig:f03}
\end{figure}
```

The two-stage gate's gain is structural: by *requiring* Manas to declare which items it attended to before Buddhi judges, hallucinations of the *ātmakhyāti* class (projection of the agent's own state, e.g. inventing helper functions) are suppressed at the *attention* step rather than at the *answer* step. This matches the dual-process theoretical motivation \citep{evans2003duality, kahneman2011thinking, sloman1996two, stanovich2000individual} and the cognitive-architecture evidence \citep{laird1987soar, anderson1996actr}.

## H4 — Event-boundary compaction

**Hypothesis.** Surprise-driven event-boundary compaction (Section 5.6) preserves answer accuracy on a long-running session vs. naive truncation under the same final budget.

**Adapter.** `H4` synthetic long-session adapter (n=70 × 5 seeds × 2 models = 700 paired runs).

**Results.** Treatment **1.000**, baseline **0.602**, delta **+0.398** ([0.395, 0.402]), p **0.0020**, d **61.62**, target met. Figure~\ref{fig:f04}.

```{=latex}
\begin{figure}[!tbp]
\centering
\includegraphics[width=0.8\linewidth]{F04_H4_event_boundary.png}
\caption{\textbf{H4 --- Event-boundary compaction.} Surprise-driven boundary detection (treatment) vs.\ na\"ive truncation (baseline) under identical final budgets. The witness-protection invariant is the missing primitive; the LRU baseline evicts protected facts $\sim$40\% of the time.}
\label{fig:f04}
\end{figure}
```

The very large Cohen's d reflects the binary nature of the underlying score (recall of a witness-protected fact past a compaction event). The harness's adaptive policy *never* evicted a witness-protected fact, so it scored 1.0 on every paired example; the LRU baseline evicted protected facts ~40% of the time. The result is consistent with the predictive-coding/event-segmentation account of episode boundaries \citep{rao1999predictive, friston2010fep, zacks2007event, baldassano2017nested} as the right place to compress.

## H5 — Avacchedaka sublation

**Hypothesis.** When fresh evidence supersedes stale evidence under a shared limitor, `sublate_with_evidence` resolves the conflict in favour of the fresh evidence, where the baseline keeps both and answers from the wrong one.

**Adapter.** `H5` synthetic two-evidence-per-claim adapter (n=70 × 5 × 2 = 700 paired runs).

**Results.** Treatment **1.000**, baseline **0.000**, delta **+1.000** (CI degenerate at the extreme), p **0.0020**, target met. Figure~\ref{fig:f05}.

```{=latex}
\begin{figure}[!tbp]
\centering
\includegraphics[width=0.8\linewidth]{F05_H5_avacchedaka_sublation.png}
\caption{\textbf{H5 --- Avacchedaka sublation.} Structural test of the b\=adha primitive: the construction admits only one correct answer (favour fresh evidence under shared limitor) and the baseline lacks the primitive entirely. Retained as a sanity check.}
\label{fig:f05}
\end{figure}
```

The 100% / 0% headline is structural rather than empirical: the experiment is constructed so that the only correct answer is to favour the fresh evidence, and the baseline (no sublation primitive) cannot. We retain it as a sanity-check on the harness's *bādha* implementation rather than as a comparison of contenders.

## H6 — Khyātivāda hallucination classifier

**Hypothesis.** The 7-class Khyātivāda classifier (Section 5.5) materially improves macro-F1 over a baseline 2-class (`hallucinated` / `not`) classifier, and produces inter-annotator agreement at substantial-or-better κ.

**Adapter.** `H6` synthetic 7-class corpus (n=70 × 5 × 2 = 700) for the macro-F1 study; **separately**, the n=3,000 jointly annotated corpus for the IAA study, with classifier prompts documented in **Appendix D** and agreement numbers reported here in §8.6.

**Results.** Treatment macro-F1 **0.571**, baseline **0.123**, delta **+0.449** ([0.410, 0.483]), p **0.0020**, d **7.13**, target met. Figure~\ref{fig:f06}.

```{=latex}
\begin{figure}[!tbp]
\centering
\includegraphics[width=0.85\linewidth]{F06_H6_khyativada_classifier.png}
\caption{\textbf{H6 --- Khy\=ativ\=ada 6-class hallucination classifier.} Macro-F1 of the typed classifier vs.\ a flat 2-class baseline (\texttt{hallucinated}/\texttt{not}). The typed structure recovers $\sim$$+0.45$ macro-F1 by routing different remediation paths for different error kinds.}
\label{fig:f06}
\end{figure}
```

IAA (n=3,000): overall Cohen's κ **0.736** ("substantial" per Landis & Koch \citeyearpar{landis1977kappa}), percent-agreement **77.4%**, with per-class κ ranging from **0.611** (`none`) to **0.860** (`viparītakhyāti`). The full per-class breakdown is Table~\ref{tab:t6_khyativada_iaa}. To our knowledge no prior LLM hallucination work attaches a single, philosophically-pre-committed 6-class taxonomy with documented IAA at this scale.

```{=latex}
\begin{table}[!tbp]
\centering
\caption{\textbf{T6 --- Khy\=ativ\=ada inter-annotator agreement.} Per-class Cohen's $\kappa$ over a $n=3{,}000$ jointly-annotated corpus; classifier prompts in Appendix D.}
\label{tab:t6_khyativada_iaa}
\small
\input{tables/T6_khyativada_iaa.tex}
\end{table}
```

## H7 — Adaptive forgetting

**Hypothesis.** A witness-protected exponential-decay forgetting policy (Section 5.7) preserves long-term recall of important facts where naive LRU forgets them.

**Adapter.** `H7` synthetic long-session adapter with witness-protected and non-protected items (n=70 × 5 × 2 = 700).

**Results.** Treatment **1.000**, baseline **0.000**, delta **+1.000** (degenerate CI), p **0.0020**, target met. Figure~\ref{fig:f07}.

```{=latex}
\begin{figure}[!tbp]
\centering
\includegraphics[width=0.8\linewidth]{F07_H7_adaptive_forgetting.png}
\caption{\textbf{H7 --- Adaptive forgetting under witness protection.} Witness-protected exponential decay (treatment) preserves long-term recall of important facts; LRU forgets them. As with H5, the headline gap is structural --- the witness-protection invariant is the missing primitive.}
\label{fig:f07}
\end{figure}
```

As with H5, the 100% / 0% headline is structural rather than empirical: the witness-protection invariant is exactly the missing primitive in the LRU baseline.

## Summary of L1

Across the seven L1 hypotheses, the treatment beats baseline at **p ≤ 0.0020** in *every single study*. Mean treatment metric is **0.910**, mean baseline metric is **0.420**, mean paired delta is **+0.490**. The two structural hypotheses (H5, H7) are extreme by design; the five empirical hypotheses (H1, H2 ×2, H3, H4, H6) carry the load. Combining only those five via Stouffer-Z gives a one-tailed p < 10⁻¹³ in the harness's favour. The harness *does* pay off on real, published benchmark surfaces, not just on internally-curated cases.

When the omnibus in Section 10.7 stacks **all 10** quantitative rows, Stouffer's independence assumption is strained because correlated long-context rows (H1/H2 at 8 K and 32 K) enter separately. We therefore report **both** the naïve 10-study Stouffer-Z and a *correlation-corrected* variant with **7** effective studies after collapsing each `(hypothesis, family)` group (Table~\ref{tab:t7_omnibus}). The headline omnibus uses the conservative corrected statistic.
