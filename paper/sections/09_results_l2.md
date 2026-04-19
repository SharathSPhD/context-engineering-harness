# Results — Layer 2: Live Case Study (P6-B)

Layer 2 supplies what L1 cannot: an end-to-end, deterministic, *ecologically-real* trace of the harness in action on real GitHub issues. The case study is consciously *not* an LLM-generated transcript: it is a typed, reproducible flow through `context_insert → sublate_with_evidence → compact → final answer` whose output is exactly the qualifier-set of the surviving live items. This isolates the *context discipline* from generation quality.

## Cases

We hand-picked three real issues drawn from popular Python projects, each with a documented mix of stale-vs-fresh evidence in the actual user-facing search results:

1. **`django_request_body`** — "*When does Django raise `RawPostDataException` after `request.POST`?*" (Django ticket [27592](https://code.djangoproject.com/ticket/27592), `pinned_commit = django-4.2.11`). Older Stack Overflow answers state Django *always* raises `RawPostDataException` after `POST`; the Django 4.1 release notes clarify it is raised *only for form-encoded* requests. The web is awash in the older "always raises" answers.

2. **`requests_retry_adapter`** — "*Correct retry-strategy spelling for `requests` + urllib3 2.x*" (`pinned_commit = requests-2.32.3+urllib3-2.2.2`). Older Stack Overflow answers recommend `urllib3.util.Retry(method_whitelist=...)`; urllib3 1.26 deprecated this in favour of `allowed_methods` and urllib3 2.0 removed `method_whitelist` entirely.

3. **`pandas_iterrows_dtype`** — "*Does `DataFrame.iterrows` preserve column dtypes?*" (`pinned_commit = pandas-2.2.2`, [pandas#15014](https://github.com/pandas-dev/pandas/issues/15014)). Dozens of older answers say yes; the pandas 2.x reference says no — rows are returned as `Series` and promoted to a *common dtype* (typically `object`), with `itertuples()` recommended when dtypes must be preserved.

For each issue we curated 3–5 evidence items (`EvidenceItem`s, see `experiments/v2/p6b/case_data.py`) tagged with `precision`, `condition`, `stale`, and where appropriate `superseded_by_id`.

## Arms

Both arms are deterministic and LLM-free; the difference is purely the *discipline* applied. The **`without_harness` (baseline)** arm processes evidence in *discovery order* and commits to the *first-seen* qualifier as the final answer, simulating the documented *Lost-in-the-Middle* anchoring bias \citep{liu2023lostmiddle} where an unaided LLM agent latches onto the earliest-surfaced (and therefore typically most-popular-and-stale) result. The **`with_harness` (treatment)** arm also processes evidence in *discovery order*, but for each new item it calls `sublate_with_evidence` on any pre-existing item the new item explicitly supersedes (via `superseded_by_id`) *or* on any pre-existing stale item under the same `(qualificand, condition)` whose `precision` is dominated by the new item; it then runs `compact` to drop sublated items from the live set, and the final answer is the qualifier-set of the surviving live items.

## Headline numbers

| metric | with-harness | without-harness | delta |
|---|---|---|---|
| Cases correct | **3 / 3** (100%) | 0 / 3 (0%) | **+3** |
| Forbidden-claim hits (lower-is-better) | **1** | 5 | **−4** |
| Stale items in final live set (LIB) | **0** | 3 | **−3** |
| Total `sublate_with_evidence` events fired | **7** | 0 | — |
| Total `compact` events fired | **3** | 0 | — |
| Context tokens used (matched) | 213 / 128 / 138 | 213 / 128 / 138 | 0 |

Source: `experiments/results/p6b/_summary.json`. Visualised in Figures~\ref{fig:f08} (per-case accuracy) and \ref{fig:f09} (forbidden-claim hits); per-case detail is in Table~\ref{tab:t3_p6b}.

```{=latex}
\begin{figure}[!tbp]
\centering
\begin{subfigure}[t]{0.48\linewidth}
  \centering
  \includegraphics[width=\linewidth]{F08_P6B_per_case_accuracy.png}
  \caption{Per-case correctness (3/3 vs.\ 0/3).}
  \label{fig:f08}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.48\linewidth}
  \centering
  \includegraphics[width=\linewidth]{F09_P6B_forbidden_claims.png}
  \caption{Forbidden-claim hits (lower is better).}
  \label{fig:f09}
\end{subfigure}
\caption{\textbf{P6-B live case study.} Three real-world stale-vs-fresh evidence trails (Django, requests, pandas) processed under \emph{identical} token budgets. The system fires 7 \texttt{sublate\_with\_evidence} events and 3 \texttt{compact} events; the baseline anchors on the first-seen (and typically stale) item.}
\label{fig:f08f09}
\end{figure}

\begin{table}[!tbp]
\centering
\caption{\textbf{T3 --- P6-B per-case detail.} Sublations fired, compactions, forbidden-claim audit, and final-answer correctness for each of the three Python issues. Source: \texttt{experiments/results/p6b/\_summary.json}.}
\label{tab:t3_p6b}
{\scriptsize\setlength{\tabcolsep}{3pt}%
\input{tables/T3_p6b_per_case.tex}}
\end{table}
```

The *context tokens used* row is critical: both arms operate under *identical* token budgets. The harness wins not by buying more context but by *using the same budget more disciplinedly*. The seven `sublate_with_evidence` events and three `compact` events are exactly the operational footprint of the Section 4.3 / 4.7 commitments.

## Per-case narrative

**`django_request_body`.** The baseline anchored on the pre-3.2 advice (the first item in discovery order, since it was the most-upvoted Stack Overflow answer when the issue was filed). The harness saw the same item, accepted it provisionally, then fired `sublate_with_evidence` against the stale Stack Overflow snippets as fresher items arrived (the official Django 4.1 release notes and the Django 4.2 docs); per `experiments/results/p6b/django_request_body.json` (`metrics.sublations`), this case fires **3 sublations** and **1 compaction**, after which the final live set contains only the fresh Django 4.x items (`store_size_active = 4`, `store_size_sublated = 1`). The final answer correctly cited the form-encoded clarification from the 4.1 release notes. In the forbidden-claim audit the baseline emitted one forbidden phrase (`Django 1.x`/`always raises`/`never raises`); the harness emitted zero.

**`requests_retry_adapter`.** Stale Stack Overflow snippets used the old `Retry(method_whitelist=...)` parameter; the fresh urllib3 v2 doc snippet uses `allowed_methods=...`. Per the per-case JSON (`requests_retry_adapter.json`, `metrics.sublations`), the harness fires **2 sublations** and **1 compaction** and commits to `allowed_methods`. The baseline locked onto `method_whitelist` and never recovered. In the forbidden-claim audit the baseline emitted two forbidden references (e.g.\ `method_whitelist`, `requests.packages.urllib3`); the harness emitted one — a single `method_whitelist` mention surviving inside the *sublation event's* explanatory `reason` string for audit purposes only, with zero references in the final live qualifier set.

**`pandas_iterrows_dtype`.** Stale snippets all said "iterrows preserves dtype"; the fresh pandas-2.x doc snippet says "no, use `itertuples` (or accept that rows are promoted to a common dtype)". Per `pandas_iterrows_dtype.json` (`metrics.sublations`), the harness fires **2 sublations** and **1 compaction**, leaving only the fresh pandas-2.x items in the live set, and commits to the *common-dtype / `itertuples`* answer. The baseline committed to the wrong answer. In the forbidden-claim audit the baseline emitted two forbidden phrases (`preserves the column dtype` and/or `you can rely on integer`); the harness emitted zero.

## Why this matters

The L2 case study is the smallest possible *ecologically valid* test of the harness's central claim: *the discipline pays off even with no model-generation step, even under identical token budgets, on real-world stale-vs-fresh evidence trails*. The 3-of-3 result is small in *n* but *epistemically large*: it shows the mechanism is not benchmark-specific. The benchmark we constructed for L3 (Section 10) makes the same point at $n=120$ with multi-seed multi-model paired statistics.

## Reproducibility

The case-study runner is `experiments/v2/p6b/run_case_study.py`. The case data is `experiments/v2/p6b/case_data.py`. Both are deterministic and LLM-free. Re-running `python -m experiments.v2.p6b.run_case_study` re-emits `experiments/results/p6b/*.json` byte-identically.
