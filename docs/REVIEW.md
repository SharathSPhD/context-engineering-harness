# Pratyakṣa v2 — Consolidated Internal Review (P9)

This document consolidates the six independent reviewer passes performed on the v2 worktree before public release. It is the input to the P10 ship checklist.

| # | Reviewer persona | Document | Verdict |
|---|---|---|---|
| 1 | code-reviewer (general) | [`REVIEW_code_reviewer.md`](REVIEW_code_reviewer.md) | **Block** — 3 critical, 7 major, 7 minor |
| 2 | kieran-python-reviewer (Pythonic strict) | [`REVIEW_kieran_python.md`](REVIEW_kieran_python.md) | **Pass with required changes** |
| 3 | adversarial-reviewer (try-to-break) | [`REVIEW_adversarial.md`](REVIEW_adversarial.md) | **Patch and ship** — narrative & stats need honesty |
| 4 | coherence-reviewer (paper internal consistency) | [`REVIEW_paper_coherence.md`](REVIEW_paper_coherence.md) | **Block** — multiple cross-section contradictions |
| 5 | feasibility-reviewer (will-it-actually-work) | [`REVIEW_paper_feasibility.md`](REVIEW_paper_feasibility.md) | **Block** — repro recipe does not match repo |
| 6 | scope-guardian-reviewer (scope discipline) | [`REVIEW_paper_scope.md`](REVIEW_paper_scope.md) | **Patch and ship** — three over-claims to cut |

## A. Cross-reviewer must-fix list (12 items)

These are the items where two or more reviewers independently flagged the same issue. They must all be addressed before tagging `v2.0.0`.

### A1. Witness-log path drift  *(adversarial + feasibility)*

- Paper (Appendix B, §6.4) says: `${CLAUDE_PLUGIN_ROOT}/witness_log.jsonl`.
- Reality: `~/.cache/pratyaksha/audit.jsonl`.
- **Fix**: align — pick one canonical path and update both sides. We chose `~/.cache/pratyaksha/audit.jsonl` (XDG-compatible, survives plugin reinstall) and updated the paper.

### A2. Lifecycle hook semantics  *(adversarial + feasibility)*

- Paper says `pretooluse-budget.sh` *denies* tool use when budget exceeded.
- Reality: it only logs and never denies.
- **Fix**: the paper now describes the hook as **advisory** with optional strict mode (`PRATYAKSHA_BUDGET_STRICT=1`) and the corresponding strict path is implemented in the script.

### A3. Khyātivāda classifier description drift  *(adversarial + feasibility + coherence)*

- Paper (§5.5, Appendix B/D) describes a few-shot Anthropic JSON classifier with rule-based guardrails.
- Reality: shipped path is heuristic-only; the few-shot Anthropic classifier exists in the experiments harness (`src/evaluation/khyativada_fewshot.py`) but is *not* wired into the plugin's `classify_khyativada` MCP tool.
- **Fix**: paper now states explicitly that the plugin ships the heuristic + guardrail classifier and that the few-shot Anthropic variant is an *experiment-only* path. Both are independently evaluated in §8 (H6).

### A4. Reproducibility recipe wrong  *(feasibility + coherence)*

- Paper (Appendix C, §7.6.2) cites module names like `experiments.v2.p6a.run_h1_h2` that do not exist; the actual entry point is `experiments.v2.p6a.run`. Wrong CLI flags. Wrong P6-C default.
- **Fix**: regenerated Appendix C from the actual command surface; the paper now matches `python -m experiments.v2.p6a.run --hypothesis H1 --seeds 1 2 3`.

### A5. P6-C budget mismatch  *(feasibility + adversarial)*

- Paper claims an 8 K-token research-block budget for P6-C.
- Code default in `experiments/v2/p6c/run_swebench_ab.py`: 512 tokens; the 8 K runs are explicit `--research-block-budget 8192` invocations.
- **Fix**: the paper's claim is correct *for the runs we report* but the code default was misleading. We changed the code default to 8192 to match the reported headline number and added a `--research-block-budget-fast 512` smoke-test path.

### A6. Stouffer-Z independence assumption  *(adversarial + coherence)*

- 10 studies stacked include H1@8K + H1@32K + H2@8K + H2@32K, which share the long-context generator and model family — so they are *correlated*, not independent.
- **Fix**: paper now reports both the naïve Stouffer-Z **and** a *correlation-corrected* effective-N Stouffer-Z that collapses each `(hypothesis, family)` row to one effective study (resulting in 7 effective studies instead of 10). Both numbers are now stated; the headline keeps the conservative 7-study value as primary.

### A7. SWE-bench A/B "synthetic" caveat  *(adversarial + scope-guardian)*

- The 100%-vs-50.3% claim is based on a heuristic `PatchSimulator` over a synthetic `research_trail`. The paper's framing implies it is "real SWE-bench Verified".
- **Fix**: the abstract and §10 now clearly state: *"on synthetic research trails over the SWE-bench Verified instance set, with patch generation deterministically anchored on the first plausible file path of the research block; the optional Docker scorer agreement is reported on a 30-instance sub-sample (κ = 0.97)."* The headline number is unchanged; its scope is corrected.

### A8. "100% in 6/6 cells" framing  *(coherence + adversarial)*

- The plain-English "100% target-path-hit rate" should be qualified as "100% in 6 of 6 (model × seed) cells, 720 / 720 of paired runs". This makes the per-(model,seed) p = 0.03125 = 1/32 visible as the floor of the permutation test.
- **Fix**: paper now reports both the per-instance permutation p (0.0005) and the per-(model,seed) p (0.03125), explicitly noting the latter is the test-floor and supplementing it with bootstrap CIs.

### A9. Tool-name drift across sections  *(coherence + adversarial)*

- §4–5 use shorthand names like `insert`, `compact_now`.
- §6, Appendix B, the actual MCP server use `context_insert`, `compact`.
- **Fix**: every section now uses the canonical shipped names; the shorthand is explicitly noted as such in §4 with the canonical name in parentheses.

### A10. Manas: skill or agent?  *(coherence)*

- §4.8 calls Manas a "skill" in passing.
- §6 and Appendix D ship Manas as an **agent** with a system prompt.
- **Fix**: §4.8 now reads "Manas is operationalised as a sub-agent (Section 6.3)". Skill terminology removed from §4.

### A11. Abstract "five public benchmarks" miscount  *(coherence)*

- Abstract says "five public benchmarks" then lists six (RULER, HELMET, NoCha, HaluEval, TruthfulQA, FACTS-Grounding) plus SWE-bench Verified for L3.
- **Fix**: corrected to "six public benchmarks for L1 plus SWE-bench Verified for L3".

### A12. "Nine L1 hypotheses" typo  *(coherence)*

- §11.1.1 mentions "nine L1 hypotheses". H1–H7 = seven.
- **Fix**: corrected to "seven L1 hypotheses (H1–H7)".

## B. Code-only must-fix list (3 items, code-reviewer critical)

### B1. `sublate_with_evidence` non-idempotent

- Repeated calls on the same `target_id` now return the same row without flipping `precision` again, and the audit log records only the first sublation. A second call returns `{"already_sublated": true}` and does *not* re-emit a witness event.

### B2. `new_id` ms-collision

- Replace `time.time_ns() // 1_000_000` collision-prone IDs with UUIDv7 (time-ordered UUIDs). Two inserts in the same ms now get distinct IDs.

### B3. Retrieval ignores `qualifier`

- Add a `qualifier_substring` filter to `context_retrieve`'s rule engine (already present in the input schema; was unused in the implementation). Tests added.

## C. Code-only should-fix list (kieran-strict)

- Replace silent `except Exception` in `server._count_tokens` with a typed `tiktoken.TokenizerError` catch.
- Drop unused imports (`re`, `Sequence`, `sys`, `_ALL_CODE_RE`) flagged in `run_swebench_ab.py`.
- Tighten `make_caller`'s return type from `"callable"` to `Callable[[ModelCallerArgs], ModelOutput]`.
- Rename `n_seeds_used` → `n_seeds`, `n_pairs` (when sourced from `n_examples_used`) → `n_examples` for honesty.
- Wrap P7 table generation in the same `try/except` the figure generation uses.
- Replace pervasive `dict[str, Any]` API surface with TypedDict / Pydantic where the surface is public.

## D. Paper-only must-fix list (scope + coherence)

- Cut "*exactly* the vocabulary" from the abstract; replace with "a vocabulary that operationally fits".
- Cut §1.4's "TRIZ generalises beyond this paper" — restate as method note only.
- Cut §3.3's "field-wide rediscovering" overreach.
- Soften §11.3's "cross-cultural agreement is evidence about something real about cognition" to "is suggestive convergence, not yet evidence" and add a one-sentence caveat.
- Soften §12's "industry-wide MCP contract" and "neuroscience cross-checks" future-work items to flag them as *aspirational*.
- Fix Appendix A's Svataḥ/parataḥ → Bayesian-prior row: either (a) implement the split in `src/aggregation/bayesian.py` and cite it, or (b) move the row to a "*conjectured but not implemented*" sub-table. Chosen: (b).
- Fix Section 1.4's mis-pointer to Appendix F (it should point to Appendix C, the reproducibility manifest).
- Fix Appendix E being incorrectly cited for Khyātivāda IAA (should cite Appendix D for prompts and §8 for IAA results).
- Re-align Manas JSON schema across §4.4, §6.3, Appendix D — pick one and use it everywhere.

## E. Strengths preserved (positive findings)

These were independently identified by multiple reviewers as *good* and should be **kept** through any subsequent refactor:

- **Test rigour**: 500+ tests with substantive assertions (not vacuous traces). Especially the `tests/test_v2/test_p7_aggregate.py::test_aggregate_is_deterministic` byte-identity check.
- **PEP 723 dependency contract**: `mcp/server.py` runs with `mcp + pydantic + tiktoken + numpy + anthropic` and nothing more. Optional `vllm`/`transformers` is genuinely opt-in.
- **`triz-engine` / `attractor-flow` / `ralph-loop` runtime independence is real** — `grep` finds the strings only in markdown / marketplace copy (acknowledged provenance), never in Python imports of the shipped plugin.
- **P6-C deterministic seeding**: bootstrap and permutation tests are seeded, results are byte-deterministic.
- **The aggregator's `_index.json` is deterministic** modulo timestamp — the byte-equality test is a model others should adopt.

## F. Release verdict

**Block release until items in A and B are addressed**; items in C and D may be folded in opportunistically with the same release. After the must-fix items are merged, re-run the full pytest suite (target: 500+ passing, 0 failing) and re-build the paper. Then proceed to P10.
