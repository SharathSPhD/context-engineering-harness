# Pratyaksha v2 — Code Review (code-reviewer pass)

## Summary

- **Total LOC in scope (Python, approximate):** ~14,200 lines across `plugin/pratyaksha-context-eng-harness/mcp/`, `src/`, `experiments/v2/`, and `tests/test_v2/` (full repo Python tree is ~21,600 lines including benchmarks, validation, and other tests).
- **Test count:** **501** tests collected under `tests/` (pytest); **`tests/test_v2` alone: 115** tests.
- **Pass/Fail recommendation for v2.0.0 release:** **Fail (blockers present).** Ship only after fixing the **Critical** items below—especially `sublate_with_evidence` idempotence and sublator id uniqueness. Separately, align shipped **semantic versioning** (`plugin.json` / `marketplace.json` still read **1.0.0** while the review targets **v2.0.0**).

**Scope note:** This worktree implements the “harness” as `src/` (e.g. `src/avacchedaka/store.py`, `src/agents/orchestrator.py`), not a top-level `harness/` directory. The user-requested paths `mcp/orchestrator.py` and `mcp/khyati_prompts.py` **do not exist** in the shipped plugin; orchestration lives in `src/agents/orchestrator.py`, and Khyātivāda prompting/classification lives under `src/evaluation/` (e.g. `khyativada_fewshot.py`).

---

## Critical (must-fix before release)

1. **`plugin/pratyaksha-context-eng-harness/mcp/server.py:422-444` — `sublate_with_evidence` is not idempotent and can corrupt the store.**  
   **Finding:** After an element is sublated, `older.precision` becomes `0.0`, but a subsequent `sublate_with_evidence` call for the same `older_id` still passes the guard `newer_precision <= older.precision` whenever `newer_precision > 0`. That allows **repeated sublation** of the same logical row, creating extra “sublator” elements and mutating `sublated_by` chains in ways no caller should rely on. In the same millisecond, `new_id = f"{args.older_id}__sublator__{int(time.time() * 1000)}"` can **collide**, causing **silent overwrite** of the sublator row in `STATE.elements[new_id] = newer`.  
   **Repro (local):** Insert `old` @ precision 0.3; call `sublate_with_evidence` twice in a tight loop—second call returns `ok: True` and can reuse the same `newer_id` timestamp bucket.  
   **Fix:** Reject when `older.sublated_by is not None` (or when `older.precision == 0.0` and already sublated—pick one invariant and enforce it everywhere). Generate `new_id` with **uuid4** or a **monotonic** counter, not `time.time()*1000` alone.

2. **`experiments/v2/p6c/run_swebench_ab.py:109-142` — harness arm can fire duplicate dominance sublations against stale snippets.**  
   **Finding:** `_build_research_block_with_harness` keeps Python-side `inserted[id] -> ResearchSnippet` objects whose `precision` **does not track** the in-process store after `sublate_with_evidence`. For every later fresh snippet, `triggers_dominance` can remain true for the same stale id because `older.precision` in the dict is still the original >0 value, even though the store already set that element’s precision to `0.0`. Combined with the server bug above, this amplifies duplicate sublator inserts during a single trial.  
   **Fix:** After each successful sublation, update local bookkeeping (or consult `context_get` / store state) so stale rows are not re-eligible; alternatively move dominance checks to store-backed precision.

3. **`plugin/pratyaksha-context-eng-harness/mcp/server.py:231-240` and `src/avacchedaka/query.py:12-22` — `qualifier` is documented as part of the typed limitor but is ignored for retrieval matching.**  
   **Finding:** `RetrieveInput.qualifier` / `AvacchedakaQuery.qualifier` never participates in `_matches` / `matches`. Any two elements that share `qualificand` + `condition` but differ in `qualifier` are indistinguishable to retrieval, contradicting the plugin manifest and server docstrings (“typed (qualificand, qualifier, condition)”).  
   **Fix:** If qualifier is part of the logical key, enforce equality (or explicit wildcard semantics) in both code paths; if not, remove it from the public contract and docs to avoid silent misuse.

---

## Major (should-fix before release)

1. **`plugin/pratyaksha-context-eng-harness/.claude-plugin/plugin.json:3-4` and `plugin/pratyaksha-context-eng-harness/marketplace.json:11-12` — version skew vs v2.0.0 target.**  
   **Finding:** Both files declare **version `1.0.0`** while this review is explicitly for a **v2.0.0** release candidate. Downstream tooling, update checks, and reproducibility footers will disagree with repository semantics.  
   **Fix:** Bump and keep manifest/marketplace/project versions aligned (and document breaking changes if any).

2. **`plugin/pratyaksha-context-eng-harness/mcp/server.py:628-664` — `budget_status` reads the entire cost ledger into memory.**  
   **Finding:** `COST_LEDGER.read_text().splitlines()` followed by full JSON parse accumulates **unbounded RAM** as `cost_ledger.jsonl` grows across months of use. Same pattern risk for audit log readers if added later.  
   **Fix:** Stream tail-only records (deque of last N), or mmap / reverse-read from EOF; cap `ledger_recent` construction without parsing the whole file.

3. **`plugin/pratyaksha-context-eng-harness/mcp/server.py:28-30,65-72` — audit log growth is unbounded.**  
   **Finding:** Every mutating tool appends to `audit.jsonl` with no rotation, size cap, or sampling—fine for short sessions, risky for always-on agents.  
   **Fix:** Rotate by size/date, or make retention configurable via `PRATYAKSHA_AUDIT_MAX_MB` / similar.

4. **`experiments/v2/p7/aggregate.py:141-156` — Stouffer combination inputs are two-sided p-values.**  
   **Finding:** `stouffer_combine` maps each `p` through `_norm_inv_cdf(1.0 - p)` and then doubles the tail via `_norm_sf(abs(z))`. Combining **two-sided** p-values with a one-sided Z construction is a known stats foot-gun; the omnibus number may be **optimistic or anti-conservative** depending on how primary p-values were computed.  
   **Fix:** Document the mapping as an explicit **heuristic** in `_summary.md`, or switch to a documented method (e.g. one-sided p directed by sign, or Fisher/Lancaster with clear assumptions).

5. **`tests/test_v2/test_p7_aggregate.py:51-84` — golden assertions depend on committed artifacts outside `tmp_path`.**  
   **Finding:** `aggregate.load_artifacts()` reads fixed paths under `experiments/results/...` in the repo, while only outputs go to `tmp_path`. Tests like `test_aggregate_emits_every_figure_and_table` and `test_aggregate_omnibus_passes_significance` therefore **fail in a clean checkout** without generated results and can **pass for the wrong reason** if artifacts drift.  
   **Fix:** Parametrize artifact dir into `load_artifacts`, ship minimal fixture JSON under `tests/fixtures/p7/`, and assert structure—not hard-coded “≥7 significant” unless those fixtures are version-locked.

6. **`experiments/v2/p6a/plugin_client.py:44-84` — global singleton server module + shared `STATE`.**  
   **Finding:** All `PratyakshaPluginClient` instances alias the same module-level `STATE`. `reset()` mutates fields in place, which is OK for serial tests but is **unsafe under parallel pytest (xdist)** or overlapping async tasks.  
   **Fix:** Thread-local / contextvar store, or per-client `ContextStore` object instead of module singleton for in-proc harness.

7. **`src/forgetting/schedules.py:80-96` — `BadhaFirstForgetting` semantics vs docstring.**  
   **Finding:** The class claims to “clear sublated elements first”, but it iterates elements with `sublated_by is not None` and sets `precision=0.0` again—typically a no-op if sublation already zeroed precision, and it does **not** remove or GC rows. Not a crash bug, but operators may misunderstand what “cleared” means.  
   **Fix:** Clarify docstring: “normalize precision on already-sublated rows” or implement a distinct policy (e.g. tombstone flag) if space reclamation is intended.

---

## Minor (nice-to-have, ship anyway)

1. **`plugin/pratyaksha-context-eng-harness/mcp/server.py:107-109` — broad `except Exception` in `_count_tokens`.**  
   **Finding:** Masks unexpected tiktoken failures beyond import/encoding issues. Acceptable for a soft budget tool, but logs lack exception type.  
   **Fix:** `logger.debug(..., exc_info=True)` at TRACE level or narrow exceptions.

2. **`src/utils/tokenizer.py:40-45` vs `plugin/.../server.py:98-109` — duplicated token counting logic.**  
   **Finding:** Two implementations (one logs on missing tiktoken, one silent heuristic). Minor divergence risk over time.  
   **Fix:** Share one helper or document “plugin must stay self-contained” explicitly (already partly true).

3. **`src/avacchedaka/element.py:19` — `datetime.utcnow` default factory.**  
   **Finding:** Deprecated semantics in Python 3.12+ docs; naive UTC timestamps are fine here but lint noise will grow.  
   **Fix:** `datetime.now(timezone.utc)` with `datetime` type, or `field(default_factory=lambda: datetime.now(timezone.utc))`.

4. **`experiments/v2/p6c/run_swebench_ab.py:394-397` — dead / confusing guard.**  
   **Finding:** `if not isinstance(caller, type(MockHarnessCaller)): pass` is always true for instance `caller` and communicates nothing.  
   **Fix:** Remove or replace with an actual live-caller branch.

5. **`experiments/v2/p6c/run_swebench_ab.py:175` — `_truncate_to_budget` uses `truncated.count("- ")` as a proxy for kept bullets.**  
   **Finding:** Fragile if evidence text contains hyphen-prefixed phrases. OK for controlled fixtures, misleading telemetry otherwise.  
   **Fix:** Count snippets by delimiter or track index while truncating.

6. **`experiments/v2/p7/aggregate.py:789-795` — `except Exception` around each figure.**  
   **Finding:** Swallows figure failures except for logs; tables still run. Reasonable for batch robustness, but CI might miss matplotlib regressions.  
   **Fix:** Optional `--strict` flag to re-raise.

7. **`experiments/v2/p7/aggregate.py:197-205` — P6-B row uses degenerate CI (`ci_low == ci_high == accuracy_delta`).**  
   **Finding:** Intentional sentinel for deterministic case study (also noted in comments for F12), but easy to misread as a real interval.  
   **Fix:** Explicit `ci_kind: "point"` metadata in derived rows.

---

## Strengths worth keeping

- **Single source of truth for plugin behaviour in experiments:** `experiments/v2/p6a/plugin_client.py` loads the real `mcp/server.py` instead of duplicating tool semantics—excellent for preventing drift between “what Claude calls” and “what the paper measures.”
- **Statistics core is auditable:** `src/benchmarks/stats.py` keeps bootstrap / paired permutation / Cohen’s d in pure NumPy with explicit edge cases (empty arrays, exact enumeration when small `n`).
- **Surprise stack degrades gracefully:** `src/compaction/surprise.py` documents backend selection (`vllm` → `transformers` → heuristic) without hard-failing imports—good production posture for optional heavy deps.
- **Hooks are intentionally non-blocking:** `hooks/pretooluse-budget.sh` and `hooks/stop-compact.sh` always `allow` / no-op on missing `jq` or gauge—correct failure mode for IDE integrations.
- **P6-C experimental design is crisp:** shared deterministic `research_evidence.generate_research_trail` + paired arms under identical budgets makes the failure mode legible (`run_swebench_ab.py` docstring matches implementation structure).

---

## Files reviewed

**Plugin runtime (`plugin/pratyaksha-context-eng-harness/`)**

- `mcp/server.py`
- `mcp/smoke_test.py`
- `.mcp.json`
- `.claude-plugin/plugin.json`
- `marketplace.json`
- `hooks/hooks.json`
- `hooks/session-start.sh`
- `hooks/pretooluse-budget.sh`
- `hooks/stop-compact.sh`
- `agents/buddhi.md`, `agents/manas.md`, `agents/sakshi-keeper.md` (spot-checked structure only)
- `commands/*.md` (spot-checked)
- `skills/context-discipline/SKILL.md`, `skills/sublate-on-conflict/SKILL.md`, `skills/witness-prefix/SKILL.md` (spot-checked)

**Harness library (`src/`, standing in for `harness/`)**

- `src/avacchedaka/store.py`
- `src/avacchedaka/query.py`
- `src/avacchedaka/element.py`
- `src/agents/orchestrator.py`
- `src/agents/buddhi.py` (imports / role only)
- `src/agents/manas.py` (imports / role only)
- `src/compaction/detector.py`
- `src/compaction/surprise.py` (partial)
- `src/forgetting/schedules.py`
- `src/benchmarks/stats.py`
- `src/utils/tokenizer.py`
- `src/evaluation/khyativada_fewshot.py` (partial)
- `tools/dev/scheduler/cli_budget.py` (partial — subprocess usage)

**Experiments (`experiments/v2/`)**

- `experiments/v2/p6a/run_plugin_inloop.py` (partial)
- `experiments/v2/p6a/plugin_client.py`
- `experiments/v2/p6b/run_case_study.py` (partial)
- `experiments/v2/p6c/research_evidence.py`
- `experiments/v2/p6c/run_swebench_ab.py` (partial)
- `experiments/v2/p7/aggregate.py` (substantial portions)

**Tests (`tests/test_v2/` and related)**

- `tests/test_v2/test_p7_aggregate.py` (partial)
- `tests/test_forgetting/test_schedules.py` (partial)
- Pytest collection counts for `tests/` and `tests/test_v2`

**Project metadata**

- `pyproject.toml` (partial)

---

## Severity counts (this pass)

- **Critical:** 3  
- **Major:** 7  
- **Minor:** 7  

_No additional Critical/Major issues were identified in: shell hooks command injection (static commands + fixed strings), `CLIBudgetScheduler` subprocess invocation pattern (argument list, not `shell=True` for the runner itself), or Khyātivāda few-shot parser (not exhaustively fuzzed in this pass)._
