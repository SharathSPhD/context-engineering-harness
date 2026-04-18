# Pratyaksha v2 — Kieran Python Review

## Verdict

**Pass with required changes.** The harness code is readable and mostly modern (3.11+ unions, dataclasses, Pydantic v2 on the MCP surface), but several spots smuggle `Any`, swallow exceptions without traceability, or encode statistical/reporting semantics incorrectly. None of these are mysterious; they are fixable without redesign.

## Scope note

- **`harness/**/*.py`:** No `harness/` directory exists at the v2 worktree root (layout uses `src/` for library code). Nothing reviewed under that glob.
- **`experiments/v2/p6b/__init__.py`**, **`experiments/v2/p6c/__init__.py`**, **`tests/test_v2/__init__.py`:** Empty modules — no findings.

## Findings (by file)

### `plugin/pratyaksha-context-eng-harness/mcp/server.py`

- L43, L65, L134, L252–263, L277–300, L320+, L457+: Heavy `dict[str, Any]` on tool boundaries and `_serialize` / classifier returns. **Suggest:** `TypedDict` or small Pydantic `BaseModel` for stable JSON shapes (`ToolOk`, `ToolErr`, `SerializedElement`) so callers and tests stop guessing keys.
- L107–109: `except Exception:` in `_count_tokens` — silent fallback with no log, no exception chaining, hides `ImportError`, broken installs, and real tiktoken failures. **Suggest:** catch `(ImportError, OSError, ValueError)` or `Exception as e:` + `logger.debug("tiktoken unavailable, char fallback", exc_info=True)` then fallback.
- L134: `salience: dict[str, Any]` on `ContextElement` — unbounded bag; if unused in tools, remove; if used, type the known keys.
- L308: `_lifespan(_app)` — parameter untyped. **Suggest:** `object` or the concrete FastMCP app type if available from `mcp`.
- L646–649: `budget_status` skips malformed JSONL lines with bare `continue` — acceptable for corruption, but **Suggest:** count/log skipped lines at `DEBUG` so silent data loss is visible.
- L271–274: `_KHYATIVADA_CLASSES` tuple is unused except indirectly via taxonomy copy — fine, but `_classify_khyativada` never validates output `"class"` is a member (heuristic could drift). **Suggest:** assert membership before return in debug builds, or map through an enum.
- Public API: module exposes many undecorated names; no `__all__`. **Suggest:** `__all__` listing tool entrypoints if this file is treated as a library surface.

### `plugin/pratyaksha-context-eng-harness/mcp/smoke_test.py`

- L62–73: `_content_to_dict(result)` — `result` untyped; on `JSONDecodeError` returns `{"_raw": text}` which can let a later `check(d["ok"] is True)` fail opaquely or mask server returning non-JSON. **Suggest:** type `result` as `mcp.types.CallToolResult`, fail hard on decode error unless `_raw` mode is explicitly for debugging.
- L306–308: Broad `except Exception` in `main` — acceptable for a smoke driver, but **Suggest:** narrow to transport/MCP errors where possible so real bugs surface.

### `experiments/v2/p6a/run.py`

- L35: `from typing import Any` — payload assembly is all `dict[str, Any]`. **Suggest:** `TypedDict` for written JSON payloads at the boundary.
- L76–77, L91–96, L162–179: `_select_specs`, `_run_one_bundle`, `_summarise_run`, `_asdict_spec` use untyped `bundle`, `run`, `spec`. **Suggest:** annotate `bundle: P6ASpecBundle`, `run: BenchmarkRun` (or protocol), `spec: HypothesisSpec`.
- L126–149: After `runner.run_hypothesis(spec_used)`, the code **re-runs** `runner.run_condition` for every `(model, seed)` for both arms. This duplicates work and risks divergence if `MultiSeedRunner` ever becomes stateful or non-deterministic across calls. **Suggest:** reuse the runs backing `outcome` or extract a single code path that produces both summary and per-condition tables.
- L176–179: `_asdict_spec` uses `hasattr(spec.direction, "value")` — structural typing smell. **Suggest:** branch on `isinstance(spec.direction, Enum)` or always store enum in `HypothesisSpec`.

### `experiments/v2/p6a/callers.py`

- L56–60: `_ALL_CODE_RE` is defined but never referenced — dead code. **Suggest:** delete or use in `_extract_pairs` for the generic pattern.
- L195: `def __init__(self, scheduler) -> None:  # type: CLIBudgetScheduler` — comment is not a type checker hint. **Suggest:** `from typing import TYPE_CHECKING` + `if TYPE_CHECKING: from tools.dev.scheduler import CLIBudgetScheduler` and `scheduler: CLIBudgetScheduler`.
- L229: `-> "callable"` — wrong hint (`callable` builtin, not `typing.Callable`). **Suggest:** `Callable[..., ModelOutput]` or a `Protocol` matching `ModelCaller`.

### `experiments/v2/p6a/specs.py`

- L31: `adapter_kwargs: dict` — bare `dict` (implicitly `dict[Any, Any]`). **Suggest:** `dict[str, Any]` or a kwargs TypedDict per adapter.

### `experiments/v2/p6a/scenarios.py`

- L17: `from typing import Sequence` — unused import. **Suggest:** remove.

### `experiments/v2/p6a/plugin_client.py`

- L44–60: `_load_plugin_server()` lacks `-> ModuleType` return type; uses `assert spec and spec.loader` which becomes `AssertionError` instead of a domain error. **Suggest:** explicit `if spec is None or spec.loader is None: raise RuntimeError(...)`; return type `ModuleType`.
- L78–83: `reset` mutates `STATE` in place — documented and intentional; fine. **Suggest:** consider `contextvars` for true per-task isolation if concurrency is ever introduced.

### `experiments/v2/p6a/run_plugin_inloop.py`

- L883–890: `_per_seed_loop(*, models, seeds, run_pair)` — parameters untyped. **Suggest:** `models: tuple[str, ...]`, `seeds: tuple[int, ...]`, and a `Protocol` or `Callable` type for `run_pair`.
- L556: Default `predicted = out.get("class", "atmakhyati")` — masks missing/invalid tool output as a real label. **Suggest:** treat missing `"class"` as an error in scoring or use `None` and count as wrong explicitly.
- L964: `"n_seeds_used": len(spec.seeds) * len(spec.models)` — name implies count of seeds; value is **model×seed product**. **Suggest:** rename to `n_model_seed_cells` or store actual row count from `per_seed_runs`.

### `experiments/v2/p6b/run_case_study.py`

- L35: `import re` — unused. **Suggest:** remove.

### `experiments/v2/p6b/case_data.py`

- no findings — frozen dataclasses, clear naming, good `__all__`.

### `experiments/v2/p6c/run_swebench_ab.py`

- L394–397: `if not isinstance(caller, type(MockHarnessCaller)): pass` — always true for `PatchSimulator`, dead branch. **Suggest:** delete block; if the intent was protocol checking, use `typing.Protocol` + `assert isinstance(caller, ModelCaller)`.
- L226–227: `import re` inside `PatchSimulator.__call__` — inner import on hot path for no benefit. **Suggest:** module-level `import re`.
- L175: `n_kept = truncated.count("- ")` — fragile proxy for snippet count (content could contain `"-"`). **Suggest:** count snippets before truncation or split on a delimiter you control.
- L319: `(example.ground_truth or {}).get(...)` — assumes shape; fine if adapter guarantees it; **Suggest:** narrow `ground_truth` type on `BenchmarkExample` if possible.
- L455–457: Paired permutation uses **two independent lists** `with_arr` and `without_arr` built by concatenating all instances — ordering must stay lockstep with generation order; documented in tests. **Suggest:** store explicit `(with, without)` pairs in the artifact to make the invariant structural, not positional.

### `experiments/v2/p6c/research_evidence.py`

- L49–60: `_STALE_TEMPLATES` / `_FRESH_TEMPLATES` as `tuple[dict, ...]` — untyped dicts. **Suggest:** `TypedDict` with `src`, `preface`, `wrong_path_offset` keys.

### `experiments/v2/p7/aggregate.py`

- L320–337, L340+, L612+, many `def figure_F03(art):` — `art` untyped; inconsistent with `figure_F01(art: dict[str, Any])`. **Suggest:** unify `ArtifactBundle` `TypedDict` or `dict[str, Any]` on all figure/table entrypoints.
- L224: `_save(fig, path)` — `fig` untyped. **Suggest:** `matplotlib.figure.Figure`.
- L386: Figure title says **"6-class"** while the taxonomy is **seven** labels — documentation bug aligned with `classify_khyativada` docstring elsewhere.
- L187, L195: `HypothesisRow.n_pairs=o["n_examples_used"]` for P6-A rows — **misleading**: `n_examples_used` is not the number of paired statistical pairs in the long-ctx runner JSON. **Suggest:** map the correct count (e.g. `len(models)*len(seeds)*n_examples` or whatever the test actually uses) or rename column to `n_examples_used` in tables.
- L791–795 vs L799–804: `aggregate()` wraps **figures** in `try/except Exception` but **tables** are unguarded — inconsistent resilience; one bad table crashes the whole P7 run. **Suggest:** mirror figure error handling for tables or fail fast everywhere.

### `experiments/v2/__init__.py`

- no findings — one-line docstring.

### `experiments/v2/p6a/__init__.py`

- no findings — module docstring only.

### `experiments/v2/p7/__init__.py`

- no findings.

### `tests/test_v2/test_p6a_run.py`

- L11: `import sys` unused. **Suggest:** remove.

### `tests/test_v2/test_p6a_callers.py`

- no findings — behaviour-focused, good use of deterministic seeds and explicit assertions.

### `tests/test_v2/test_p6a_scenarios.py`

- no findings — strong coverage of determinism and structural invariants.

### `tests/test_v2/test_p6a_plugin_client.py`

- no findings — asserts real response shapes; could additionally assert `"ok" is False` paths but not required.

### `tests/test_v2/test_p6a_run_plugin_inloop.py`

- L32: `_run(...) -> dict` — untyped return; **Suggest:** `dict[str, Any]` or a typed payload.
- Tests assert `target_met` and positive deltas — they encode the **experimental design** (harness wins by construction). **Suggest:** label these as smoke/regression guards in docstrings so future readers do not confuse them with unbiased statistical claims.

### `tests/test_v2/test_p6b_case_study.py`

- no findings — excellent: parametrised fixtures, tests private helpers where they encode invariants, end-to-end `run_all` I/O check.

### `tests/test_v2/test_p6c_run_swebench_ab.py`

- no findings — exercises harness/off paths, simulator anchoring, and scaling assumptions (`n_pairs`).

### `tests/test_v2/test_p6c_research_evidence.py`

- no findings — tight property tests on the generator.

### `tests/test_v2/test_p7_aggregate.py`

- L20–24: `_norm_inv_cdf` round-trip uses `1 - _norm_sf(z)` — fine as a smoke; not a formal inverse-CDF test. **Suggest:** rename test to “approximate round_trip” if tightening numerics later.
- L60–61, L82–83: Tests assume **real artifact tree** in-repo — brittle if results are absent in a clean checkout; acceptable if CI always generates artifacts first. **Suggest:** document ordering dependency in test module docstring (generate P6 before P7 test) or use committed fixture copies.

## Patterns to repeat

- **Frozen dataclasses** for scenario fixtures (`H3Case`, `H7Scenario`, `CaseStudy`, `ResearchSnippet`) — clear immutability story.
- **Explicit experiment runners** with argparse, structured JSON artifacts, and separate **summary** files — good reproducibility hygiene.
- **In-process plugin client** (`PratyakshaPluginClient`) that loads the real MCP module once — high fidelity for tests without IPC.
- **Parametrised tests** over `ALL_CASE_STUDIES` with stable ids — strong signal when a case regresses.
- **Pydantic v2 `BaseModel` + `Field`** on MCP tool inputs in `server.py` — idiomatic validation at the boundary.

## Patterns to retire

- **`dict[str, Any]` as the universal JSON sponge** on every internal helper — use `TypedDict` / models at persistence boundaries.
- **`except Exception: pass` or silent `continue`** without structured logging — especially in `_count_tokens` and JSONL readers.
- **Comment-as-type** (`# type: Foo`) and `-> "callable"` — use real annotations and `TYPE_CHECKING`.
- **Dead code** (`pass` blocks, unused regex compilers, unused imports) — indicates missing lint (Ruff `F401`, `F841`).
- **Misleading metric names** (`n_seeds_used`, `n_pairs` populated from `n_examples_used`) — reviewers and meta-analysis will misread effect sizes.

## Test-suite assessment

**Strengths:** `test_v2` is unusually strong for research code — deterministic generators are pinned, plugin I/O shapes are asserted, P6-B covers both arms and transcript presence, P6-C checks pairing counts and anchoring behaviour, P7 tests Stouffer machinery and full emit paths.

**Weak spots:** Several tests **encode the desired experimental outcome** (`target_met`, positive δ, treatment > baseline) as hard assertions. That is legitimate as **regression guards** for a harness built to demonstrate lift, but it is not a substitute for unbiased evaluation — document that distinction. `test_p7_aggregate` depends on pre-generated result files in the tree; without those artifacts, the “emit every figure” test fails for reasons unrelated to matplotlib code. The MCP `smoke_test.py` can mask non-JSON tool responses via `_raw`.

---

*Reviewer persona: Kieran-strict Python clarity & maintainability. File paths relative to worktree root.*
