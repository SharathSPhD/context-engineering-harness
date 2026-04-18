# Paper Feasibility Review

Scope: `paper/sections/*.md`, `paper/appendices/*.md`, and the shipped plugin under `plugin/pratyaksha-context-eng-harness/`, cross-checked against `experiments/v2/*` and `mcp/server.py`. This review answers whether a real third-party deployer or PhD student can rely on the paper’s install, reproducibility, and scaling claims.

## Critical (will not work as described)

- **Claim:** Appendix C gives exact commands to re-run L1/L3 and aggregation “as we ran.”  
  **Evidence it won’t work:** `Appendix C` instructs `git clone …/pratyaksha-context-eng-harness.git` then `uv run pytest` and modules `experiments.v2.p6a.run_h1_h2`, `run_h3_h4_h5`, `run_h6_h7`, and `experiments.v2.p7.aggregate --in … --out …`. The harness-only repository does not contain the `experiments/` tree or `src/benchmarks/` package; the runnable code lives in this monorepo (`context-engineering-synthesis`). There is no `run_h1_h2` module — H1/H2 are `python -m experiments.v2.p6a.run`, and H3–H7 are `python -m experiments.v2.p6a.run_plugin_inloop`. P7 accepts `--out-dir` only, not `--in` / `--out`. P6-B output is `_summary.json`, not `summary.json`. P6-C uses `--n-examples` and `--research-budget-tokens`, not `--n-instances` / `--research-block-budget`; artefacts are `swebench_ab.json` and `_summary.json` in the output dir, not a `runs/` tree as written in Appendix C.  
  **Fix:** Pin the **monorepo** URL/commit, replace Appendix C with the actual module names and flags, and document required extras (`[dev]`, `[benchmarks]`) from `pyproject.toml`.

- **Claim:** Section 6.4 / Appendix B — `pretooluse-budget` refuses harness MCP tools when the budget gauge exceeds a hard threshold (~95%).  
  **Evidence it won’t work:** `hooks/pretooluse-budget.sh` explicitly always returns `"permissionDecision": "allow"` (comments: “never blocks”). Section 6.6 text and the hooks JSON in the paper describe enforcement that is not implemented.  
  **Fix:** Either implement deny/ask above threshold or rewrite the paper to match the advisory-only hook.

- **Claim:** Appendix B — every tool appends to `${CLAUDE_PLUGIN_ROOT}/witness_log.jsonl`.  
  **Evidence it won’t work:** `mcp/server.py` appends mutating operations to `~/.cache/pratyaksha/audit.jsonl` (and related JSONL paths under that cache dir), not `witness_log.jsonl` under the plugin root.  
  **Fix:** Align documentation with `_audit()` or change the server to the documented path.

- **Claim:** Section 7.6.2 — P6-C builds a “fixed-budget research block (default **8 K tokens**).”  
  **Evidence it won’t work:** The reference implementation `experiments/v2/p6c/run_swebench_ab.py` uses `--research-budget-tokens` default **512**, not 8192. A student following the methodology section literally will not reproduce the paper’s stated experimental setup unless flags match whatever was actually used to generate tables.  
  **Fix:** Make Section 7, Appendix C, and the argparse defaults agree; check in the exact CLI used for the published run.

## Substantive (will work but with serious caveats not stated)

- **Claim:** “Hot-swappable across Cursor, Claude Code, and Claude Desktop” with one `marketplace.json` (Section 6.9).  
  **Evidence / caveat:** The MCP launcher in `.mcp.json` assumes `uv`, `sh`, `CLAUDE_PLUGIN_ROOT` (with a fallback to `~/.claude/plugins/cache/...`), and adds `~/.local/bin` to `PATH`. Cursor and Desktop differ in how they resolve plugin roots, env injection, and hook surfaces; **Claude Code–style lifecycle hooks** (`SessionStart` / `PreToolUse` / `Stop`) are not guaranteed to exist or behave identically on every host the paper lists. A third party should expect “MCP tools work everywhere hooks are supported” rather than “identical lifecycle everywhere.”  
  **Fix:** Split claims into (a) MCP portability and (b) hook portability; list host-specific setup for non–Claude Code clients.

- **Claim:** Section 5.5 / 6.2 — Khyātivāda classification is a “Claude prompt with structured JSON” / LLM classifier.  
  **Evidence / caveat:** Shipped `classify_khyativada` in `mcp/server.py` is documented in-code as a **heuristic backend** (`_classify_khyativada` over `claim`/`ground_truth`/`context`); it does not call the Anthropic API. Paper results involving a few-shot Claude classifier would not reproduce from the plugin alone without a separate code path.  
  **Fix:** State which evaluation stack used an LLM classifier vs which shipped tool is heuristic-only.

- **Claim:** No fine-tuning; Buddhi/Manas/Sākṣī are prompt-only (Section 5.4).  
  **Evidence / caveat:** True for **weights**, but the harness still **depends on instruction-following**: Manas must emit structured JSON, Buddhi must call tools in order, and the host must orchestrate sub-agents. A base model with weak tool/JSON discipline (e.g. Llama-3-8B-base) may fail silently or ignore `set_sakshi` / retrieval gates — the paper acknowledges cross-family measurement gaps (Section 11.1.3) but not this **minimum capability floor**.  
  **Fix:** Add an explicit “minimum model capability” assumption (structured output + reliable tool use).

- **Claim:** Section 7 / 11 — “real” benchmark families (RULER, HELMET, NoCha, SWE-bench Verified).  
  **Evidence / caveat:** Discussion 11.1.1 states CI and most L1 runs use **synthetic-fallback** adapters unless HF credentials and network are available; P6-C defaults to **synthetic** instances unless `--load-real`. Version pins for upstream benchmark **snapshots** are not cited in the sections reviewed.  
  **Fix:** Cite dataset revision / HF config / commit for any “real” run; separate “synthetic parity” numbers from upstream leaderboard numbers.

- **Claim:** Section 6.7 — grep audit shows **zero** references to dev-time tools.  
  **Evidence / caveat:** There are **no Python imports** of `attractor-flow` / `ralph-loop` (verified). However, the strings `attractor-flow` and `ralph-loop` **do appear** in shipped markdown (`agents/sakshi-keeper.md`, `skills/witness-prefix/SKILL.md`, `README.md`, `marketplace.json` feature text). If the claim is literal “grep the tree,” it fails; if the claim is “no runtime dependency / import,” it holds.  
  **Fix:** Use precise language: “no imports / no runtime dependency,” not “zero references.”

- **Claim:** Section 7.8 / Appendix C — full test suite green at **499** passing.  
  **Evidence / caveat:** `uv run pytest tests/ -q` on this worktree: **498 passed, 1 failed, 2 skipped** (`test_with_harness_run_is_deterministic[requests_retry_adapter]` in `tests/test_v2/test_p6b_case_study.py`).  
  **Fix:** Update the paper’s counts or fix the failing test before asserting the number.

## Minor (will work, but the paper should add a caveat)

- **P7 aggregation inputs:** `aggregate.py` expects `experiments/results/p6a/_summary.json`, `_summary_plugin.json`, `p6c/swebench_ab.json`, and `experiments/h6_khyativada_classifier/results/agreement_report.json`. Appendix C does not mention generating the H6 agreement artefact or the exact P6-A summary filenames — a replicator can run P6-A/P6-B/P6-C and still get empty P7 tables until those files exist.

- **Internal reproducibility pointer error:** `paper/sections/07_methodology.md` §7.8 says the “full reproducibility manifest” lives in **Appendix F**, but **Appendix F** is negative/null results; **Appendix C** is titled “Reproducibility Manifest.” This will send a replicator to the wrong appendix.

- **Section 7 vs. code on P6-C token budget:** §7.6.2 states a “fixed-budget research block (default **8 K tokens**)” for P6-C, while `run_swebench_ab.py` defaults `--research-budget-tokens` to **512**. The paper text and the shipped runner disagree unless every reported number used an explicit 8192 override.

- **Appendix F references `attractor-flow-state/journals/`** — fine for transparency, but reinforces that **monorepo** assets exist outside the shipped plugin; keep plugin vs research repo boundaries explicit.

- **Scaling (100 GB logs, 1 M store items, 10-day sessions):** The MCP `ContextStore` is an **in-process dict** (`_State.elements`); `audit.jsonl` grows by append with no rotation described in code reviewed. The paper’s Discussion names validity threats but does **not** bound disk/RAM for long-lived agents or spell out log rotation / sharding. Add deployment limits or operational guidance.

- **`session-start.sh` vs `agents/sakshi-keeper.md`:** The shell hook only emits `additionalContext` JSON; it does **not** invoke the Sakshi agent. The agent markdown says the hook “calls you” — minor inconsistency for implementers.

- **Optional stack:** `pyproject.toml` lists `chromadb`, `mlflow`, etc., for the research repo; the plugin README/marketplace correctly positions heavy stacks as non-required for MCP — still worth one sentence so a student does not confuse **research** deps with **plugin** deps.

## Audit results

Commands and reads (representative):

```bash
# Exclusion audit (user-requested pattern) under plugin/
grep -r "attractor\|ralph_loop\|ralph-loop" plugin/
# (from repo root; equivalent path used:)
grep -r "attractor\|ralph_loop\|ralph-loop" "/Users/sharath/Library/CloudStorage/OneDrive-Personal/wsl_projects/context/.worktrees/v2/plugin/"
# Result: matches in marketplace.json, README.md, skills/witness-prefix/SKILL.md, agents/sakshi-keeper.md (text mentions); no .py/.sh imports of those packages.

grep -r "import.*attractor\|import.*ralph\|from attractor\|from ralph" \
  "/Users/sharath/Library/CloudStorage/OneDrive-Personal/wsl_projects/context/.worktrees/v2/plugin/pratyaksha-context-eng-harness" \
  --include="*.py" --include="*.sh"
# Result: no matches (no code dependency).

cd "/Users/sharath/Library/CloudStorage/OneDrive-Personal/wsl_projects/context/.worktrees/v2" && uv run pytest tests/ -q
# Result (tail): 1 failed, 498 passed, 2 skipped; failure: tests/test_v2/test_p6b_case_study.py::test_with_harness_run_is_deterministic[requests_retry_adapter]
```

Files read for install path and behaviour:

- `plugin/pratyaksha-context-eng-harness/.claude-plugin/plugin.json`
- `plugin/pratyaksha-context-eng-harness/marketplace.json`
- `plugin/pratyaksha-context-eng-harness/.mcp.json`
- `plugin/pratyaksha-context-eng-harness/hooks/hooks.json`, `session-start.sh`, `pretooluse-budget.sh`, `stop-compact.sh`
- `plugin/pratyaksha-context-eng-harness/mcp/server.py` (header, `_State`, `_audit`, `classify_khyativada`)
- `experiments/v2/p6a/run.py`, `specs.py`, `run_plugin_inloop.py` (CLI)
- `experiments/v2/p6c/run_swebench_ab.py` (defaults and outputs)
- `experiments/v2/p7/aggregate.py` (inputs/CLI)
- `paper/sections/01_introduction.md`, `03_origin.md`, `05_architecture.md`, `06_plugin.md`, `07_methodology.md`, `11_discussion.md`
- `paper/appendices/B_mcp_tools.md` (partial), `C_reproducibility.md`, `F_negative_results.md`
