# Pratyaksha v2 — Adversarial Review

Scope: `plugin/pratyaksha-context-eng-harness`, `experiments/v2/*`, `experiments/results/p7/*`, `paper/appendices/*`, `src/benchmarks/stats.py`. Method: read canonical sources; spot-check determinism with two `aggregate` runs (index equal after dropping `ts`).

## Issues that would block a real reviewer

- **[high — paper `appendices/B_mcp_tools.md` vs `plugin/.../mcp/server.py`]** Appendix B asserts audits go to `${CLAUDE_PLUGIN_ROOT}/witness_log.jsonl` and that PreToolUse “refuses execution” when the budget gauge is hot. The shipped server writes to `~/.cache/pratyaksha/audit.jsonl` (see `_audit` in `server.py`), and `pretooluse-budget.sh` **always** emits `"permissionDecision": "allow"`. A reviewer treating the appendix as normative will conclude the implementation is wrong—or the paper is materially false.

- **[high — omnibus statistics / `experiments/v2/p7/aggregate.py` + `results/p7/tables/T7_omnibus_stouffer.md`]** Stouffer combination assumes (at least approximately) **independent** p-values. The omnibus includes **four** long-context rows that share the same mock runner, models, seeds, and `n_examples_override` (`H1`×2 + `H2`×2 from one artifact), plus five plugin-in-loop rows from a second artifact—again same machinery. Those p-values are **not** independent draws; treating them as k=10 separate “studies” overstates evidence. The table also shows **repeated identical p-values** (e.g. `0.0005`, `0.001953`), consistent with permutation/grid floors rather than continuous evidence accumulation.

- **[high — external validity of headline SWE numbers / `experiments/v2/p6c/run_swebench_ab.py` + `results/p6c/swebench_ab.json`]** Published headline JSON has `load_real_swebench_verified: false` and uses `PatchSimulator`, not repository-resident models. The simulator **anchors on the first `` `*.py` `` path** in the concatenated prompt; the harness arm **reorders** research text so high-precision (correct-file) snippets dominate retrieval output, while the baseline keeps shuffled discovery order (`research_evidence.generate_research_trail` ends with `rng.shuffle(out)`). The code comments **admit** the asymmetry is the point—but that makes the **720-pair** result a measure of “this stub + this trail generator,” not SWE-bench Verified in the wild. Marketing that blurs this crosses into misleading empirical claims.

- **[high — `outcome_per_seed_mean` in same artifact]** `n_pairs: 6` with `p_value: 0.03125` is **exactly** `1/32`, the smallest attainable p from a **full enumeration** paired sign-flip test on 6 differences (`2^5` equally likely mean statistics under the usual symmetry argument). It should be reported as “at the discrete permutation floor,” not as a continuous “~0.03” evidential strength. The paired Cohen’s d ≈ **13.5** on six aggregate means is a red flag for **degenerate variance** (effect sizes that explode when the per-seed means barely move relative to machine precision / rounding).

- **[medium-high — marketplace / install story / `marketplace.json` + `.mcp.json`]** `marketplace.json` claims the feature set has “no dependency on … vllm” while simultaneously advertising “Token-level surprise (vLLM/HF/heuristic).” The shipped `boundary_compact` tool implements **only** the heuristic path—there is **no** vLLM call path in `server.py`, so “when vLLM returns nonsense” is mostly a non-scenario for this artifact. Drop-in install assumes `uv` on `PATH` (`$HOME/.local/bin` appended in `.mcp.json`); environments without `uv` or with a different plugin root than `CLAUDE_PLUGIN_ROOT` break silently or point at the wrong directory. **Schema validation** against Cursor/Claude Code marketplace JSON Schema was **not** performed here (no schema pinned in-repo); `rating: 5.0` / `downloads: 0` look hand-filled and may fail strict validators.

## Issues a sympathetic reviewer might let pass

- **[medium — P6-B “LLM-free” case study / `experiments/v2/p6b/run_case_study.py`]** The without-harness arm uses a deliberately crippled policy (`first_seen_wins` on evidence order). That is **not** a competitive LLM baseline; it is a **toy** showing bookkeeping helps when the baseline is forbidden from using precision metadata. Fair if framed as a **mechanism demonstration**; unfair if sold as “accuracy on real agents.”

- **[medium — Khyātivāda `confidence` / `server.py::_classify_khyativada`]** Classical Khyātivāda classes are not accompanied by Bayesian “confidence” in the sources. The plugin emits **fixed** heuristic confidences (e.g. `0.82`, `0.55`) from string patterns—fine as engineering, but **philosophically anachronistic** if presented as textual exegesis.

- **[medium — requested `harness/aggregation/sublation.py`]** There is **no** `harness/aggregation/sublation.py` in this worktree; sublation logic for the MCP surface lives inline in `server.py` (precision zeroing + `sublated_by`). Any claim that “seven sublation rules” are implemented as a separate faithful `bādha` engine should be checked against **actual** code paths referenced in the paper.

- **[low-medium — audit / witness growth]** There is **no rotation** for `audit.jsonl` / `cost_ledger.jsonl`; long sessions can grow without bound until the OS denies writes. (The paper’s `witness_log.jsonl` narrative does not match the implementation path—see blocker above.)

- **[low — `aggregate.py` “determinism”]** `_index.json` is stable run-to-run except for `ts` (verified). `_summary.md` always changes its **timestamp** line. Matplotlib PNG bytes can still drift across **versions/platforms** even when data are identical.

- **[low — Appendix F negative results]** F.1–F.4 are “we replaced component X with the current winning design,” which is **historical** but not statistically independent from the positive evaluation (the replacement *is* what gets measured). F.5 (vLLM vs heuristic null) is the clearest **genuine** null; F.6 is a good engineering cautionary tale.

## Things that hold up under attack

- **Minimal MCP runtime deps (PEP 723 in `server.py`).** The documented `mcp` + `pydantic` + `tiktoken` story matches the inline script metadata; missing `anthropic`, `vllm`, and `huggingface-hub` do **not** stop the server for the implemented tools (there is no HF/vLLM branch in `boundary_compact`).

- **Bootstrap / permutation seeds for P6-C.** `bootstrap_ci(..., seed=0)` and `paired_permutation_test(..., seed=0)` are wired explicitly in `run_swebench_ab.py`, so **those** intervals are repeatable given fixed inputs.

- **Honest labeling of synthetic default.** `swebench_ab.json` and the runner docstring state synthetic-by-default and name the PatchSimulator’s first-path anchoring behaviour; the vulnerability is **downstream marketing**, not the JSON artifact hiding the mode flag.

- **Hooks and `set -u` / `pipefail`.** The shell hooks are short, drain stdin to avoid SIGPIPE, and avoid `-e` on budget hooks so arithmetic/jq failures degrade to “allow/no-op” rather than hard-failing the agent loop—reasonable for resilience.

## Recommendation

**Patch and ship** the plugin as a **self-contained MCP + hooks** bundle **after** reconciling `paper/appendices/B_mcp_tools.md` (and any public README claims) with `server.py` and `hooks/*.sh`, and **reframing** the quantitative headline: either split “engineering demonstration” metrics from “benchmark replication,” or add real-SWE / real-model runs as the only line in the abstract. **Do not** ship the Stouffer-Z omnibus as a literal meta-analysis p-value without a dependence adjustment (or counting only **preregistered** independent families). For the per-(model, seed) SWE statistic, disclose **discrete test floor** (`p = 1/32` for six paired means) and treat Cohen’s d on six aggregates as non-interpretable.
