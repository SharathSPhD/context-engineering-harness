# Release notes — Pratyakṣa Context-Engineering Harness v2.1 (live-HF rerun)

v2.1 is a **data-fidelity patch** on top of v2.0. The plugin binary, the
paper's central architecture, and the philosophical vocabulary are
unchanged. The only new scientific content is a preregistered,
`strict_hf`-gated live rerun of four headline benchmark bundles against
real Hugging Face data — which puts a partial floor under the
synthetic-fallback-adapter threat named in v2.0's §11 discussion.

## What's new

### 1. Live Hugging Face rerun (`core4`)

Four bundles were re-run end-to-end against live Hugging Face datasets
at pinned commit SHAs between 2026-04-20 and 2026-04-21:

| bundle                | dataset @ commit SHA                                  |  treatment | baseline |       Δ | paired *p* |    *d_z* |      *n*   | status                           |
| --------------------- | ----------------------------------------------------- | ---------- | -------- | ------- | ---------- | -------- | ---------- | -------------------------------- |
| `H1_ruler_8192_live`  | `simonjegou/ruler@24adcea`                            |     1.0000 |   0.7667 | +0.2333 |     0.0005 |    0.547 |         60 | complete, target met             |
| `H1_ruler_16384_live` | `simonjegou/ruler@24adcea`                            |     1.0000 |   0.9333 | +0.0667 |     0.1174 |    0.265 |         60 | complete, underpowered           |
| `H_TQA_live_v2`       | `truthfulqa/truthful_qa@741b827`                      |     0.0500 |   0.0833 | −0.0333 |     0.7386 |   −0.091 |         60 | complete, null                   |
| `H_SWEB_live_n15`     | `princeton-nlp/SWE-bench_Verified@c104f84` (haiku)    |     0.1259 |   0.0167 | +0.1093 |     0.0322 |    0.488 |         30 | partial (CLI-blocked, see App G) |

Conservative reading: the system clears **both** preregistered gates
(*d*z ≥ 0.5 **and** *p* ≤ 0.05) on **one** independently-sourced live
surface (RULER 8K, *d*z = 0.547) and clears the *p* ≤ 0.05 gate on a
**second** (SWE-bench-haiku, *d*z = 0.488 — just below the *d* gate),
in addition to the full v2.0 synthetic-fallback battery at *N* ≈
180–700 per bundle. RULER 16K points in the predicted direction but is
underpowered at *N* = 60. TruthfulQA returns a null at this revision
and *N*. The SWE-bench paired *p* = 0.032 above is computed under the
pre-registered rule that CLI-aborted attempts score 0 on both sides;
restricting to attempts where neither side aborted leaves only 1 clean
paired observation, so the SWE signal is essentially imputation-driven
at this *N*.

### 2. Strict live-HF guard on every adapter

Every live-capable adapter
(`RulerNIAHAdapter`, `TruthfulQAAdapter`, `HaluEvalQAAdapter`,
`SWEBenchVerifiedAdapter`) accepts a new `strict_hf: bool` flag. When
the live pre-registration spec sets `strict_hf=True` and
`load_real=True`, any Hugging Face loading failure becomes a hard
`ProvenanceIntegrityError` rather than a silent fall-through to
synthetic data.

The runner `experiments/v2/p6a/run_live_hf.py` also runs a
"probe-then-verify" step: it loads one example per bundle before any
API call goes out, reads back `provenance.source`, and aborts with
exit code `3` if the source is not the literal string `"huggingface"`.

### 3. Pre-registration enforcement on the runner CLI

Under `--scope core4` and `--scope full_battery`, the runner refuses
to start if any of `--ruler-n` / `--hallu-n` / `--sweb-n`, `--seeds`, `--models`, or `--tiers`
differ from the `LIVE_DEFAULT_*` constants baked into
`experiments/v2/p6a/specs_live.py`. The `--allow-preregistration-override`
flag exists as an explicit escape hatch, but exits with code `2` and
writes a `PreregistrationViolation` line to the log when the default
pre-registration is bypassed.

### 4. Scheduler: billed-input-token budgeting

`CLIBudgetScheduler` now budgets on **billed** input tokens
(`cache_hit = 0`) only, not on disk-cache hits. Repeated replays of
previously-cached prompts no longer exhaust the Anthropic rolling
window. `WindowSummary.billed_input_tokens` and
`scheduler.status()["billed_input_tokens"]` are new public fields;
`is_window_exhausted(max_input_tokens=...)` compares against the
billed field.

### 5. Consolidator stamps `hf_revision`

`tools/dev/live_hf_consolidate.py` resolves every live bundle's
`hf_dataset_id` to its current Hugging Face commit SHA via the Hub API
and stamps it as `provenance.hf_revision` on each entry in
`_summary_live.json`. This is idempotent, memoised per process, and
gracefully degrades to `null` if `huggingface_hub` is missing or the
network call fails.

### 6. Ralph live-loop orchestrator

`tools/dev/ralph_live_loop.py` keeps the live runner + consolidator
running across Anthropic 5-hour rolling-window exhaustions. It reads
the scheduler's `next_window_at` field to align sleep intervals with
real quota resets, threads the `--out-dir` into the consolidator,
treats any status other than `complete` as "still needs a rerun", and
imposes a 6-hour per-inner-subprocess timeout so runaway calls cannot
block the loop indefinitely.

## What hasn't changed

- The plugin binary
  (`release/pratyaksha-context-eng-harness-v2.0.0.zip`)
- The paper's architecture, theory, and L1–L3 synthetic-fallback
  results
- The philosophical vocabulary mapping (avacchedaka, bādha,
  manas/buddhi, sākṣī, khyātivāda)
- The Zenodo preprint DOI (cite the 19653013 record)

## Known limitations

- **SWE-bench CLI abort.** ~77 % of `claude-haiku-4-5` SWE-bench
  attempts and 100 % of `claude-sonnet-4-6` attempts aborted before
  emitting a scoreable answer, because the `claude` CLI 2.1
  SessionStart hook does not complete inside the scheduler's 300 s
  timeout on very large prompts. The haiku-only `n = 30` paired slice
  is reported; see Appendix G.
- **Underpowered live rerun.** `N = 60` per bundle (`N = 30` for
  SWE-bench haiku) fits one Anthropic rolling window but is below the
  `N = 180`–`N = 700` used in the v2.0 battery. The live rerun partial
  reproduction should not be read as a large-`N` live validation; a
  wider `N = 180` live pass is tracked as future work.
- **v2.1 does NOT ship a new plugin build.** The v2.1 artifact set is
  the paper update, the live-HF scripts, and the consolidated
  `_summary_live.json` — not a new `pratyaksha-context-eng-harness-*.zip`.

## Reproducibility

From a clean checkout with `HF_TOKEN` set in `.env`:

```bash
uv sync
uv run python -m experiments.v2.p6a.run_live_hf \
    --scope core4 --live --continue-on-partial
uv run python -m tools.dev.live_hf_consolidate
uv run python -m tools.dev.sweb_outcomes
uv run pytest -q tests/dev/ tests/test_benchmarks/ \
    tests/test_v2/test_p6a_callers.py tests/test_v2/test_p6a_run.py
```

Each command is idempotent; `run_live_hf.py` resumes from per-bundle
JSONL checkpoints under `.cache/live_hf_checkpoints/` when interrupted
by an Anthropic rolling-window exhaustion. Expect ~60–90 minutes of
wall-clock per completed 5-hour window.

## Artifacts

- `experiments/results/p6a/_summary_live.json` — consolidated live
  bundle outcomes with pinned `hf_revision`.
- `experiments/results/p6a/swe_bench_outcomes.json` — haiku-only
  SWE-bench paired slice with full transparency about the CLI abort
  rate.
- `.cache/live_hf_checkpoints/*.jsonl` — per-record live run
  checkpoints (4 files, ~200 MB of JSONL).
- `paper/appendices/G_live_hf.tex` — Appendix G: live-HF protocol,
  provenance, and SWE-bench CLI limitation analysis.
- `paper/tables/T8_p6a_live_hf.{tex,md}` — Table T8 (live-HF summary).
- `paper/sections/08_results_l1.tex` — new subsection §8.9 (`live-hf-rerun-summary`) pointing to T8 and
  Appendix G.
- `paper/sections/11_discussion.tex` — updated threat paragraph on
  synthetic-fallback risk, now citing the live rerun.
- `paper/appendices/C_reproducibility.tex` — new §L1 live-HF rerun
  reproducibility block.
- `release/SHA256SUMS` — regenerated checksums for every release
  artifact including the v2.1 additions.

## Acknowledgements

The live-HF rerun was executed under three consecutive Anthropic
rolling-window budgets across 2026-04-20 and 2026-04-21. The
adversarial + code + performance reviewer fleet raised the
`strict_hf`, pre-registration-enforcement, `billed_input_tokens`,
`ralph_loop` timeout, and `hf_revision` provenance requirements in
parallel review passes that blocked the first live launch until they
were resolved.

Cite v2.0 for the full harness and paper; cite v2.1 only for the
live-HF rerun addendum.
