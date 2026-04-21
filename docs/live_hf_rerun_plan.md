# Live Hugging Face Rerun Plan (v2)

_Authoritative record of the live-HF rerun scope, locked after the 2026-04-18 adversarial audit and the full-quota restoration._

## Goal

Replace Layer-1 mock headlines in `paper/` and `docs/` with live-evidence numbers drawn from real Hugging Face datasets, at statistical power sufficient to detect Cohen's d >= 0.4 per bundle under the paired permutation test.

**Power caveat (not i.i.d.):** each bundle ships `15 examples × 2 seeds × 2 models = 60` paired observations, but the pairs share their HF example pool across models and seeds; observations are therefore correlated, not i.i.d. The ~0.8 nominal power at alpha=0.05 assumes independence and is an upper bound on the true per-bundle detectable-effect ceiling. Per-model paired tests (n≈30 pairs each) are the sharper read; the 60-pair pooled test is reported alongside but is explicitly flagged as non-independent in the paper and article. Dropping below d≈0.5 per model without clear structural rationale should be read as "effect-not-detected at this N", not "effect absent".

## Final live scope (7 bundles)

| # | Bundle label | Adapter | HF dataset | Tier | Paper anchor |
|---|---|---|---|---|---|
| 1 | `H1_ruler_8192_live` | `RulerNIAHAdapter` | `simonjegou/ruler` | 8192 | §8.1 |
| 2 | `H1_ruler_16384_live` | `RulerNIAHAdapter` | `simonjegou/ruler` | 16384 | §8.1 |
| 3 | `H1b_ruler_multi_8192_live` | `RulerNIAHMultiAdapter` | `simonjegou/ruler` | 8192 | §8.1 companion |
| 4 | `H1b_ruler_multi_16384_live` | `RulerNIAHMultiAdapter` | `simonjegou/ruler` | 16384 | §8.1 companion |
| 5 | `H_TQA_live_v2` | `TruthfulQAAdapter` | `truthful_qa` (generation) | n/a | §8 hallu |
| 6 | `H_HEQA_live_v2` | `HaluEvalQAAdapter` | `pminervini/HaluEval` (qa) | n/a | §8 hallu |
| 7 | `H_SWEB_live_n15` | `SWEBenchVerifiedAdapter` | `princeton-nlp/SWE-bench_Verified` | n/a | §10 supplement |

**Tier note:** `simonjegou/ruler` ships configs `{4096, 8192, 16384}` only. 32K is not live-reachable from this dataset; requesting it silently falls back to synthetic, which violates the live-HF mandate. The second tier is therefore 16384 — the largest genuinely-live RULER tier — and the paper language narrows its "32K" claims accordingly.

Every bundle uses `n_examples = 15`, `seeds = (0, 1)`, `models = (claude-haiku-4-5, claude-sonnet-4-6)` — 60 paired observations per bundle.

## Explicit drops and rationale

| Benchmark | Dropped because |
|---|---|
| HELMET-Recall | `HelmetRecallAdapter.load_real` is not wired; `_hf_unwired()` raises. Wiring requires schema discovery across 7 task families — well beyond 50 lines. Retain synthetic/mock HELMET numbers in paper with explicit disclosure. |
| HELMET-RAG | Same as HELMET-Recall; overlap with Recall anyway. |
| HaluEval-Discriminate | Binary variant of HaluEval-QA; redundant signal. |
| NoCha | No paper hypothesis binds to it; drop to save budget. |
| FACTS-Grounding | Public split lacks gold answers; needs judge-LLM adapter (separate project). |

## Hypothesis ID migration

The old live spec reused `hypothesis_id="H2"` for `ruler_niah_multi`, colliding with paper §8.2 HELMET-Recall. Renamed to `H1b` (RULER multi-key is structurally part of the long-context family). HELMET-Recall retains the paper's `H2` label.

## Archived pre-registration-violating artifacts

Moved to `.cache/archive/prereg_violation_20260418/`:
- `.cache/live_hf_checkpoints/H_TQA_live.jsonl` (15/240 partial) and `H_TQA_live_partial.json`
- `.cache/live_hf_checkpoints/H_HEQA_live.jsonl` (0/240) and `H_HEQA_live_partial.json`

Moved to `.cache/archive/ruler_n5_legacy/`:
- `experiments/results/p6a/H1_ruler_8192_live.json` (n=5, from initial spot-check)
- `experiments/results/p6a/H2_ruler_multi_8192_live.json` (n=5, legacy ID)

Rationale: the new v2 runs use pre-registered N and seeds; resuming from partial records would be a peeking / garden-of-forking-paths violation.

## Budget accounting

- Per-call Claude CLI overhead: ~45K input tokens (system prompt + tools + hooks).
- Per-bundle cost: 60 paired calls × ~45K = ~2.7M input tokens.
- Per-window cap: 2M input tokens / 5h.
- Per-bundle windows: ~2.
- Full-battery windows: 7 × 2 = 14 windows ≈ 70 wall-hours sequential, less with interleaving.

List-price cost estimate (cold cache, worst case):
- Haiku portion: ~1.35M tokens × $1/Mtok = ~$1.35
- Sonnet portion: ~1.35M tokens × $3/Mtok = ~$4.05
- Per bundle: ~$5–7
- Full battery: ~$35–50

## Ralph loop

A thin orchestrator (`tools/dev/ralph_live_loop.py`) runs `run_live_hf.py --scope full_battery --continue-on-partial` in a loop:
1. Launch run.
2. On `QuotaExhausted` (exit code 0 with partial artifacts), consolidate and re-check completion.
3. If all bundles are `status=complete`, exit.
4. Otherwise sleep until next 5h window and retry.

Safety caps: `--max-iterations`, `--max-wall-hours`, `--dry-run` probe for completion without touching the API.

## Reviewer gates

Before any window kickoff:
- `adversarial-reviewer`: scope + stats + overclaim audit.
- `code-reviewer`: adapter + spec + loop code.
- `performance-reviewer`: scheduler config + concurrency.

After final consolidation:
- Same three plus `security-reviewer` on subprocess wiring.

### Phase 0 review response (2026-04-18)

The first-round reviewer gate surfaced five blocker/high-severity issues;
all were fixed before Phase 1 kickoff:

1. **Silent synthetic fallback under `load_real=True`** (adversarial BLOCKER).
   Added `strict_hf: bool` field to `RulerNIAHAdapter`, `TruthfulQAAdapter`,
   `HaluEvalQAAdapter`, and `SWEBenchVerifiedAdapter`. When
   `load_real=True` *and* `strict_hf=True`, a failed HF load raises
   `RuntimeError` instead of silently falling back to synthetic. All seven
   live-battery bundles pass `strict_hf=True`. Defense-in-depth: the
   live runner also pulls a probe example before starting a bundle and
   refuses to run if `metadata["source"] != "huggingface"`.
2. **Pre-registration overridable via CLI** (adversarial BLOCKER).
   `run_live_hf.py` now rejects any `--ruler-n`, `--hallu-n`, `--sweb-n`,
   `--seeds`, or `--tiers` value that differs from the locked
   `LIVE_DEFAULT_*` constants when `--scope full_battery`. Overrides
   require explicit `--allow-override-preregistration`, which documents
   that the resulting receipt is *not* the locked run.
3. **Ralph loop does not honor quota-window reset time** (performance HIGH).
   `_compute_sleep_s` now parses `scheduler_status["next_window_at"]`
   (ISO-8601, already exposed by `CLIBudgetScheduler.status()`) and
   aligns the next iteration to that boundary + 30 s padding. Floor is
   `--min-sleep-s` (default 300 s); hard ceiling is 2 h to guard against
   bad scheduler state.
4. **Ralph loop ignores `--out-dir`** (code-review MAJOR). The
   consolidator and status probe are now both threaded through the same
   `out_dir`; `_done_probe` honors `--out-dir` end-to-end.
5. **No subprocess timeout** (code-review MAJOR). Inner `run_live_hf.py`
   invocations now have a 6-hour timeout (slightly above the 5-hour
   Claude window). Exceeding it returns exit code 124 and the outer
   loop resumes.

Additional hygiene fixes:
- `--tiers` help text corrected (previously said 32768 default).
- Completion predicate buckets any non-`complete` status (including
  unknown) as non-complete; unexpected statuses are logged.
- Ralph docstring corrected — the previous draft claimed the loop skips
  honest-null bundles, which it does not and should not.

Second-round minor fixes (2026-04-18, same day):
- `--models` added to the pre-registration lock. A `full_battery`
  invocation that passes `--models claude-haiku-4-5` alone (or any
  model list other than `(claude-haiku-4-5, claude-sonnet-4-6)`) now
  exits with code 2 unless `--allow-override-preregistration` is set.
- Probe provenance failures now raise a dedicated
  `ProvenanceIntegrityError` and are caught at the CLI entry point,
  exiting cleanly with code 3 instead of propagating a bare stack
  trace. The ralph loop treats code 3 as a terminal bundle error and
  surfaces it to the reviewer, rather than letting it masquerade as a
  quota event.
- Statistical-power language qualified (here, in `specs_live.py`, and
  in the paper narrative): the `60 paired observations` figure is an
  upper bound on sample count; because the same 15 HF examples feed
  every (seed, model) pair, the sharper read is the per-model paired
  test (n≈30 pairs), with the 60-pair pooled test reported alongside
  and flagged as non-i.i.d.

These changes retire the adversarial reviewer's "do not proceed"
verdict. Phase 1 (live loop) is unblocked.

## Completion criteria

1. All 7 bundles have `status=complete` in `experiments/results/p6a/_summary_live.json`.
2. Every bundle has `provenance.source = "huggingface"` and `provenance.load_real = true`.
3. Scheduler ledger shows no HALT events after the detector fix (only `QUOTA_EXHAUSTED` as graceful boundary).
4. `paper/main.tex` compiles with updated tables + appendix G.
5. `adversarial-reviewer` final pass returns zero blocker findings.

## Out of scope (not touched this pass)

- HELMET-Recall live wiring (separate project).
- Cross-model expansion to GPT-4o / Qwen / Llama.
- Human-vs-human κ on Khyātivāda.
- Full 500-instance live SWE-bench (this run uses 15 instances as a statistically-valid supplement, not a leaderboard entry).
- Re-running the L1 synthetic-mock baseline (kept byte-identical for pre-plan acceptance).
