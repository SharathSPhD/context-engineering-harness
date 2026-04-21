# Release notes — Pratyakṣa Context-Engineering Harness v2.1.1 (power-extension addendum)

v2.1.1 is a **small, surgical, pre-registered amendment** on top of
v2.1. It doubles the paired-observation budget on two of the three
contingent v2.1 reads, carries a subprocess-timeout fix for the
third, and ships an offline scoring tool so partially-billed
`_ext` checkpoints produce deterministic final numbers without any
further CLI traffic. The plugin binary, the paper's central
architecture, and the philosophical vocabulary are unchanged. The
v2.1 rows in Table T8 are **unchanged**; the three v2.1.1 `_ext`
bundles sit alongside them as companion reads.

## Scope of v2.1.1

The v2.1 live-HF rerun left three conclusions contingent on the
N = 15 / N = 60 paired-observation budget:

- **RULER 16K** — directional but underpowered (*p* = 0.117, *d_z* = 0.265).
- **TruthfulQA** — null at *p* = 0.74, CI wide enough to leave a
  small positive effect compatible with the data.
- **SWE-bench Verified** — partial at the CLI layer: 77 % of
  `claude-haiku-4-5` attempts and 100 % of `claude-sonnet-4-6`
  attempts aborted inside the Claude CLI's `SessionStart` hook under
  the scheduler's 300 s subprocess timeout.

v2.1.1 addresses each minimally rather than opening a new omnibus:

| bundle                       | v2.1 N  | v2.1.1 N / fix                                              |
| ---------------------------- | ------- | ----------------------------------------------------------- |
| `H1_ruler_16384_live_ext`    | 15      | **30** per (model, seed, condition) cell                    |
| `H_TQA_live_v2_ext`          | 15      | **30** per cell                                             |
| `H_SWEB_live_ext`            | 15      | **15** (held); CLI timeout **300 s → 900 s**                |

The RULER and TruthfulQA adapters use a deterministic
`random.Random(seed).shuffle` over the Hugging Face revision at the
pinned commit, so the first 15 rows per cell of the N = 30 pull are a
strict superset of
the v2.1 N = 15 pull at the same seed. The v2.1 checkpoint JSONLs
are seeded verbatim into the `_ext` files so the first 15 rows per
cell replay as cache-hits and only the additional 15 bill.

We explicitly refuse to extend N on SWE-bench: the heuristic scorer
is floor-ceiling-bounded, so more paired observations there do not
buy statistical power. The v2.1.1 fix on SWE-bench is a pure
infrastructure fix (`scheduler_timeout_s = 900`) that converts
already-paid budget from CLI-aborts into scored patches.

## What actually happened

Anthropic's account-level rolling quota exhausted partway through
the amendment. After halting the run we treated the checkpoint
JSONLs as final and re-scored them **offline** (no further CLI
traffic) with the new `tools/dev/score_ext_checkpoints.py` script.

| bundle                       | n_paired | treatment | baseline | Δ        | 95 % CI            | paired *p* | *d_z*   | verdict                                          |
| ---------------------------- | -------- | --------- | -------- | -------- | ------------------ | ---------- | ------- | ------------------------------------------------ |
| `H1_ruler_16384_live_ext`    | 103      | 0.9903    | 0.9417   | +0.0485  | **[+0.010, +0.087]** | 0.064      | 0.225   | tighter CI (excludes 0), still below *p* gate   |
| `H_TQA_live_v2_ext`          | 60       | 0.0500    | 0.0833   | −0.0333  | —                  | 0.736      | −0.091  | null (byte-identical to v2.1; no new rows)      |
| `H_SWEB_live_ext`            | 1        | —         | —        | —        | —                  | —          | —       | CLI fix shipped, not exercised under load       |

**RULER 16K.** Haiku fully extended to n = 60; sonnet seed 0 to n = 28
(two scheduler-level CLI aborts) and sonnet seed 1 still at n = 15.
Per-model: haiku Δ = +0.050, *p* = 0.26; sonnet Δ = +0.047,
*p* = 0.51. Neither preregistered gate (Δ ≥ 0.05 **and** *p* ≤ 0.05)
clears; the reading is **tighter interval, same verdict —
directional, still underpowered**.

**TruthfulQA.** No new rows billed before the quota halted; the
checkpoint is byte-identical to v2.1's N = 60. The null stands.

**SWE-bench.** Only one paired (seed, instance) observation billed
before the quota halted, yielding essentially no new information.
The v2.1 `claude-haiku-4-5`-only row in Table T8 remains the
operative SWE-bench live read; the new 900 s subprocess-timeout
floor is shipped-but-unexercised.

## What's new in the codebase

### 1. Power-extension spec bundles

`experiments/v2/p6a/specs_live.py` adds `LIVE_EXT_RULER_N = 30`,
`LIVE_EXT_HALLU_N = 30`, `LIVE_EXT_SWEB_TIMEOUT_S = 900`, and a
`power_ext_specs()` constructor returning the three new
`_ext` bundles with their own hypothesis ids (`H1_ext`,
`H_TQA_v2_ext`, `H_SWEB_ext`).

### 2. `power_ext` scope and pre-registration lock

`experiments/v2/p6a/run_live_hf.py` adds a `--scope power_ext` arm
and a new `--scheduler-timeout-s` CLI flag. The pre-registration
lock rejects overrides of N, seeds, models, tiers, or the 900 s
subprocess timeout on the `power_ext` scope.

### 3. Offline checkpoint scorer

`tools/dev/score_ext_checkpoints.py` reads `_ext` checkpoint JSONLs,
reconstructs `BenchmarkRun` objects from the per-row records,
computes pooled + per-model paired statistics (bootstrap CI,
paired-permutation *p*, Cohen's *d_z*), and writes
`*_ext_score.json` side-cars next to the checkpoints. No CLI
traffic; deterministic; the three `_ext` score files are committed
as the v2.1.1 primary evidence.

### 4. Consolidator splices score overlays

`tools/dev/live_hf_consolidate.py` now finds `*_ext_score.json`
overlays and splices their pooled + per-model outcomes and caveats
into the corresponding `partial_quota` entries in
`experiments/results/p6a/_summary_live.json`, so the file grows
from 4 bundles (v2.1) to 7 bundles (v2.1 + three v2.1.1 `_ext`
companions).

### 5. SWE outcomes side-car emits both

`tools/dev/sweb_outcomes.py` now emits both the v2.1 haiku-only
paired slice (`experiments/results/p6a/swe_bench_outcomes.json`)
and the v2.1.1 haiku-only paired slice
(`experiments/results/p6a/swe_bench_outcomes_ext.json`, essentially
uninformative at n_pair = 1).

## What hasn't changed

- The plugin binary
  (`release/pratyaksha-context-eng-harness-v2.0.0.zip`)
- The paper's architecture, theory, and L1–L3 synthetic-fallback
  results
- The v2.1 live-HF Table T8 rows (the 4 original bundles)
- The v2.1 `_summary_live.json` completed-bundle entries
  (idempotent; byte-identical under replay)
- The philosophical vocabulary mapping (avacchedaka, bādha,
  manas/buddhi, sākṣī, khyātivāda)
- The Zenodo preprint DOI (cite the 19653013 record)

## Known limitations (carried from v2.1 and refined)

- **v2.1.1 is partial by design, not by bug.** The Anthropic
  rolling-window quota ended the v2.1.1 rollout after the RULER
  16K haiku + sonnet-seed-0 extensions and before the TruthfulQA
  and SWE-bench extensions. A future refresh window is the only
  remaining blocker; the code, the pre-registration, and the
  reproduction recipe all carry through without further edits.
- **RULER 16K sonnet is still at the v2.1 floor for seed 1.** The
  per-model read splits are included in the score JSON and the
  paper's Appendix G; neither seed, individually, is sufficient to
  clear the *d* gate at this budget.
- **The `SessionStart`-hook CLI abort is partially non-timeout.**
  During the v2.1.1 RULER 16K extension, two sonnet-seed-0 records
  still aborted at the CLI layer even with the 900 s timeout in
  effect. This suggests the CLI failure mode is partially a plain
  process exit on some invocations, not only a timeout. The two
  error rows are imputed as 0 per the pre-registered convention;
  the paper's Appendix G reports the sensitivity analysis.
- **v2.1.1 does NOT ship a new plugin build.** The v2.1.1 artifact
  set is the paper update (Table T8 addendum, §8 addendum, §11
  addendum, Appendix G § v2.1.1, Appendix C § L1 power-extension),
  the `score_ext_checkpoints.py` tool, the updated consolidator,
  and the three `_ext` checkpoint + score side-cars. No new
  `pratyaksha-context-eng-harness-*.zip`.

## Reproducibility

From a clean checkout with `HF_TOKEN` set in `.env`, after running
the v2.1 `core4` scope to populate the v2.1 checkpoints:

```bash
# Seed the _ext checkpoints from v2.1 (idempotent):
cp .cache/live_hf_checkpoints/H1_ruler_16384_live.jsonl \
   .cache/live_hf_checkpoints/H1_ruler_16384_live_ext.jsonl
cp .cache/live_hf_checkpoints/H_TQA_live_v2.jsonl \
   .cache/live_hf_checkpoints/H_TQA_live_v2_ext.jsonl
# SWE seed file: drop rows with error != "":
python3 -c "
import json, pathlib
src = pathlib.Path('.cache/live_hf_checkpoints/H_SWEB_live_n15.jsonl')
dst = pathlib.Path('.cache/live_hf_checkpoints/H_SWEB_live_ext.jsonl')
keep = [l for l in src.read_text().splitlines()
        if l.strip() and not json.loads(l).get('error')]
dst.write_text('\n'.join(keep) + '\n')
"

# (live run — requires Anthropic rolling-window budget).
# The `power_ext` pre-registration lock requires the N / tiers /
# timeout overrides below, since the CLI defaults still match v2.1.
# The `--bootstrap-n 10000 --permutation-n 10000` flags align the
# live scorer with `score_ext_checkpoints.py`, which uses
# RunnerConfig dataclass defaults (10_000) rather than the
# argparse-level default of 2_000.
uv run python -m experiments.v2.p6a.run_live_hf \
    --scope power_ext --live --continue-on-partial \
    --ruler-n 30 --hallu-n 30 --tiers 16384 \
    --scheduler-timeout-s 900 \
    --bootstrap-n 10000 --permutation-n 10000

# (offline re-score — no CLI traffic, deterministic)
uv run python -m tools.dev.score_ext_checkpoints
uv run python -m tools.dev.live_hf_consolidate
uv run python -m tools.dev.sweb_outcomes
```

Each command is idempotent. The offline re-score + consolidation
step produces the final v2.1.1 numbers from whatever rows were
billed before the quota halted; repeated runs against the same
checkpoints return byte-identical score JSON.

## Artifacts

- `experiments/results/p6a/_summary_live.json` — now 7 bundles
  (v2.1 + v2.1.1 companions) with pooled + per-model outcomes and
  caveats on every `_ext` entry.
- `experiments/results/p6a/swe_bench_outcomes.json` — v2.1
  haiku-only paired slice (unchanged).
- `experiments/results/p6a/swe_bench_outcomes_ext.json` — v2.1.1
  haiku-only paired slice (essentially uninformative).
- `.cache/live_hf_checkpoints/H1_ruler_16384_live_ext.jsonl` —
  RULER 16K extension checkpoint (208 rows, n_pair = 103).
- `.cache/live_hf_checkpoints/H_TQA_live_v2_ext.jsonl` —
  TruthfulQA extension checkpoint (byte-identical seed, 60 rows).
- `.cache/live_hf_checkpoints/H_SWEB_live_ext.jsonl` — SWE-bench
  extension checkpoint (14 rows, n_pair = 1 after imputation).
- `.cache/live_hf_checkpoints/*_ext_score.json` — offline score
  side-cars with pooled + per-model outcomes.
- `tools/dev/score_ext_checkpoints.py` — new offline scorer.
- `tools/dev/live_hf_consolidate.py` — extended to splice
  `*_ext_score.json` overlays into `_summary_live.json`.
- `tools/dev/sweb_outcomes.py` — extended to emit both v2.1 and
  v2.1.1 SWE-bench outcome side-cars.
- `paper/appendices/G_live_hf.tex` — Appendix G now includes the
  v2.1.1 power-extension addendum, pre-registration integrity
  statement, what-actually-happened narrative, and replay recipe.
- `paper/tables/T8_p6a_live_hf.tex` — Table T8 now has three
  additional rows under a v2.1.1 power-extension addendum header.
- `paper/sections/08_results_l1.tex` — §8 summary now includes a
  v2.1.1 power-extension addendum paragraph.
- `paper/sections/11_discussion.tex` — §11 synthetic-fallback
  threat paragraph now references v2.1.1 outcomes.
- `paper/appendices/C_reproducibility.tex` — §L1 reproducibility
  now has a power-extension replay recipe.
- `docs/article_substack.md` and `docs/index.html` — v2.1.1
  power-extension addendum block now published on both the Substack
  source and the public landing page.
- `release/pratyaksha-v2.1.1-preprint.pdf` — rebuilt preprint PDF.
- `release/SHA256SUMS` — regenerated checksums (now also covers
  `docs/article_substack.md` and `docs/index.html`).

## Acknowledgements

The v2.1.1 power-extension amendment was filed and the CLI
pre-registration lock was extended *before* any `_ext` row was
billed, so the three `_ext` bundles carry their own hypothesis ids
and never override the v2.1 rows. The amendment is closed in the
partial state recorded above; a future refresh window exercising
the shipped-but-unused SWE-bench CLI-timeout fix and the remaining
TruthfulQA + sonnet-seed-1 RULER 16K cells is queued as open work
but is not required for the v2.1.1 read to stand.

Cite v2.0 for the full harness and paper; cite v2.1 for the
original live-HF rerun addendum; cite v2.1.1 only for the
power-extension companion rows.
