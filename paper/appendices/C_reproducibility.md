# Reproducibility Manifest

This manifest gives a deterministic recipe to re-run every numerical claim in the paper. Every command below is exactly what we ran; outputs are written to `experiments/results/` under deterministic file names.

## Pinned dependencies

```
python  >= 3.11
uv      >= 0.4.0
anthropic>= 0.34
mcp     >= 1.0
pydantic>= 2.6
tiktoken>= 0.7
numpy   >= 1.26
pytest  >= 8.0
```

Optional (only for Qwen3 surprise backend in `boundary_compact`):

```
vllm  >= 0.5.0
transformers>= 4.43.0
torch >= 2.3.0
```

`tectonic` (for paper PDF) is installed via `brew install tectonic`. No other system dependencies are required.

## One-shot environment bootstrap

```bash
git clone https://github.com/SharathSPhD/pratyaksha-context-eng-harness.git
cd pratyaksha-context-eng-harness
uv sync                               # creates .venv, pins versions
uv run pytest tests/ -q               # 499 passes, 2 skipped
```

## Random-seed contract

Every experiment fixes seeds in three places:

1. The Python `random` module: `random.seed(seed)`.
2. NumPy: `np.random.default_rng(seed)`.
3. Anthropic API: an explicit `seed` keyword on every `Messages.create` call (when supported by the model).

Seeds used are `[1, 2, 3]` for the canonical L1/L3 runs, and a single `seed=0` for the deterministic L2 case study.

## Re-running each layer

### L1 — public benchmarks (H1–H7)

H1–H2 use the registered `BenchmarkAdapter` runner:

```bash
uv run python -m experiments.v2.p6a.run --hypotheses H1 H2 --seeds 1 2 3
```

H3–H7 use the deterministic plugin-in-process harness:

```bash
uv run python -m experiments.v2.p6a.run_plugin_inloop --hypotheses H3 H4 H5 H6 H7 --seeds 1 2 3
```

(Each module also accepts `all` in place of explicit hypothesis lists.)

Outputs: JSON under `experiments/results/p6a/` consumed by P7 (tables **T1**–**T3**, figures **F01**–**F07**).

### L2 — live case study (P6-B)

```bash
uv run python -m experiments.v2.p6b.run_case_study
```

Output: `experiments/results/p6b/summary.json` (deterministic across machines because the harness is LLM-free for this layer).

### L3 — SWE-bench Verified A/B (P6-C)

```bash
uv run python -m experiments.v2.p6c.run_swebench_ab \
  --n-instances 120 \
  --seeds 1 2 3 \
  --models claude-haiku-4-5 claude-sonnet-4-6 \
  --research-block-budget 8192
```

The headline numbers in the paper use **`--research-block-budget 8192`**. Fast CI smoke re-runs may pass **`--research-block-budget-fast 512`** instead.

Output: `experiments/results/p6c/summary.json` and per-instance traces under `experiments/results/p6c/runs/`.

### Aggregator — figures and tables (P7)

```bash
uv run python -m experiments.v2.p7.aggregate \
  --in experiments/results \
  --out experiments/results/p7
```

Output: `experiments/results/p7/figures/F01.png` … `F13.png`, `experiments/results/p7/tables/T1_*.md` … `T7_*.md`, plus `_index.json` and `_summary.md`.

## Cost ledger

The custom `CLIBudgetScheduler` writes `cost_ledger.db` with one row per outgoing API call. To re-derive total token usage and total wall-clock spend:

```bash
uv run python -m harness.scheduler.report --db cost_ledger.db
```

The ledger from our run is shipped as `experiments/results/cost_ledger.snapshot.db` for inspection.

## Determinism audit

`tests/test_v2/test_p7_aggregate.py::test_aggregate_is_deterministic` asserts that two consecutive runs of the aggregator produce byte-identical figure binaries and table strings. This pins the entire pipeline against silent stochastic drift.

## What is *not* deterministic

The L1 and L3 layers depend on the Anthropic API. Anthropic does not guarantee bitwise-identical completions across provisioning windows even with a fixed `seed`. We therefore report multi-seed means with bootstrap CIs and paired permutation tests, not point estimates. The deltas reported in Sections 8 and 10 are robust to single-seed drift on the order of $\pm 0.02$ on every metric we tested.

The L2 layer is fully deterministic: it is LLM-free.

## Worked-example payloads (Redis-caching turn, §6.7)

The five host-visible artefacts of the §6.7 / Fig.~\ref{fig:swimlane} worked example. All payloads are reproduced verbatim from `docs/worked_example_redis.jsonl` in the plugin repo; they replay byte-for-byte against the cached evidence trail under `examples/redis_session/`.

### `mcp__pratyaksha_mcp__manas_step` (request)

```json
{
  "jsonrpc": "2.0", "id": 1,
  "method": "tools/call",
  "params": {
    "name": "mcp__pratyaksha_mcp__manas_step",
    "arguments": {
      "query": "how do I cache a user session in Redis?",
      "k": 3,
      "budget_tokens": 1200
    }
  }
}
```

### `detect_conflict` (response excerpt)

```json
{
  "jsonrpc": "2.0", "id": 2,
  "result": {
    "conflicted": true,
    "kind": "TYPE_CLASH",
    "qualificand": "Redis-session",
    "qualifier_slot": "expiry-policy",
    "items": [
      {"id": "item_71", "qualifier": "use SETEX with ttl=3600",
       "condition": "blog 2019-03",   "precision": 0.55, "source": "blog"},
      {"id": "item_88", "qualifier": "use SET ... EX 3600 XX KEEPTTL",
       "condition": "Redis 7.x docs",  "precision": 0.95, "source": "docs"}
    ]
  }
}
```

### `sublate_with_evidence` (request + verdict)

```json
{
  "jsonrpc": "2.0", "id": 3,
  "method": "tools/call",
  "params": {
    "name": "sublate_with_evidence",
    "arguments": {
      "target_id":  "item_71",
      "by_id":      "item_88",
      "reason":     "Redis 7 docs (prec=8) > blog (prec=2) under shared limitor"
    }
  }
}
```

### `mcp__pratyaksha_mcp__buddhi_step` (final answer envelope)

```json
{
  "jsonrpc": "2.0", "id": 4,
  "result": {
    "answer":      "Use SET key value EX 3600 XX KEEPTTL ...",
    "khyati_class":"yathārtha",
    "confidence":   0.91,
    "grounding":  ["item_88"],
    "sublations": ["item_71"]
  }
}
```

### Sākṣī audit-log (one line per stage; `~/.cache/pratyaksha/audit.jsonl`)

```json
{"turn":42,"evt":"USER_TURN","query":"how do I cache a user session in Redis?","ts":1734519301.12}
{"turn":42,"evt":"ATTEND","k":3,"budget_tokens":1200,"items":["item_55","item_71","item_88"]}
{"turn":42,"evt":"BADHA_DETECT","kind":"TYPE_CLASH","slot":"Redis-session/expiry-policy"}
{"turn":42,"evt":"BADHA_RESOLVE","sublated":"item_71","survivor":"item_88","reason":"docs>blog","hash":"sha256:af3..."}
{"turn":42,"evt":"ANSWER","khyati":"yathārtha","conf":0.91,"grounding":["item_88"]}
```

### `/context-status` snapshot (post-turn, abridged)

```text
context window  :  3 live, 1 sublated, 0 evicted
budget          :  1{,}214 / 1{,}200 hard cap (101.2%, soft-warn)
sublations turn :  1 (item_71 → item_88, reason "docs>blog")
witness file    :  ~/.cache/pratyaksha/audit.jsonl  (last line: turn=42 evt=ANSWER)
khyati class    :  yathārtha   (conf 0.91)
```
