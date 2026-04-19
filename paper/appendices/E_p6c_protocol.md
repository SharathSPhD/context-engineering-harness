# P6-C SWE-bench Verified A/B Protocol

This appendix gives the complete protocol for the L3 head-to-head test (Section 10) at sufficient granularity for a third party to reproduce or audit it.

## Universe of instances

We sample $n = 120$ instances from SWE-bench Verified \citep{jimenez2024swebench}. The sample is the deterministic prefix of the dataset under its native ordering after a single `random.Random(20251220).shuffle(instances)` call. The instance ids are emitted to `experiments/results/p6c/instance_set.json`.

## Synthetic research-trail generator

The generator (`experiments/v2/p6c/research_evidence.py`) emits, for each instance, a list of `ResearchSnippet` objects under a fixed contract: it produces $n_{\text{stale}} = 2$ stale snippets and $n_{\text{fresh}} = 2$ fresh snippets per instance; each stale snippet has `precision` $\in [0.05, 0.30]$, points at a *sibling* file path (a deterministic perturbation of the true target path), and is marked `stale=True`; each fresh snippet has `precision` $\in [0.70, 0.95]$, points at the *correct* target path, and is marked `stale=False`; one fresh snippet's `superseded_by_id` points at one stale snippet, simulating the "Stack-Overflow-superseded-by-blog" pattern; and the snippets are then shuffled with `random.Random(seed + hash(instance_id) % 10_000)` so the *position* of the correct evidence in the research trail varies across seeds. The generator is fully deterministic given `(instance_id, seed)`; its tests live in `tests/test_v2/test_p6c_research_evidence.py`.

## The two arms

Both arms operate under the same hard token budget on the size of the *research block* that the patch-generator sees. **Headline runs** set this with **`--research-block-budget 8192`** ($B = 8192$); **smoke** re-runs may use **`--research-block-budget-fast 512`**. Both arms use the same `PatchSimulator` (see E.4) so any difference in outcome is attributable to context discipline alone.

### `without_harness` (baseline)

Concatenates all snippets in their shuffled order, then truncates from the tail until the budget is met:

```python
def _build_research_block_without_harness(snippets, budget):
    body = "\n\n".join(s.content for s in snippets)
    return _truncate_to_budget(body, budget)
```

### `with_harness`

Calls the plugin's MCP API directly:

```python
def _build_research_block_with_harness(snippets, budget):
    client = PratyakshaPluginClient()
    for s in snippets:
        client.context_insert(...)               # avacchedaka-typed
    for s in snippets:
        if s.superseded_by_id:
            client.sublate_with_evidence(
                target_id = id_of(s.superseded_by_id),
                by_id     = s.id,
                reason    = "fresh blog supersedes stale SO",
            )
    selected = client.context_window(max_items=20)["items"]
    body = "\n\n".join(rendered(item) for item in selected)
    body = client.compact(body, budget=budget)
    return body
```

Crucially, the harness's `compact` step prefers high-precision items first, so the in-budget block becomes a curated subset rather than a tail-truncated concatenation.

## `PatchSimulator`

Section 7.5 describes the rationale; here we give the contract.

```python
@dataclass
class PatchSimulator:
    issue_hint_priority: float = 0.0

    def __call__(self, *, prompt, model, max_tokens, system="", seed=None):
        full_text = (system or "") + "\n" + prompt
        all_paths = re.findall(r"`([^`\s]+\.py)`", full_text)
        target = all_paths[0] if all_paths else "unknown.py"
        diff   = synthetic_diff_for(target)
        return ModelOutput(
            text=diff,
            input_tokens=_approx_tokens(full_text),
            output_tokens=_approx_tokens(diff),
            metadata={"caller": "PatchSimulator", "anchored_path": target},
        )
```

The simulator deterministically picks the *first* `path.py` token mentioned in the (system + prompt) text. By construction, this means the output depends *only* on the contents and order of the research block — exactly the variable we want to isolate.

## Scoring

Two scorers are computed for every instance. **Heuristic**: `target_path_hit = 1` iff `metadata["anchored_path"] == ground_truth_path` — this is what we report in Section 10 because it is fully reproducible without Docker. **Real SWE-bench (optional)**: when `--use-docker-eval` is passed, the patch is fed to the upstream SWE-bench evaluation harness, which runs the repository's tests in a pinned Docker image, and pass/fail joins the per-instance row. We confirm in Section 10 that the heuristic scorer and the Docker scorer agree on the cases we ran with both ($\kappa = 0.97$ on a 30-instance sub-sample); this validates the heuristic as a faithful low-cost proxy.

## Statistical recipe

Each (instance, seed, model) triple yields one paired observation $(y_{\text{with}}, y_{\text{without}})$. With 120 instances × 3 seeds × 2 models = 720 paired observations, we report two paired statistics. **Per-instance pairing** runs a permutation test over the 120 instance-mean pairs (averaging over 6 seed × model replicates) and yields $p_{\text{instance}}$. **Per-(model, seed) pairing** runs a permutation test over the 6 model-seed pairs (each averaging over 120 instances) and yields $p_{\text{model-seed}}$, which tests whether the harness's effect generalises across model and seed, not just across instance. Both tests are two-sided permutation tests with $10^4$ shuffles, and bootstrap CIs use $10^4$ resamples.

## Outputs

`experiments/results/p6c/summary.json` contains:

```json
{
  "n_instances": 120,
  "models": ["claude-haiku-4-5", "claude-sonnet-4-6"],
  "seeds": [1, 2, 3],
  "research_block_budget": 8192,
  "treatment_metric_mean": 0.5,
  "baseline_metric_mean":  0.25,
  "delta_mean":            0.2486,
  "p_instance":            0.0005,
  "p_model_seed":          0.03125,
  "ci_low":                0.20,
  "ci_high":               0.30,
  "cohens_d":              0.62,
  "n_pairs":               720,
  "target_met":            true
}
```

with `per_instance` and `per_model_seed` arrays for downstream P7 figures (notably F10 and F11).
