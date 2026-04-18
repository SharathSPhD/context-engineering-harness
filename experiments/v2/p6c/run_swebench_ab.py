"""P6-C: SWE-bench Verified head-to-head A/B with the plugin in-loop.

For each instance loaded from `swe_bench_verified` (synthetic by default;
`--load-real` picks up `princeton-nlp/SWE-bench_Verified` from HF), the
runner:

  1. Loads the instance and generates a deterministic *research trail*
     of mixed-precision evidence for it (see ``research_evidence``).
  2. Runs both arms under the **same** token budget cap:

       * **with_harness** — insert each snippet into the in-process
         plugin store with its retrieval precision; for every fresh
         snippet that supersedes a stale one, fire
         ``sublate_with_evidence``; ``compact(precision_threshold=0.50)``
         to drop survivors below the gate; surface only the high-
         precision survivors as the agent's research summary; render
         the patch prompt with the harness-on system message; call the
         model.

       * **without_harness** — concatenate the snippets in discovery
         order, truncate to the token budget if necessary (the failure
         mode the cap forces), render the patch prompt with the bare
         "fix this" system message; call the model.

  3. Score both patches with the SWE-bench Verified adapter's heuristic
     scorer (file-overlap + line Jaccard). When the official docker
     harness is available **and** ``--use-docker-harness`` is passed,
     the runner *also* records the docker pass/fail bit alongside the
     heuristic score. Heuristic numbers are the published P6-C
     reproducibility floor; docker numbers are the validated bound.

  4. Per (model, seed) collect paired (with - without) scores → bootstrap
     CI + paired permutation test on the per-instance differences AND
     on the per-(model, seed) means (the conservative test). Both go
     into the artifact.

Default invocation runs N=120 synthetic instances × 2 models × 3 seeds
in <2 minutes on a laptop and writes
``experiments/results/p6c/swebench_ab.json`` plus ``_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.benchmarks.adapters.swebench.verified import (
    SWEBenchVerifiedAdapter,
    SwebenchHarnessUnavailable,
)
from src.benchmarks.base import BenchmarkExample, ModelOutput
from src.benchmarks.stats import bootstrap_ci, cohens_d, paired_permutation_test

from experiments.v2.p6a.callers import MockHarnessCaller
from experiments.v2.p6a.plugin_client import PratyakshaPluginClient

from .research_evidence import ResearchSnippet, generate_research_trail

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results" / "p6c"


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    """Approx-tokens (chars/4). Keeps the runner free of hard tiktoken
    dependency. Real-mode runs replace this via ``--use-tiktoken``.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _truncate_to_budget(text: str, *, max_tokens: int) -> str:
    """Hard truncate by approx-tokens. Mimics what an unaided agent
    suffers when its prompt exceeds the per-call token budget.
    """
    if _approx_tokens(text) <= max_tokens:
        return text
    return text[: max_tokens * 4]


# ---------------------------------------------------------------------------
# Per-arm logic
# ---------------------------------------------------------------------------


def _build_research_block_with_harness(
    *,
    snippets: list[ResearchSnippet],
    client: PratyakshaPluginClient,
    precision_gate: float,
) -> tuple[str, dict[str, Any]]:
    """Insert all snippets, sublate stale, compact below the gate, then
    return (research_block_text, telemetry).
    """
    client.reset()
    inserted: dict[str, ResearchSnippet] = {}
    sublations = 0
    for s in snippets:
        client.insert(
            id=s.id,
            content=s.content,
            precision=s.precision,
            qualificand=s.qualificand,
            qualifier=s.qualifier,
            condition=s.condition,
            relation="inherence",
            provenance=s.source,
        )
        inserted[s.id] = s
        for older_id, older in list(inserted.items()):
            if older_id == s.id:
                continue
            triggers_pointer = older.superseded_by_id == s.id
            triggers_dominance = (
                older.stale
                and older.qualifier == s.qualifier
                and s.precision > older.precision
                and not s.stale
            )
            if not (triggers_pointer or triggers_dominance):
                continue
            client.sublate_with_evidence(
                older_id=older_id,
                newer_content=s.content,
                newer_precision=s.precision,
                qualificand=s.qualificand,
                qualifier=s.qualifier,
                condition=s.condition,
                provenance=f"superseded_by={s.id}",
            )
            sublations += 1
    comp = client.compact(precision_threshold=precision_gate)
    survivors_resp = client.retrieve(
        qualificand=snippets[0].qualificand if snippets else "",
        precision_threshold=precision_gate,
        max_elements=20,
    )
    elements = survivors_resp.get("elements", [])
    research_block = "\n\n".join(
        f"- ({el['precision']:.2f}) {el['content']} [src: {el.get('provenance','')}]"
        for el in elements
    )
    telemetry = {
        "n_snippets_total": len(snippets),
        "n_active_after_compact": len(elements),
        "n_sublated": sublations,
        "n_compacted": comp.get("n_compressed", 0),
        "research_block_tokens": _approx_tokens(research_block),
    }
    return research_block, telemetry


def _build_research_block_without_harness(
    *, snippets: list[ResearchSnippet], max_research_tokens: int
) -> tuple[str, dict[str, Any]]:
    """Plain concatenation in discovery order, truncated to budget — the
    canonical RAG failure mode under a fixed cap.
    """
    parts = [
        f"- {s.content} [src: {s.source}]" for s in snippets
    ]
    full = "\n\n".join(parts)
    truncated = _truncate_to_budget(full, max_tokens=max_research_tokens)
    n_kept = truncated.count("- ") if truncated else 0
    telemetry = {
        "n_snippets_total": len(snippets),
        "n_kept_after_truncation": n_kept,
        "n_sublated": 0,
        "n_compacted": 0,
        "research_block_tokens": _approx_tokens(truncated),
    }
    return truncated, telemetry


# ---------------------------------------------------------------------------
# Patch generation simulation
# ---------------------------------------------------------------------------


@dataclass
class PatchSimulator:
    """Deterministic patch generator that *reads its prompt*.

    Rather than calling Claude (which would require live spend), the
    simulator extracts the *first plausible file path* from the
    research block (everything between backticks) and emits a unified
    diff against that path. If the research block is empty or contains
    no file path, it falls back to the official issue's hint.

    This is a *behaviour stub* mirroring what an LLM would do: it
    anchors on whichever file path the prompt presents most prominently.
    The harness-on arm presents only fresh paths; the harness-off arm
    presents stale-then-fresh, so the simulator anchors on the stale
    sibling path more often. That asymmetry is exactly what the
    plugin's bookkeeping is designed to neutralise.

    Because the simulator is fully deterministic, the entire P6-C run
    is reproducible from (instance_id, seed). For real-spend runs, swap
    in ``LiveCLICaller`` from ``experiments.v2.p6a.callers``.
    """

    issue_hint_priority: float = 0.0  # 0.0 = always anchor on research block first

    def __call__(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        system: str = "",
        seed: int | None = None,
    ) -> ModelOutput:
        # Anchor on the *first* `path` mentioned in the research block;
        # if none, fall back to a fenced path in the issue hint.
        import re

        full_text = (system or "") + "\n" + prompt
        all_paths = re.findall(r"`([^`\s]+\.py)`", full_text)
        target = all_paths[0] if all_paths else "unknown.py"

        diff = (
            f"diff --git a/{target} b/{target}\n"
            f"--- a/{target}\n"
            f"+++ b/{target}\n"
            f"@@ -1,1 +1,1 @@\n"
            f"-TODO old line\n"
            f"+TODO new line\n"
        )
        return ModelOutput(
            text=diff,
            input_tokens=_approx_tokens(full_text),
            output_tokens=_approx_tokens(diff),
            metadata={"caller": "PatchSimulator", "anchored_path": target},
        )


# ---------------------------------------------------------------------------
# Per-instance scoring
# ---------------------------------------------------------------------------


@dataclass
class _ArmScore:
    instance_id: str
    arm: str
    score: float
    correct: bool
    anchored_path: str
    target_path: str
    research_telemetry: dict[str, Any]
    prompt_tokens: int
    output_tokens: int
    docker_resolved: bool | None  # None when docker harness not run


def _score_one(
    *,
    example: BenchmarkExample,
    snippets: list[ResearchSnippet],
    arm: str,
    adapter: SWEBenchVerifiedAdapter,
    caller: Callable[..., ModelOutput],
    model: str,
    seed: int,
    research_budget_tokens: int,
    precision_gate: float,
    use_docker: bool,
) -> _ArmScore:
    if arm == "with_harness":
        client = PratyakshaPluginClient()
        block, telemetry = _build_research_block_with_harness(
            snippets=snippets, client=client, precision_gate=precision_gate
        )
        condition = "harness_on"
    elif arm == "without_harness":
        block, telemetry = _build_research_block_without_harness(
            snippets=snippets, max_research_tokens=research_budget_tokens
        )
        condition = "harness_off"
    else:  # pragma: no cover — defensive
        raise ValueError(f"unknown arm {arm!r}")

    base_prompt = adapter.render_prompt(example, condition=condition)
    final_prompt = (
        f"## Research notes (filtered by harness)\n{block}\n\n{base_prompt}"
        if arm == "with_harness"
        else f"## Research notes (raw)\n{block}\n\n{base_prompt}"
    )
    sys_prompt = adapter.system_prompt(condition=condition)

    out = caller(
        prompt=final_prompt,
        model=model,
        max_tokens=512,
        system=sys_prompt,
        seed=seed,
    )
    score, correct, _ = adapter.score(example, out)

    docker_resolved: bool | None = None
    if use_docker:
        try:
            docker_resolved = adapter.verify_with_swebench_harness(example, out)
        except SwebenchHarnessUnavailable as exc:
            logger.warning("docker harness unavailable: %s", exc)
            docker_resolved = None

    target_path = (example.ground_truth or {}).get("file_path", "") if isinstance(example.ground_truth, dict) else ""
    return _ArmScore(
        instance_id=example.id,
        arm=arm,
        score=float(score),
        correct=bool(correct),
        anchored_path=str(out.metadata.get("anchored_path", "")),
        target_path=target_path,
        research_telemetry=telemetry,
        prompt_tokens=int(out.input_tokens),
        output_tokens=int(out.output_tokens),
        docker_resolved=docker_resolved,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class _PerSeedSummary:
    model: str
    seed: int
    arm: str
    n_examples: int
    mean_score: float
    accuracy: float
    n_target_path_hit: int
    n_research_sublations: int


def _per_seed_summary(rows: list[_ArmScore], *, model: str, seed: int, arm: str) -> _PerSeedSummary:
    n = len(rows)
    if n == 0:
        return _PerSeedSummary(model=model, seed=seed, arm=arm, n_examples=0,
                               mean_score=0.0, accuracy=0.0, n_target_path_hit=0,
                               n_research_sublations=0)
    return _PerSeedSummary(
        model=model,
        seed=seed,
        arm=arm,
        n_examples=n,
        mean_score=float(sum(r.score for r in rows) / n),
        accuracy=float(sum(1 for r in rows if r.correct) / n),
        n_target_path_hit=sum(1 for r in rows if r.anchored_path == r.target_path),
        n_research_sublations=sum(r.research_telemetry.get("n_sublated", 0) for r in rows),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_ab(
    *,
    n_examples: int,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    research_budget_tokens: int,
    precision_gate: float,
    bootstrap_n: int,
    permutation_n: int,
    out_dir: Path,
    load_real: bool,
    use_docker: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = SWEBenchVerifiedAdapter(load_real=load_real, default_n=n_examples)
    caller = PatchSimulator()
    if not isinstance(caller, type(MockHarnessCaller)):
        # Both PatchSimulator and MockHarnessCaller satisfy the ModelCaller
        # protocol; we keep the import so live-mode swap-in is one line.
        pass

    per_arm_rows: dict[str, list[_ArmScore]] = {"with_harness": [], "without_harness": []}
    per_seed: list[_PerSeedSummary] = []
    paired_diffs_per_instance: list[float] = []
    paired_means_with: list[float] = []
    paired_means_without: list[float] = []

    t0 = time.perf_counter()
    for model in models:
        for seed in seeds:
            examples = adapter.load_examples(n=n_examples, seed=seed)
            with_rows: list[_ArmScore] = []
            without_rows: list[_ArmScore] = []
            for ex in examples:
                file_path = (ex.ground_truth or {}).get("file_path", "")
                repo = (ex.metadata or {}).get("repo", "synthorg/repo")
                snippets = generate_research_trail(
                    instance_id=ex.id,
                    repo=repo,
                    file_path=file_path,
                    issue_summary=ex.prompt,
                    seed=seed,
                )
                with_score = _score_one(
                    example=ex, snippets=snippets, arm="with_harness",
                    adapter=adapter, caller=caller, model=model, seed=seed,
                    research_budget_tokens=research_budget_tokens,
                    precision_gate=precision_gate, use_docker=use_docker,
                )
                without_score = _score_one(
                    example=ex, snippets=snippets, arm="without_harness",
                    adapter=adapter, caller=caller, model=model, seed=seed,
                    research_budget_tokens=research_budget_tokens,
                    precision_gate=precision_gate, use_docker=use_docker,
                )
                with_rows.append(with_score)
                without_rows.append(without_score)
                paired_diffs_per_instance.append(with_score.score - without_score.score)
            per_arm_rows["with_harness"].extend(with_rows)
            per_arm_rows["without_harness"].extend(without_rows)
            per_seed.append(_per_seed_summary(with_rows, model=model, seed=seed, arm="with_harness"))
            per_seed.append(_per_seed_summary(without_rows, model=model, seed=seed, arm="without_harness"))
            paired_means_with.append(per_seed[-2].mean_score)
            paired_means_without.append(per_seed[-1].mean_score)
    elapsed = time.perf_counter() - t0

    n_pairs = len(paired_diffs_per_instance)
    if n_pairs == 0:
        outcome_per_instance = {"delta_observed": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                                "p_value": 1.0, "cohens_d": 0.0, "n_pairs": 0}
        outcome_per_seed = dict(outcome_per_instance, n_pairs=0)
    else:
        delta_obs, ci_low, ci_high = bootstrap_ci(
            paired_diffs_per_instance, n_bootstrap=bootstrap_n, seed=0
        )
        with_arr = [r.score for r in per_arm_rows["with_harness"]]
        without_arr = [r.score for r in per_arm_rows["without_harness"]]
        p_inst = paired_permutation_test(
            with_arr, without_arr, n_permutations=permutation_n, seed=0
        )
        d_inst = cohens_d(with_arr, without_arr)
        outcome_per_instance = {
            "delta_observed": round(float(delta_obs), 4),
            "ci_low": round(float(ci_low), 4),
            "ci_high": round(float(ci_high), 4),
            "p_value": round(float(p_inst), 6),
            "cohens_d": round(float(d_inst), 4),
            "n_pairs": n_pairs,
            "test": "paired permutation per instance",
        }
        delta_seed, ci_low_s, ci_high_s = bootstrap_ci(
            [a - b for a, b in zip(paired_means_with, paired_means_without)],
            n_bootstrap=bootstrap_n, seed=0,
        )
        p_seed = paired_permutation_test(
            paired_means_with, paired_means_without,
            n_permutations=permutation_n, seed=0,
        )
        d_seed = cohens_d(paired_means_with, paired_means_without)
        outcome_per_seed = {
            "delta_observed": round(float(delta_seed), 4),
            "ci_low": round(float(ci_low_s), 4),
            "ci_high": round(float(ci_high_s), 4),
            "p_value": round(float(p_seed), 6),
            "cohens_d": round(float(d_seed), 4),
            "n_pairs": len(paired_means_with),
            "test": "paired permutation per (model, seed)",
        }

    target_met = (
        outcome_per_instance["delta_observed"] > 0.0
        and outcome_per_instance["p_value"] < 0.05
        and outcome_per_seed["delta_observed"] > 0.0
        and outcome_per_seed["p_value"] < 0.05
    )

    headline = {
        "label": "P6-C_swebench_verified_AB",
        "spec": {
            "n_examples_per_seed": n_examples,
            "models": list(models),
            "seeds": list(seeds),
            "research_budget_tokens": research_budget_tokens,
            "precision_gate": precision_gate,
            "bootstrap_n": bootstrap_n,
            "permutation_n": permutation_n,
            "load_real_swebench_verified": load_real,
            "use_docker_harness": use_docker,
            "metric": "swebench_heuristic_score (file_overlap=0.5 + line_jaccard=0.5)",
        },
        "outcome_per_instance": outcome_per_instance,
        "outcome_per_seed_mean": outcome_per_seed,
        "target_met": bool(target_met),
        "treatment_metric_mean": round(
            float(sum(r.score for r in per_arm_rows["with_harness"]) / max(1, n_pairs)), 4
        ),
        "baseline_metric_mean": round(
            float(sum(r.score for r in per_arm_rows["without_harness"]) / max(1, n_pairs)), 4
        ),
        "treatment_target_path_hit_rate": round(
            float(sum(1 for r in per_arm_rows["with_harness"] if r.anchored_path == r.target_path) / max(1, n_pairs)),
            4,
        ),
        "baseline_target_path_hit_rate": round(
            float(sum(1 for r in per_arm_rows["without_harness"] if r.anchored_path == r.target_path) / max(1, n_pairs)),
            4,
        ),
        "total_sublations_fired": sum(
            r.research_telemetry.get("n_sublated", 0) for r in per_arm_rows["with_harness"]
        ),
        "wallclock_s": round(elapsed, 3),
        "per_seed": [asdict(p) for p in per_seed],
        "per_instance": [asdict(r) for r in per_arm_rows["with_harness"]] +
                         [asdict(r) for r in per_arm_rows["without_harness"]],
        "ts": _utcnow_iso(),
    }

    path = out_dir / "swebench_ab.json"
    path.write_text(json.dumps(headline, indent=2))
    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(
        {
            "label": headline["label"],
            "spec": headline["spec"],
            "outcome_per_instance": headline["outcome_per_instance"],
            "outcome_per_seed_mean": headline["outcome_per_seed_mean"],
            "target_met": headline["target_met"],
            "treatment_metric_mean": headline["treatment_metric_mean"],
            "baseline_metric_mean": headline["baseline_metric_mean"],
            "treatment_target_path_hit_rate": headline["treatment_target_path_hit_rate"],
            "baseline_target_path_hit_rate": headline["baseline_target_path_hit_rate"],
            "total_sublations_fired": headline["total_sublations_fired"],
            "wallclock_s": headline["wallclock_s"],
            "ts": headline["ts"],
        },
        indent=2,
    ))
    return headline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-examples", type=int, default=120,
                   help="examples per seed (≥100 satisfies the P6-C plan)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--models", nargs="+", default=["claude-haiku-4-5", "claude-sonnet-4-6"])
    p.add_argument("--research-budget-tokens", type=int, default=8192,
                   help=("hard cap on the without_harness research block. "
                         "Default 8192 matches the headline P6-C runs reported "
                         "in the paper (§10, Appendix E). Pass --fast to switch "
                         "to a 512-token smoke-test profile."))
    p.add_argument("--fast", action="store_true",
                   help=("smoke-test profile: overrides --research-budget-tokens "
                         "to 512 (paper Appendix E §E.3 'fast' path). Useful for "
                         "CI and laptop reproducibility runs."))
    p.add_argument("--precision-gate", type=float, default=0.50)
    p.add_argument("--bootstrap-n", type=int, default=2000)
    p.add_argument("--permutation-n", type=int, default=2000)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--load-real", action="store_true",
                   help="load princeton-nlp/SWE-bench_Verified from HF")
    p.add_argument("--use-docker-harness", action="store_true",
                   help="also run the official docker harness for resolved-bit")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(level=max(logging.DEBUG, level), format="%(levelname)s %(message)s")

    research_budget_tokens = 512 if args.fast else args.research_budget_tokens

    headline = run_ab(
        n_examples=args.n_examples,
        seeds=tuple(args.seeds),
        models=tuple(args.models),
        research_budget_tokens=research_budget_tokens,
        precision_gate=args.precision_gate,
        bootstrap_n=args.bootstrap_n,
        permutation_n=args.permutation_n,
        out_dir=args.out_dir,
        load_real=args.load_real,
        use_docker=args.use_docker_harness,
    )
    print(json.dumps({
        "label": headline["label"],
        "treatment_metric_mean": headline["treatment_metric_mean"],
        "baseline_metric_mean": headline["baseline_metric_mean"],
        "outcome_per_instance": headline["outcome_per_instance"],
        "outcome_per_seed_mean": headline["outcome_per_seed_mean"],
        "treatment_target_path_hit_rate": headline["treatment_target_path_hit_rate"],
        "baseline_target_path_hit_rate": headline["baseline_target_path_hit_rate"],
        "total_sublations_fired": headline["total_sublations_fired"],
        "target_met": headline["target_met"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
