"""Live-HF spec bundles for the Phase-1 / v2 full-battery re-runs.

These bundles are **not** imported by the mock-mode P6-A runner; they live
in their own module so the synthetic baseline in ``specs.py`` stays
byte-identical to its pre-plan state (preprint acceptance criterion).

Bundles exposed (locked 2026-04-18 after adversarial audit; see
``docs/live_hf_rerun_plan.md``):

* ``h1_ruler_live_specs()`` — single-needle RULER NIAH over
  ``simonjegou/ruler``, per token-tier (8192, 16384). 32768 is not a
  published config on this dataset and would silently fall back to
  synthetic, so it is excluded.
* ``h1b_ruler_multi_live_specs()`` — multi-needle RULER NIAH over the
  same dataset. Renamed from the earlier ``h2_ruler_multi_live_specs`` to
  free up ``hypothesis_id="H2"`` for paper §8.2 (HELMET-Recall).
* ``h_tqa_live_v2_spec()`` — TruthfulQA at pre-registered N=15.
* ``h_heqa_live_v2_spec()`` — HaluEval-QA at pre-registered N=15.
* ``h_swebench_verified_live_n15_spec()`` — SWE-bench Verified live
  supplement (heuristic scorer, 15 instances × 2 seeds × 2 models).
* ``full_battery_specs()`` — returns all seven live bundles in fixed
  order; this is what ``run_live_hf.py --scope full_battery`` loads.

Explicitly dropped from live scope (noted here to prevent accidental
re-adds): HELMET-Recall (``load_real`` not wired — see
``src/benchmarks/adapters/longctx/helmet.py::_hf_unwired``), HELMET-RAG
(same), HaluEval-Discriminate (binary variant of HEQA), NoCha (no paper
hypothesis binds to it), FACTS-Grounding (public split lacks gold
answers). All retain their synthetic/mock signals in the paper with
explicit disclosure.

Each bundle's ``adapter_kwargs`` flips ``load_real=True`` *and*
``strict_hf=True`` explicitly. The ``strict_hf`` flag makes the adapter
raise rather than silently fall back to synthetic if the HF dataset
cannot be loaded — that silent fallback was the exact scientific-
integrity failure mode flagged in the 2026-04-18 adversarial review.
"""
from __future__ import annotations

from src.benchmarks.hypothesis import HypothesisSpec, TargetDirection

from .specs import DEFAULT_MODELS, P6ASpecBundle  # noqa: F401 — re-export

# Live-HF defaults are intentionally smaller than the synthetic defaults:
# every example is a paid CLI call, not a free mock. The locked
# minimum-viable statistical spec is N=15 examples × 2 seeds × 2 models
# = 60 paired observations per bundle. The ~0.8 nominal power at α=0.05
# for Cohen's d ≥ 0.4 on a paired permutation test is an *upper bound*
# that assumes independence; because the same 15 HF examples feed every
# (seed, model) pair, observations are correlated, so the sharper read
# is the per-model paired test (n≈30 pairs each) with the 60-pair
# pooled test reported alongside and explicitly flagged as non-i.i.d.
# Effects below d≈0.5 per model at this N should be read as
# "not detected at N=15" rather than "absent".
LIVE_DEFAULT_SEEDS: tuple[int, ...] = (0, 1)
LIVE_DEFAULT_RULER_N: int = 15
LIVE_DEFAULT_HALLU_N: int = 15
LIVE_DEFAULT_SWEB_N: int = 15
# RULER real configs on simonjegou/ruler ship 4096, 8192, 16384 only —
# 32768 does not resolve and silently falls back to synthetic, which
# violates the live-HF mandate. 16384 is the largest genuinely-live
# tier; paper language updates accordingly.
LIVE_DEFAULT_TIERS: tuple[int, ...] = (8_192, 16_384)

# HF dataset IDs mirrored here so provenance blocks stay stable even if
# the adapter defaults drift.
HF_IDS: dict[str, dict[str, str]] = {
    "ruler_niah": {"dataset": "simonjegou/ruler", "task": "niah_single_1"},
    "ruler_niah_multi": {"dataset": "simonjegou/ruler", "task": "niah_multikey_1"},
    "truthful_qa": {"dataset": "truthful_qa", "config": "generation"},
    "halu_eval_qa": {"dataset": "pminervini/HaluEval", "config": "qa"},
    "swe_bench_verified": {"dataset": "princeton-nlp/SWE-bench_Verified"},
}


# --- RULER (H1 single-needle) ----------------------------------------


def h1_ruler_live_specs(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = LIVE_DEFAULT_SEEDS,
    n_examples: int = LIVE_DEFAULT_RULER_N,
    target_tokens_tiers: tuple[int, ...] = LIVE_DEFAULT_TIERS,
) -> list[P6ASpecBundle]:
    """One live RULER NIAH single-needle bundle per token-tier.

    Label shape: ``H1_ruler_<tier>_live`` — keeps the synthetic file
    names (``H1_ruler_<tier>.json``) unshadowed on disk.
    """
    out: list[P6ASpecBundle] = []
    for tokens in target_tokens_tiers:
        out.append(
            P6ASpecBundle(
                label=f"H1_ruler_{tokens}_live",
                spec=HypothesisSpec(
                    hypothesis_id="H1",
                    description=(
                        "[LIVE-HF] Avacchedaka structured retriever prompt lifts "
                        f"RULER NIAH accuracy at {tokens}-token contexts on the "
                        "real simonjegou/ruler dataset; effect ≥ 5 pts."
                    ),
                    adapter_name="ruler_niah",
                    treatment_condition="harness_on",
                    baseline_condition="harness_off",
                    metric="accuracy",
                    direction=TargetDirection.GREATER,
                    delta=0.05,
                    n_examples=n_examples,
                    seeds=seeds,
                    models=models,
                    significance_alpha=0.05,
                    notes=(
                        f"Tier={tokens} tokens. load_real=True, "
                        "dataset=simonjegou/ruler, task=niah_single_1. "
                        f"Pre-registered N={n_examples}, seeds={list(seeds)}. "
                        "Stopping rule: stop at N, no peeking."
                    ),
                ),
                adapter_kwargs={
                    "target_tokens": tokens,
                    "default_n": n_examples,
                    "load_real": True,
                    "strict_hf": True,
                    "hf_task_filter": "niah_single_1",
                },
            )
        )
    return out


# --- RULER (H1b multi-needle) ----------------------------------------


def h1b_ruler_multi_live_specs(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = LIVE_DEFAULT_SEEDS,
    n_examples: int = LIVE_DEFAULT_RULER_N,
    target_tokens_tiers: tuple[int, ...] = LIVE_DEFAULT_TIERS,
) -> list[P6ASpecBundle]:
    """RULER multi-needle variant of H1 (paper §8.1 companion).

    Renamed from ``h2_ruler_multi_live_specs`` to use ``hypothesis_id="H1b"``.
    The prior ``H2`` label collided with paper §8.2 (HELMET-Recall); since
    multi-key RULER is a structural sibling of single-key RULER, promoting
    it to H1b is more honest than reusing H2.
    """
    out: list[P6ASpecBundle] = []
    for tokens in target_tokens_tiers:
        out.append(
            P6ASpecBundle(
                label=f"H1b_ruler_multi_{tokens}_live",
                spec=HypothesisSpec(
                    hypothesis_id="H1b",
                    description=(
                        "[LIVE-HF] Avacchedaka structured retriever prompt lifts "
                        f"RULER multi-key NIAH accuracy at {tokens}-token "
                        "contexts on the real simonjegou/ruler dataset."
                    ),
                    adapter_name="ruler_niah_multi",
                    treatment_condition="harness_on",
                    baseline_condition="harness_off",
                    metric="score",
                    direction=TargetDirection.GREATER,
                    delta=0.10,
                    n_examples=n_examples,
                    seeds=seeds,
                    models=models,
                    significance_alpha=0.05,
                    notes=(
                        f"Tier={tokens} tokens, k=4. load_real=True, "
                        "dataset=simonjegou/ruler, task=niah_multikey_1. "
                        f"Pre-registered N={n_examples}, seeds={list(seeds)}."
                    ),
                ),
                adapter_kwargs={
                    "target_tokens": tokens,
                    "default_n": n_examples,
                    "load_real": True,
                    "strict_hf": True,
                    "hf_task_filter": "niah_multikey_1",
                },
            )
        )
    return out


# --- Hallucination (TruthfulQA + HaluEval-QA v2) ---------------------


def h_tqa_live_v2_spec(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = LIVE_DEFAULT_SEEDS,
    n_examples: int = LIVE_DEFAULT_HALLU_N,
) -> P6ASpecBundle:
    """TruthfulQA v2 live bundle at pre-registered N=15.

    ``_v2`` suffix distinguishes from the archived ``H_TQA_live``
    partial, which peeked at N=15/240 before the pre-registration was
    locked. Label + hypothesis_id are distinct so the v2 run cannot
    resume the archived checkpoint.
    """
    return P6ASpecBundle(
        label="H_TQA_live_v2",
        spec=HypothesisSpec(
            hypothesis_id="H_TQA_v2",
            description=(
                "[LIVE-HF] Harness misconception-self-check lifts TruthfulQA "
                "truthful rate on the real truthful_qa/generation split. "
                "Pre-registered N=15, seeds=(0,1), no peeking."
            ),
            adapter_name="truthful_qa",
            treatment_condition="harness_on",
            baseline_condition="harness_off",
            metric="accuracy",
            direction=TargetDirection.GREATER,
            delta=0.05,
            n_examples=n_examples,
            seeds=seeds,
            models=models,
            significance_alpha=0.05,
            notes=(
                "load_real=True, dataset=truthful_qa, config=generation. "
                "Scorer uses substring matching over the HF `correct_answers` "
                "and `incorrect_answers` lists — documented brittleness; see "
                "findings block in the output JSON."
            ),
        ),
        adapter_kwargs={"default_n": n_examples, "load_real": True, "strict_hf": True},
    )


def h_heqa_live_v2_spec(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = LIVE_DEFAULT_SEEDS,
    n_examples: int = LIVE_DEFAULT_HALLU_N,
) -> P6ASpecBundle:
    """HaluEval-QA v2 live bundle at pre-registered N=15."""
    return P6ASpecBundle(
        label="H_HEQA_live_v2",
        spec=HypothesisSpec(
            hypothesis_id="H_HEQA_v2",
            description=(
                "[LIVE-HF] Harness abstention scaffold lifts HaluEval-QA "
                "closed-book accuracy on the real qa split. "
                "Pre-registered N=15, seeds=(0,1), no peeking."
            ),
            adapter_name="halu_eval_qa",
            treatment_condition="harness_on",
            baseline_condition="harness_off",
            metric="accuracy",
            direction=TargetDirection.GREATER,
            delta=0.05,
            n_examples=n_examples,
            seeds=seeds,
            models=models,
            significance_alpha=0.05,
            notes="load_real=True, dataset=pminervini/HaluEval, config=qa.",
        ),
        adapter_kwargs={"default_n": n_examples, "load_real": True, "strict_hf": True},
    )


def hallu_live_specs(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = LIVE_DEFAULT_SEEDS,
    n_examples: int = LIVE_DEFAULT_HALLU_N,
) -> list[P6ASpecBundle]:
    """Hallucination live bundles (v2 pre-registered).

    FACTS-Grounding is intentionally not included: the public release
    ships no gold answers; our substring-match scorer degenerates to
    0.0 accuracy, a pure token burn. HaluEval-Discriminate is the
    binary variant of HEQA and is redundant for the paper claim.
    """
    return [
        h_tqa_live_v2_spec(models=models, seeds=seeds, n_examples=n_examples),
        h_heqa_live_v2_spec(models=models, seeds=seeds, n_examples=n_examples),
    ]


# --- SWE-bench Verified (L3 live supplement) -------------------------


def h_swebench_verified_live_n15_spec(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = LIVE_DEFAULT_SEEDS,
    n_examples: int = LIVE_DEFAULT_SWEB_N,
) -> P6ASpecBundle:
    """SWE-bench Verified live supplement at N=15.

    Uses the heuristic (file_overlap + line_jaccard) scorer from the
    adapter — NOT the docker-backed harness. Publishing guidance: this
    is reported as a live *supplement* alongside the deterministic
    simulator in paper §10; it is not a leaderboard entry.
    """
    return P6ASpecBundle(
        label="H_SWEB_live_n15",
        spec=HypothesisSpec(
            hypothesis_id="H_SWEB",
            description=(
                "[LIVE-HF] Plan-then-act harness discipline lifts the heuristic "
                "patch-quality score on SWE-bench Verified (N=15 instances, "
                "file_overlap + line_jaccard scorer); real "
                "princeton-nlp/SWE-bench_Verified. "
                "Pre-registered N=15, seeds=(0,1), no peeking."
            ),
            adapter_name="swe_bench_verified",
            treatment_condition="harness_on",
            baseline_condition="harness_off",
            metric="score",
            direction=TargetDirection.GREATER,
            delta=0.05,
            n_examples=n_examples,
            seeds=seeds,
            models=models,
            significance_alpha=0.05,
            notes=(
                "load_real=True, dataset=princeton-nlp/SWE-bench_Verified, "
                "split=test. Heuristic offline scorer (file_overlap_weight=0.5, "
                "line_jaccard_weight=0.5, correctness_threshold=0.5). The "
                "docker-backed harness is NOT invoked in this bundle — see "
                "SWEBenchVerifiedAdapter.verify_with_swebench_harness for the "
                "published-grade pipeline."
            ),
        ),
        adapter_kwargs={"default_n": n_examples, "load_real": True, "strict_hf": True},
    )


# --- Aggregated scopes -----------------------------------------------


def ruler_live_specs(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = LIVE_DEFAULT_SEEDS,
    n_examples: int = LIVE_DEFAULT_RULER_N,
    target_tokens_tiers: tuple[int, ...] = LIVE_DEFAULT_TIERS,
) -> list[P6ASpecBundle]:
    """RULER single + multi bundles combined, all tiers."""
    return (
        h1_ruler_live_specs(
            models=models,
            seeds=seeds,
            n_examples=n_examples,
            target_tokens_tiers=target_tokens_tiers,
        )
        + h1b_ruler_multi_live_specs(
            models=models,
            seeds=seeds,
            n_examples=n_examples,
            target_tokens_tiers=target_tokens_tiers,
        )
    )


def core4_specs(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = LIVE_DEFAULT_SEEDS,
    ruler_n: int = LIVE_DEFAULT_RULER_N,
    hallu_n: int = LIVE_DEFAULT_HALLU_N,
    sweb_n: int = LIVE_DEFAULT_SWEB_N,
    target_tokens_tiers: tuple[int, ...] = LIVE_DEFAULT_TIERS,
) -> list[P6ASpecBundle]:
    """Minimum-viable 4-bundle live battery for paper §7/§8/§10 headline
    claims:

    * H1_ruler_8192_live + H1_ruler_16384_live — long-context lost-in-
      the-middle slope (one tier alone proves the *effect*; two tiers
      prove the *slope*).
    * H_TQA_live_v2 — hallucination / calibration claim.
    * H_SWEB_live_n15 — coding / agentic claim.

    Drops H1b_ruler_multi_{8K,16K} (variant of H1) and H_HEQA_live_v2
    (variant of H_TQA) from the full battery to keep one Anthropic
    rolling-window's worth of budget sufficient for completion.
    Pre-registration locks N, seeds, models, and tiers identically to
    full_battery; only the bundle *count* shrinks.
    """
    return [
        *h1_ruler_live_specs(
            models=models,
            seeds=seeds,
            n_examples=ruler_n,
            target_tokens_tiers=target_tokens_tiers,
        ),
        h_tqa_live_v2_spec(models=models, seeds=seeds, n_examples=hallu_n),
        h_swebench_verified_live_n15_spec(
            models=models, seeds=seeds, n_examples=sweb_n
        ),
    ]


def full_battery_specs(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = LIVE_DEFAULT_SEEDS,
    ruler_n: int = LIVE_DEFAULT_RULER_N,
    hallu_n: int = LIVE_DEFAULT_HALLU_N,
    sweb_n: int = LIVE_DEFAULT_SWEB_N,
    target_tokens_tiers: tuple[int, ...] = LIVE_DEFAULT_TIERS,
) -> list[P6ASpecBundle]:
    """The locked 7-bundle live-HF battery.

    Execution order is deliberate: cheapest-first (RULER single 8K) →
    most expensive (SWE-bench). That way if we run out of budget in
    the final window, the earlier bundles are already complete and
    the paper's H1 and hallu sections can still update, even if §10
    SWE-bench has to publish mock numbers.
    """
    return [
        *h1_ruler_live_specs(
            models=models,
            seeds=seeds,
            n_examples=ruler_n,
            target_tokens_tiers=target_tokens_tiers,
        ),
        *h1b_ruler_multi_live_specs(
            models=models,
            seeds=seeds,
            n_examples=ruler_n,
            target_tokens_tiers=target_tokens_tiers,
        ),
        h_tqa_live_v2_spec(models=models, seeds=seeds, n_examples=hallu_n),
        h_heqa_live_v2_spec(models=models, seeds=seeds, n_examples=hallu_n),
        h_swebench_verified_live_n15_spec(
            models=models, seeds=seeds, n_examples=sweb_n
        ),
    ]


# --- Backwards-compat aliases ----------------------------------------
# Preserved for the existing run_live_hf scope="ruler" / scope="hallu"
# paths. New code should call the v2 / H1b names.

h2_ruler_multi_live_specs = h1b_ruler_multi_live_specs
h_truthful_qa_live_spec = h_tqa_live_v2_spec
h_halu_eval_qa_live_spec = h_heqa_live_v2_spec
