"""HypothesisSpec definitions for P6-A re-runs of H1 and H2.

H1 (long-context variant): "The Avacchedaka harness lifts NIAH accuracy on
RULER under fixed token budget, multi-seed, multi-model."

H2 (long-context variant): "The Avacchedaka harness lifts multi-key recall
on HELMET-recall under fixed token budget."

Both hypotheses use the *same* treatment / baseline labels recognised by
the long-context adapter family ("harness_on" / "harness_off") so swapping
in real CLI traffic does not require renaming conditions.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.benchmarks.hypothesis import HypothesisSpec, TargetDirection


# Models we re-run. Haiku is the cheaper, lower-context model; Sonnet is
# the workhorse. Running both lets us report a per-model breakdown and a
# joint paired test in the paper. Both names match the production CLI.
DEFAULT_MODELS: tuple[str, ...] = ("claude-haiku-4-5", "claude-sonnet-4-6")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class P6ASpecBundle:
    label: str
    spec: HypothesisSpec
    adapter_kwargs: dict


def h1_specs(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_examples: int = 30,
    target_tokens_tiers: tuple[int, ...] = (8_192, 32_768),
) -> list[P6ASpecBundle]:
    """One spec per token-budget tier; we want to see the H1 lift at every depth."""
    out: list[P6ASpecBundle] = []
    for tokens in target_tokens_tiers:
        out.append(
            P6ASpecBundle(
                label=f"H1_ruler_{tokens}",
                spec=HypothesisSpec(
                    hypothesis_id="H1",
                    description=(
                        "Avacchedaka structured retriever prompt lifts RULER NIAH "
                        f"accuracy at {tokens}-token contexts vs the unstructured "
                        "baseline; effect ≥ 5 pts."
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
                        f"Tier={tokens} tokens. Adapter is the synthetic NIAH; "
                        "the live re-run path will swap in load_real=True."
                    ),
                ),
                adapter_kwargs={"target_tokens": tokens, "default_n": n_examples},
            )
        )
    return out


def h2_specs(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_examples: int = 20,
    target_tokens_tiers: tuple[int, ...] = (8_192, 32_768),
) -> list[P6ASpecBundle]:
    """H2 long-context analogue: multi-key recall on HELMET-recall.

    The 'harness_on' condition prepends a structured 'list ALL codes' instruction;
    the 'harness_off' condition just appends the question. We expect a sharp
    lift in joint multi-key recall — this is exactly what the Avacchedaka
    typed-storage interface buys you in a live agent loop.
    """
    out: list[P6ASpecBundle] = []
    for tokens in target_tokens_tiers:
        out.append(
            P6ASpecBundle(
                label=f"H2_helmet_recall_{tokens}",
                spec=HypothesisSpec(
                    hypothesis_id="H2",
                    description=(
                        "Avacchedaka structured-recall prompt lifts HELMET-recall "
                        f"joint accuracy (k=5) at {tokens}-token contexts vs "
                        "unstructured baseline; effect ≥ 10 pts."
                    ),
                    adapter_name="helmet_recall",
                    treatment_condition="harness_on",
                    baseline_condition="harness_off",
                    metric="score",  # score = fraction of needles recovered
                    direction=TargetDirection.GREATER,
                    delta=0.10,
                    n_examples=n_examples,
                    seeds=seeds,
                    models=models,
                    significance_alpha=0.05,
                    notes=f"Tier={tokens} tokens, k=5 needles per example.",
                ),
                adapter_kwargs={
                    "target_tokens": tokens,
                    "default_n": n_examples,
                    "needles_per_example": 5,
                },
            )
        )
    return out


def all_specs(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[P6ASpecBundle]:
    return h1_specs(models=models, seeds=seeds) + h2_specs(models=models, seeds=seeds)
