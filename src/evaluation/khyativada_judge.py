"""Deterministic LLM-as-judge surrogate for the Khyātivāda taxonomy.

In a production paper run the second annotator is the ``FewShotKhyativadaClassifier``
calling Claude through ``ClaudeCLIClient``. That path is real but
network-dependent and gated by the budget scheduler.

For CI, smoke tests, and offline reproductions we need a *second*
annotator that:

* Is independent of the heuristic in ``KhyativadaClassifier`` (so κ
  measures something).
* Is realistic: it picks the gold label most of the time, but with
  systematic confusions that resemble what a real LLM judge produces
  on this taxonomy (anyathakhyati ↔ akhyati, atmakhyati ↔ none, etc.).
* Is fully deterministic for a given seed so kappa numbers reproduce
  exactly between runs.

The simulator is honest about what it is: every prediction carries
``source="simulated_judge"`` so the resulting agreement report cannot
be confused with a real LLM judge run.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass

from src.evaluation.khyativada_corpus import CorpusRow


@dataclass(frozen=True)
class JudgePrediction:
    """A single annotator output."""

    item_id: str
    label: str
    confidence: float
    source: str


# Realistic confusion model: which other labels does each gold class
# get conflated with by a real LLM judge using the same 6+1 taxonomy?
# Probabilities are conditional on a "wrong" outcome — the marginal
# wrong-rate is controlled by ``error_rate``.
_CONFUSION_MODEL: dict[str, list[tuple[str, float]]] = {
    "anyathakhyati": [("akhyati", 0.55), ("viparitakhyati", 0.30), ("none", 0.15)],
    "atmakhyati": [("none", 0.55), ("anirvacaniyakhyati", 0.30), ("anyathakhyati", 0.15)],
    "anirvacaniyakhyati": [("asatkhyati", 0.55), ("atmakhyati", 0.30), ("none", 0.15)],
    "asatkhyati": [("anirvacaniyakhyati", 0.55), ("atmakhyati", 0.30), ("none", 0.15)],
    "viparitakhyati": [("anyathakhyati", 0.60), ("akhyati", 0.30), ("none", 0.10)],
    "akhyati": [("anyathakhyati", 0.55), ("viparitakhyati", 0.25), ("none", 0.20)],
    "none": [("atmakhyati", 0.45), ("anyathakhyati", 0.30), ("akhyati", 0.25)],
}


def _draw_wrong_label(rng: random.Random, gold_label: str) -> str:
    """Sample a wrong label using the confusion-model distribution."""
    table = _CONFUSION_MODEL.get(gold_label)
    if not table:
        return "none" if gold_label != "none" else "anyathakhyati"
    weights = [w for _, w in table]
    return rng.choices([lab for lab, _ in table], weights=weights)[0]


def simulate_judge(
    rows: Iterable[CorpusRow],
    *,
    accuracy: float = 0.78,
    seed: int = 1,
) -> list[JudgePrediction]:
    """Simulate an LLM judge on the corpus rows.

    Args:
        rows: Corpus to annotate.
        accuracy: Probability the judge picks the gold label
            (default 0.78 — close to a competent few-shot Claude judge
            on the synthetic taxonomy, well above chance and below
            ceiling so κ remains meaningful).
        seed: Master seed for the per-row RNG.

    Returns:
        One :class:`JudgePrediction` per input row, in input order.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be in [0, 1]")

    out: list[JudgePrediction] = []
    for row in rows:
        row_seed = int(
            hashlib.sha256(f"judge-{seed}-{row.id}".encode()).hexdigest()[:8],
            16,
        )
        rng = random.Random(row_seed)
        if rng.random() < accuracy:
            label = row.gold_label
            conf = 0.75 + 0.20 * rng.random()
        else:
            label = _draw_wrong_label(rng, row.gold_label)
            conf = 0.55 + 0.20 * rng.random()
        out.append(JudgePrediction(item_id=row.id, label=label, confidence=conf, source="simulated_judge"))
    return out


__all__ = ["JudgePrediction", "simulate_judge"]
