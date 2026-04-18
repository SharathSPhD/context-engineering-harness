"""Bayesian Beta-posterior aggregation for conflicting-source RAG.

Replaces the original :class:`PrecisionWeightedRAG` (top-k by precision)
with a calibrated Beta-Bernoulli model over candidate answers.

Model
-----
Each retrieved source carries a precision ``p ∈ [0, 1]`` interpreted as
the source's reliability when it asserts an answer. We treat the question
as having a (possibly unknown) set of candidate answers proposed by the
sources. For each candidate answer ``a``, we maintain a Beta posterior
``Beta(α_a, β_a)`` over the probability that ``a`` is the correct
answer, starting from an uninformative ``Beta(1, 1)`` prior.

For every source ``s`` with precision ``p_s``:

* If ``s`` asserts ``a``  → evidence *for* ``a`` of weight ``p_s``,
  evidence *against* ``a`` of weight ``1 - p_s``.
* If ``s`` asserts ``a' ≠ a`` → in a binary world the source is
  effectively voting *against* ``a`` with weight ``p_s``. In the general
  multi-answer world we model it as evidence *against* ``a`` of weight
  ``p_s`` and a small mass ``(1 - p_s) / (K - 1)`` of evidence *for*
  ``a`` (the source is wrong with probability ``1 - p_s`` and any of
  the other ``K - 1`` candidates could be the truth, ``a`` included).

Each evidence increment is scaled by ``evidence_strength`` (pseudo-count
strength) to control how strongly precision translates into Beta
updates. Higher values produce sharper posteriors.

Outputs
-------
* :meth:`posteriors` returns the (α, β) pair per candidate answer.
* :meth:`predict` returns ``(best_answer, posterior_mean)`` for the
  argmax candidate. The posterior mean is a calibrated probability and
  feeds directly into :mod:`src.calibration.metrics` (Brier, ECE).
* :meth:`detect_conflict` flags low-confidence cases by looking at the
  posterior mean *margin* between the top-2 candidates.

The class also preserves the surface used by H2 (``select_sources`` and
``build_prompt``) so it is a drop-in for :class:`PrecisionWeightedRAG`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class _Posterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1.0))


class BayesianBetaRAG:
    """Beta-Bernoulli aggregator over candidate answers from noisy sources."""

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        evidence_strength: float = 4.0,
        conflict_margin: float = 0.15,
    ) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be positive")
        if evidence_strength <= 0:
            raise ValueError("evidence_strength must be positive")
        if not 0.0 <= conflict_margin <= 1.0:
            raise ValueError("conflict_margin must be in [0, 1]")
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.evidence_strength = float(evidence_strength)
        self.conflict_margin = float(conflict_margin)

    def posteriors(self, sources: list[dict]) -> "OrderedDict[str, _Posterior]":
        """Return per-candidate-answer Beta posterior, in stable insertion order."""
        if not sources:
            return OrderedDict()

        candidates: list[str] = []
        seen: set[str] = set()
        for s in sources:
            ans = s.get("answer")
            if ans is None or ans in seen:
                continue
            seen.add(ans)
            candidates.append(ans)
        k = len(candidates)
        if k == 0:
            return OrderedDict()

        alphas: dict[str, float] = {a: self.prior_alpha for a in candidates}
        betas: dict[str, float] = {a: self.prior_beta for a in candidates}
        s_weight = self.evidence_strength

        for s in sources:
            voter_answer = s.get("answer")
            if voter_answer is None:
                continue
            p = float(s.get("precision", 0.5))
            p = max(0.0, min(1.0, p))
            for a in candidates:
                if a == voter_answer:
                    alphas[a] += p * s_weight
                    betas[a] += (1.0 - p) * s_weight
                else:
                    betas[a] += p * s_weight
                    if k > 1:
                        alphas[a] += (1.0 - p) * s_weight / (k - 1)

        return OrderedDict((a, _Posterior(alphas[a], betas[a])) for a in candidates)

    def predict(self, sources: list[dict]) -> tuple[str | None, float]:
        """Return ``(argmax_answer, posterior_mean)``.

        ``argmax_answer`` is ``None`` when no candidate could be formed.
        The returned probability is the calibrated posterior mean of the
        winning answer, suitable for Brier/ECE scoring.
        """
        post = self.posteriors(sources)
        if not post:
            return None, 0.0
        best_answer = max(post.items(), key=lambda kv: kv[1].mean)
        return best_answer[0], best_answer[1].mean

    def detect_conflict(self, sources: list[dict]) -> bool:
        """True when the posterior margin between top-2 candidates is small.

        Replaces the legacy "answers-differ + precision-gap < 0.3" rule
        with a principled check on the calibrated posterior means.
        """
        post = self.posteriors(sources)
        if len(post) < 2:
            return False
        means = sorted((p.mean for p in post.values()), reverse=True)
        return (means[0] - means[1]) < self.conflict_margin

    def select_sources(self, sources: list[dict], top_k: int = 3) -> list[dict]:
        """Return ``top_k`` sources by precision (legacy surface)."""
        return sorted(sources, key=lambda s: s.get("precision", 0.5), reverse=True)[:top_k]

    def build_prompt(self, question: str, sources: list[dict]) -> str:
        """Compose a prompt with calibrated conflict signalling."""
        selected = self.select_sources(sources)
        conflict = self.detect_conflict(sources)
        source_text = "\n".join(
            f"[Source precision={s['precision']:.2f}] {s['content']}" for s in selected
        )
        if conflict:
            best, prob = self.predict(sources)
            conflict_note = (
                f"\nNote: Sources conflict. Posterior favors '{best}' with "
                f"probability {prob:.2f}; express calibrated uncertainty."
            )
        else:
            conflict_note = ""
        return f"Sources:\n{source_text}{conflict_note}\n\nQuestion: {question}\nAnswer:"
