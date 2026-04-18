"""HELMET adapter — long-context multi-task aggregator.

HELMET (Princeton, 2024) aggregates 7 long-context task families. The v2
harness ships first-class adapters for the two families that drive H1 and
H2 most directly:

  * helmet_rag — retrieval-augmented QA over a long document. Treatment
    condition prepends a structured "use only the document below"
    instruction; baseline condition appends a naive question.
  * helmet_recall — multi-key recall (HELMET's RAG/recall variant where
    several discrete facts must be returned together).

The other HELMET families (summarization w/ citations, re-ranking,
generation w/ citations, ICL, code completion) are stubs that raise
NotImplementedError so we never silently report a fake number for them.
The real HF dataset (`princeton-nlp/HELMET`) will be wired into
`load_examples(load_real=True)` in P3 once the synthetic story is
locked down.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.benchmarks.base import BenchmarkAdapter, BenchmarkExample, ModelOutput
from src.benchmarks.registry import register

from ._hf_loader import HFUnavailable
from ._synthetic import generate_examples

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _hf_unwired(name: str) -> HFUnavailable:
    """HELMET configs are wired in P3 once per-task schemas are locked down."""
    return HFUnavailable(
        f"{name}: HuggingFace loader for HELMET is wired in P3 (p3_longctx); "
        "use load_real=False for synthetic until then"
    )


@register
@dataclass
class HelmetRagAdapter(BenchmarkAdapter):
    """HELMET RAG-style QA over a long synthetic document."""

    name: str = "helmet_rag"
    requires_long_context: bool = True
    license_note: str = "synthetic; real HELMET (MIT-licensed) loaded via HF when enabled"

    target_tokens: int = 8_192
    default_n: int = 30
    load_real: bool = False

    def load_examples(self, *, n: int | None = None, seed: int = 0) -> list[BenchmarkExample]:
        n_use = n or self.default_n
        if self.load_real:
            try:
                raise _hf_unwired("helmet_rag")
            except HFUnavailable as exc:
                logger.warning("HELMET RAG real loader unavailable (%s); using synthetic", exc)
        synth = generate_examples(
            n=n_use,
            seed=seed,
            target_tokens=self.target_tokens,
            needles_per_example=1,
        )
        return [
            BenchmarkExample(
                id=f"helmet-rag-{s.id}",
                prompt=s.needles[0].key,
                ground_truth=s.needles[0].value,
                context=s.haystack,
                metadata={"target_tokens": s.target_tokens, "source": "synthetic"},
            )
            for s in synth
        ]

    def render_prompt(self, example: BenchmarkExample, *, condition: str) -> str:
        if condition == "harness_on":
            return (
                "You are a retrieval-augmented assistant. The DOCUMENT below "
                "contains the answer. Quote the exact 6-character activation code.\n\n"
                f"DOCUMENT:\n{example.context}\n\n"
                f"QUESTION: What is the activation code for vault {example.prompt}?\n"
                "ANSWER (code only):"
            )
        return (
            f"{example.context}\n\n"
            f"What is the activation code for vault {example.prompt}?"
        )

    def score(self, example: BenchmarkExample, output: ModelOutput) -> tuple[float, bool, str]:
        gt = str(example.ground_truth).upper()
        pred = re.sub(r"\s+", "", output.text).upper()
        ok = gt in pred
        return (1.0 if ok else 0.0), ok, output.text.strip()


@register
@dataclass
class HelmetRecallAdapter(BenchmarkAdapter):
    """HELMET recall-style task — return all keys at once."""

    name: str = "helmet_recall"
    requires_long_context: bool = True
    license_note: str = "synthetic; HELMET MIT-licensed real data via HF when enabled"

    target_tokens: int = 8_192
    needles_per_example: int = 5
    default_n: int = 20
    load_real: bool = False

    def load_examples(self, *, n: int | None = None, seed: int = 0) -> list[BenchmarkExample]:
        n_use = n or self.default_n
        if self.load_real:
            try:
                raise _hf_unwired("helmet_recall")
            except HFUnavailable as exc:
                logger.warning("HELMET Recall real loader unavailable (%s); using synthetic", exc)
        synth = generate_examples(
            n=n_use,
            seed=seed,
            target_tokens=self.target_tokens,
            needles_per_example=self.needles_per_example,
        )
        return [
            BenchmarkExample(
                id=f"helmet-recall-{s.id}",
                prompt="ALL_VAULTS",
                ground_truth={n.key: n.value for n in s.needles},
                context=s.haystack,
                metadata={
                    "target_tokens": s.target_tokens,
                    "k": self.needles_per_example,
                    "source": "synthetic",
                },
            )
            for s in synth
        ]

    def render_prompt(self, example: BenchmarkExample, *, condition: str) -> str:
        gt = example.ground_truth
        assert isinstance(gt, dict)
        keys = ", ".join(gt.keys())
        if condition == "harness_on":
            return (
                "You are a structured retriever. List ALL activation codes "
                "for the requested vaults. Reply with one line per vault in "
                "the format `<vault-id>=<code>`.\n\n"
                f"DOCUMENT:\n{example.context}\n\n"
                f"REQUESTED VAULTS: {keys}\n\n"
                "ANSWERS:"
            )
        return (
            f"{example.context}\n\n"
            f"List the activation codes for these vaults: {keys}"
        )

    def score(self, example: BenchmarkExample, output: ModelOutput) -> tuple[float, bool, str]:
        gt = example.ground_truth
        assert isinstance(gt, dict)
        text = output.text.upper().replace(" ", "")
        hits = sum(1 for v in gt.values() if v.upper() in text)
        frac = hits / len(gt) if gt else 0.0
        return frac, frac == 1.0, output.text.strip()
