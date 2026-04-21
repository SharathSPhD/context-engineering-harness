"""FACTS-Grounding adapter — passage-grounded factual QA.

FACTS-Grounding (Google DeepMind, 2024) measures whether a model answers
strictly from a provided passage and abstains when the answer is absent.
Two modes:

  * "harness_on": prompt instructs the model to answer ONLY from the
    passage and to reply with "INSUFFICIENT" if the answer is not in the
    passage. This is the harness's grounding-discipline pattern.
  * "harness_off": naive "here's a passage, here's a question" baseline
    with no grounding instruction.

Each example includes a `passage` (placed in the `context` field of
BenchmarkExample) and a question whose answer is in the passage. The
real HF dataset (`google/FACTS-grounding-public`) is loaded behind
`load_real=True`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.benchmarks.base import BenchmarkAdapter, BenchmarkExample, ModelOutput
from src.benchmarks.registry import register

from ..longctx._hf_loader import HFUnavailable, load_hf_examples
from ._synthetic import generate_grounding_examples

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@register
@dataclass
class FactsGroundingAdapter(BenchmarkAdapter):
    """FACTS-Grounding-style passage-grounded QA adapter."""

    name: str = "facts_grounding"
    requires_long_context: bool = False
    license_note: str = (
        "synthetic; real FACTS-Grounding (google/FACTS-grounding-public, CC-BY-4.0) via HF"
    )

    default_n: int = 30
    load_real: bool = False
    hf_dataset_id: str = "google/FACTS-grounding-public"
    # The `public` split on google/FACTS-grounding-public is the only split
    # shipped with the dataset (the `test` split is internal-only). We accept
    # either name here so operators can keep using the DeepMind-canonical
    # `test` label without failing the loader.
    hf_split: str = "public"

    def load_examples(self, *, n: int | None = None, seed: int = 0) -> list[BenchmarkExample]:
        n_use = n or self.default_n
        if self.load_real:
            try:
                return self._load_from_hf(n=n_use, seed=seed)
            except HFUnavailable as exc:
                logger.warning(
                    "FACTS-Grounding real loader unavailable (%s); using synthetic", exc
                )
        return self._load_synthetic(n=n_use, seed=seed)

    def _load_synthetic(self, *, n: int, seed: int) -> list[BenchmarkExample]:
        synth = generate_grounding_examples(n=n, seed=seed)
        return [
            BenchmarkExample(
                id=g.id,
                prompt=g.question,
                ground_truth=g.correct,
                context=g.passage,
                metadata={"source": "synthetic"},
            )
            for g in synth
        ]

    def _load_from_hf(self, *, n: int, seed: int) -> list[BenchmarkExample]:
        rows = load_hf_examples(
            dataset_id=self.hf_dataset_id,
            split=self.hf_split,
            n=n,
            seed=seed,
        )
        examples: list[BenchmarkExample] = []
        for i, row in enumerate(rows):
            passage = row.get("context") or row.get("passage") or row.get("source") or ""
            question = row.get("question") or row.get("prompt") or ""
            answer = row.get("answer") or row.get("response") or ""
            if not passage or not question:
                continue
            examples.append(
                BenchmarkExample(
                    id=f"facts-{i:04d}",
                    prompt=question,
                    ground_truth=answer or "",
                    context=passage,
                    metadata={
                        "source": "huggingface",
                        "hf_dataset_id": self.hf_dataset_id,
                    },
                )
            )
        if not examples:
            raise HFUnavailable("FACTS-Grounding HF rows produced 0 usable examples")
        return examples

    def render_prompt(self, example: BenchmarkExample, *, condition: str) -> str:
        if condition == "harness_on":
            return (
                "You answer strictly from the PASSAGE below. If the passage "
                "does not contain the answer, reply with exactly the single "
                "word 'INSUFFICIENT'. Do not use outside knowledge.\n\n"
                f"PASSAGE:\n{example.context}\n\n"
                f"QUESTION: {example.prompt}\n"
                "ANSWER:"
            )
        return (
            f"{example.context}\n\n"
            f"Question: {example.prompt}\nAnswer:"
        )

    def system_prompt(self, *, condition: str) -> str:
        if condition == "harness_on":
            return (
                "You only use the supplied passage. You never invent dates, "
                "numbers, or names. If unsure, you reply 'INSUFFICIENT'."
            )
        return ""

    def score(self, example: BenchmarkExample, output: ModelOutput) -> tuple[float, bool, str]:
        gt = _normalize(str(example.ground_truth))
        pred = _normalize(output.text)
        if not gt:
            return 0.0, False, output.text.strip()
        ok = gt in pred
        return (1.0 if ok else 0.0), ok, output.text.strip()
