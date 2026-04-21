"""RULER adapter — needle-in-a-haystack at configurable token budgets.

Faithful to the NIAH family from RULER (Hsieh et al., NVIDIA, 2024). The
adapter ships with a tokenizer-exact synthetic generator that produces the
single- and multi-needle variants at any token budget; this is what the
harness uses for CI, smoke tests, and the plugin's offline mode. The real
HF dataset (`simonjegou/ruler`) can be wired in later via `load_real=True`
without changing the public surface.

Two conditions:
  - "harness_off" (baseline): the model receives the full haystack inline.
  - "harness_on"  (treatment): the model receives a structured prompt that
    explicitly instructs it to scan for the requested vault key — this is
    what a context-engineering harness does in practice.

Hypothesis H1 / H2 use the (treatment - baseline) accuracy delta on this
adapter at each tested token budget tier (8K, 16K, 32K, 64K, 128K).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.benchmarks.base import BenchmarkAdapter, BenchmarkExample, ModelOutput
from src.benchmarks.registry import register

from ._hf_loader import HFUnavailable, load_hf_examples
from ._real_corpus import (
    RealCorpus,
    RealCorpusUnavailable,
    build_real_haystack,
    make_real_corpus,
)
from ._synthetic import generate_examples, make_needle

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


@register
@dataclass
class RulerNIAHAdapter(BenchmarkAdapter):
    """RULER single-needle NIAH adapter.

    Use `load_real=True` to fetch from `simonjegou/ruler` on HuggingFace
    (Apache-2.0). Falls back to the synthetic generator when `datasets`
    is not installed, the network is offline, or `CEH_DISABLE_HF=1` is set
    — emitting a warning so the source-of-data is never ambiguous.
    """

    name: str = "ruler_niah"
    requires_long_context: bool = True
    license_note: str = "synthetic generator; real RULER (Apache-2.0) loaded via HF when enabled"

    target_tokens: int = 8_192
    needles_per_example: int = 1
    default_n: int = 50

    load_real: bool = False
    # When True, an HF load failure is an error, not a silent synthetic
    # fallback. Live-HF bundles set this so a broken dataset or a wrong
    # task filter fails the bundle rather than quietly publishing
    # synthetic results under `provenance.source = "huggingface"`.
    strict_hf: bool = False
    hf_dataset_id: str = "simonjegou/ruler"
    hf_split: str = "test"
    hf_task_filter: str = "niah_single_1"

    use_real_corpus: bool = False
    real_corpus_sources: tuple[str, ...] = ("wikipedia", "arxiv")

    def load_examples(self, *, n: int | None = None, seed: int = 0) -> list[BenchmarkExample]:
        n_use = n or self.default_n
        if self.load_real:
            try:
                return self._load_from_hf(n=n_use, seed=seed)
            except HFUnavailable as exc:
                if self.strict_hf:
                    raise RuntimeError(
                        f"RULER live loader failed with strict_hf=True at "
                        f"{self.target_tokens} tokens: {exc}. Refusing to "
                        "fall back to synthetic under load_real=True."
                    ) from exc
                logger.warning(
                    "RULER real loader unavailable (%s); falling back to synthetic at %d tokens",
                    exc,
                    self.target_tokens,
                )
        if self.use_real_corpus:
            try:
                return self._load_synthetic_with_real_corpus(n=n_use, seed=seed)
            except RealCorpusUnavailable as exc:
                logger.warning(
                    "RULER real-corpus distractors unavailable (%s); falling back to template filler",
                    exc,
                )
        return self._load_synthetic(n=n_use, seed=seed)

    def _load_synthetic_with_real_corpus(self, *, n: int, seed: int) -> list[BenchmarkExample]:
        """Build NIAH examples with real Wikipedia/arXiv distractors.

        Uses the same needle structure as the pure-template path so the
        scorers and prompts work unchanged — only the distractor content
        changes.
        """
        corpus: RealCorpus = make_real_corpus(self.real_corpus_sources)
        examples: list[BenchmarkExample] = []
        for i in range(n):
            seed_i = seed * 10_000 + i
            needles = [make_needle(seed_i, k) for k in range(self.needles_per_example)]
            haystack = build_real_haystack(
                target_tokens=self.target_tokens,
                needles=needles,
                seed=seed_i,
                corpus=corpus,
            )
            primary = needles[0]
            ground_truth = (
                primary.value
                if self.needles_per_example == 1
                else {n.key: n.value for n in needles}
            )
            examples.append(
                BenchmarkExample(
                    id=f"real-{seed:02d}-{i:04d}",
                    prompt=primary.key,
                    ground_truth=ground_truth,
                    metadata={
                        "target_tokens": self.target_tokens,
                        "needle_count": len(needles),
                        "source": "synthetic+real_corpus",
                        "real_corpus_sources": list(self.real_corpus_sources),
                    },
                    context=haystack,
                )
            )
        return examples

    def _load_synthetic(self, *, n: int, seed: int) -> list[BenchmarkExample]:
        synthetic = generate_examples(
            n=n,
            seed=seed,
            target_tokens=self.target_tokens,
            needles_per_example=self.needles_per_example,
        )
        examples: list[BenchmarkExample] = []
        for s in synthetic:
            primary = s.needles[0]
            ground_truth = (
                primary.value
                if self.needles_per_example == 1
                else {n.key: n.value for n in s.needles}
            )
            examples.append(
                BenchmarkExample(
                    id=s.id,
                    prompt=primary.key,
                    ground_truth=ground_truth,
                    metadata={
                        "target_tokens": s.target_tokens,
                        "needle_count": len(s.needles),
                        "source": "synthetic",
                    },
                    context=s.haystack,
                )
            )
        return examples

    def _load_from_hf(self, *, n: int, seed: int) -> list[BenchmarkExample]:
        config = self._hf_config_for_budget()
        rows = load_hf_examples(
            dataset_id=self.hf_dataset_id,
            split=self.hf_split,
            config=config,
            n=None,
            seed=seed,
        )
        wanted_task = self.hf_task_filter
        filtered = [r for r in rows if r.get("task") == wanted_task] or rows
        if n is not None:
            import random as _random

            rng = _random.Random(seed)
            rng.shuffle(filtered)
            filtered = filtered[:n]
        examples: list[BenchmarkExample] = []
        for i, row in enumerate(filtered):
            haystack = row.get("input") or row.get("context") or ""
            outputs = row.get("outputs") or row.get("answer") or row.get("answers") or []
            if isinstance(outputs, str):
                outputs = [outputs]
            if not outputs:
                continue
            gt = outputs[0]
            examples.append(
                BenchmarkExample(
                    id=f"ruler-real-{config}-{i:04d}",
                    prompt=row.get("query") or "the requested needle",
                    ground_truth=gt,
                    metadata={
                        "target_tokens": self.target_tokens,
                        "needle_count": 1,
                        "source": "huggingface",
                        "hf_dataset_id": self.hf_dataset_id,
                        "hf_config": config,
                        "hf_task": row.get("task"),
                    },
                    context=haystack,
                )
            )
        if not examples:
            raise HFUnavailable(
                f"RULER HF rows for config={config} task={wanted_task} produced 0 usable examples"
            )
        return examples

    def _hf_config_for_budget(self) -> str:
        """Map self.target_tokens to the closest published RULER config."""
        published = (4096, 8192, 16384, 32768, 65536, 131072)
        target = min(published, key=lambda x: abs(x - self.target_tokens))
        return str(target)

    def render_prompt(self, example: BenchmarkExample, *, condition: str) -> str:
        if condition == "harness_on":
            return (
                f"You are a focused long-context retriever. "
                f"Skim the document below and report ONLY the activation code for vault {example.prompt}. "
                f"Reply with the bare activation code exactly as it appears (digits and/or letters, "
                f"any length), with no surrounding prose.\n\n"
                f"=== DOCUMENT ===\n{example.context}\n=== END DOCUMENT ===\n\n"
                f"What is the activation code for vault {example.prompt}?"
            )
        return (
            f"{example.context}\n\n"
            f"Question: What is the activation code for vault {example.prompt}?\n"
            f"Answer:"
        )

    def system_prompt(self, *, condition: str) -> str:
        if condition == "harness_on":
            return (
                "You answer with the single requested code only, "
                "verbatim and unmodified. Do not truncate, pad, reformat, "
                "summarize, or explain your reasoning."
            )
        return ""

    def score(self, example: BenchmarkExample, output: ModelOutput) -> tuple[float, bool, str]:
        if isinstance(example.ground_truth, dict):
            return self._score_multi(example, output)
        return self._score_single(example.ground_truth, output)

    @staticmethod
    def _score_single(gt: str, output: ModelOutput) -> tuple[float, bool, str]:
        norm_pred = _normalize(output.text)
        norm_gt = _normalize(gt)
        ok = norm_gt in norm_pred
        return (1.0 if ok else 0.0), ok, output.text.strip()

    @staticmethod
    def _score_multi(example: BenchmarkExample, output: ModelOutput) -> tuple[float, bool, str]:
        norm_pred = _normalize(output.text)
        gt = example.ground_truth
        assert isinstance(gt, dict)
        hits = sum(1 for v in gt.values() if _normalize(v) in norm_pred)
        frac = hits / len(gt) if gt else 0.0
        return frac, frac == 1.0, output.text.strip()


@register
@dataclass
class RulerNIAHMultiAdapter(RulerNIAHAdapter):
    """Multi-key NIAH variant — tests retrieval over 4 simultaneous needles."""
    name: str = "ruler_niah_multi"
    needles_per_example: int = 4
