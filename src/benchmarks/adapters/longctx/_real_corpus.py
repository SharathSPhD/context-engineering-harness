"""Real long-context distractor corpora.

Backs the "real" mode of the long-context adapters with token-exact
slices of Wikipedia and arXiv text instead of the
hand-written-template filler used by ``_synthetic.py``.

Why a separate module
---------------------
Real corpora make the haystack distribution close to what the model
will see in production (book-length narratives, technical articles,
mixed-domain noise) — something the synthetic templates cannot
emulate. But fetching real text:

* Costs network and disk on first load.
* Adds a hard ``datasets`` dependency.
* Cannot run inside CI without explicit opt-in.

This module therefore:

1. Is opt-in via the ``CEH_REAL_LONGCTX=1`` env var or the explicit
   ``use_real_corpus=True`` flag on the adapters / builders.
2. Caches all fetched chunks under
   ``$XDG_CACHE_HOME/ceh/longctx/`` (default
   ``~/.cache/ceh/longctx/``) keyed by ``(source, language, chunk_id)``
   so subsequent runs are deterministic and offline-able.
3. Raises :class:`RealCorpusUnavailable` (a subclass of
   :class:`HFUnavailable`) so callers can re-use the same fallback
   path they already use for HF datasets.

Public API
----------
* :class:`RealCorpus` — abstract protocol-like base.
* :class:`HFWikipediaCorpus` — Wikipedia article text via
  ``wikimedia/wikipedia``.
* :class:`HFArxivCorpus` — arXiv abstracts via
  ``ccdv/arxiv-summarization`` (paper-level abstracts, license-clean).
* :class:`MixedCorpus` — round-robins over multiple corpora.
* :func:`make_real_corpus` — factory that resolves the best available
  corpus given the environment and optional source list.
* :func:`build_real_haystack` — tokenizer-exact haystack assembler
  that drops needles into real distractor passages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from src.utils.tokenizer import count_tokens

from ._hf_loader import HFUnavailable
from ._synthetic import Needle

logger = logging.getLogger(__name__)


class RealCorpusUnavailable(HFUnavailable):
    """Raised when no real corpus can be loaded (env disabled, no datasets, no net)."""


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    p = Path(base) / "ceh" / "longctx"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_cache(key: str) -> list[str] | None:
    f = _cache_dir() / f"{key}.jsonl"
    if not f.exists():
        return None
    try:
        return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    except Exception as exc:
        logger.warning("Cache read failed for %s: %s", key, exc)
        return None


def _write_cache(key: str, chunks: list[str]) -> None:
    f = _cache_dir() / f"{key}.jsonl"
    try:
        f.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks))
    except Exception as exc:  # pragma: no cover — disk full / permissions
        logger.warning("Cache write failed for %s: %s", key, exc)


def _check_opt_in() -> None:
    """Refuse to fetch unless the user explicitly opted in."""
    if os.environ.get("CEH_DISABLE_HF") == "1":
        raise RealCorpusUnavailable("CEH_DISABLE_HF=1 set; refusing to fetch real corpus")
    if os.environ.get("CEH_REAL_LONGCTX") != "1":
        raise RealCorpusUnavailable(
            "real long-context corpora are opt-in; set CEH_REAL_LONGCTX=1 to enable"
        )


class RealCorpus:
    """Abstract base for a paragraph stream backed by real text."""

    name = "abstract"

    def passages(self, *, target_total_tokens: int, seed: int) -> Iterator[str]:
        """Yield passages until at least ``target_total_tokens`` have been emitted."""
        raise NotImplementedError


@dataclass
class HFWikipediaCorpus(RealCorpus):
    """Wikipedia article text via the ``wikimedia/wikipedia`` HF dataset.

    Uses the ``20231101.en`` snapshot by default (most recent license-clean
    English snapshot). Pulls the ``text`` field per article and slices it
    into ~paragraph chunks. Cached locally per-snapshot so subsequent runs
    do not re-stream the dataset.
    """

    name: str = "wikipedia"
    snapshot: str = "20231101.en"
    n_articles: int = 200
    chunk_target_tokens: int = 200

    def _cache_key(self) -> str:
        h = hashlib.sha1(f"{self.snapshot}-{self.n_articles}-{self.chunk_target_tokens}".encode()).hexdigest()[:10]
        return f"wikipedia-{self.snapshot}-{h}"

    def _load_chunks(self) -> list[str]:
        cached = _read_cache(self._cache_key())
        if cached is not None:
            return cached
        _check_opt_in()
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RealCorpusUnavailable("datasets not installed") from exc
        try:
            ds = load_dataset(
                "wikimedia/wikipedia",
                self.snapshot,
                split="train",
                streaming=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RealCorpusUnavailable(f"wikipedia load failed: {exc}") from exc

        chunks: list[str] = []
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            for para in self._split_to_chunks(text):
                chunks.append(para)
            if len(chunks) >= self.n_articles * 8:
                break
        if not chunks:
            raise RealCorpusUnavailable("wikipedia returned no usable chunks")
        _write_cache(self._cache_key(), chunks)
        return chunks

    def _split_to_chunks(self, text: str) -> Iterable[str]:
        """Split a long article into chunks of ~chunk_target_tokens tokens."""
        out: list[str] = []
        cur: list[str] = []
        cur_tokens = 0
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            tcount = count_tokens(para)
            if cur_tokens + tcount <= self.chunk_target_tokens:
                cur.append(para)
                cur_tokens += tcount
            else:
                if cur:
                    out.append("\n\n".join(cur))
                cur = [para]
                cur_tokens = tcount
        if cur:
            out.append("\n\n".join(cur))
        return out

    def passages(self, *, target_total_tokens: int, seed: int) -> Iterator[str]:
        chunks = self._load_chunks()
        rng = random.Random(seed)
        order = list(range(len(chunks)))
        rng.shuffle(order)
        emitted = 0
        for i in order:
            if emitted >= target_total_tokens:
                return
            yield chunks[i]
            emitted += count_tokens(chunks[i])


@dataclass
class HFArxivCorpus(RealCorpus):
    """arXiv abstracts via the ``ccdv/arxiv-summarization`` HF dataset.

    Abstracts are short (~200 tokens) and self-contained, which makes them
    excellent topic-shift distractors. The ``article`` field is used.
    """

    name: str = "arxiv"
    config: str = "section"
    split: str = "train"
    n_papers: int = 1500

    def _cache_key(self) -> str:
        return f"arxiv-{self.config}-{self.split}-{self.n_papers}"

    def _load_chunks(self) -> list[str]:
        cached = _read_cache(self._cache_key())
        if cached is not None:
            return cached
        _check_opt_in()
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RealCorpusUnavailable("datasets not installed") from exc
        try:
            ds = load_dataset(
                "ccdv/arxiv-summarization",
                self.config,
                split=self.split,
                streaming=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RealCorpusUnavailable(f"arxiv load failed: {exc}") from exc

        chunks: list[str] = []
        for row in ds:
            text = (row.get("abstract") or row.get("article") or "").strip()
            if not text or len(text) < 200:
                continue
            chunks.append(text)
            if len(chunks) >= self.n_papers:
                break
        if not chunks:
            raise RealCorpusUnavailable("arxiv returned no usable chunks")
        _write_cache(self._cache_key(), chunks)
        return chunks

    def passages(self, *, target_total_tokens: int, seed: int) -> Iterator[str]:
        chunks = self._load_chunks()
        rng = random.Random(seed)
        order = list(range(len(chunks)))
        rng.shuffle(order)
        emitted = 0
        for i in order:
            if emitted >= target_total_tokens:
                return
            yield chunks[i]
            emitted += count_tokens(chunks[i])


@dataclass
class MixedCorpus(RealCorpus):
    """Round-robin over multiple corpora to simulate domain-mixed noise."""

    name: str = "mixed"
    corpora: tuple[RealCorpus, ...] = ()

    def __post_init__(self) -> None:
        if not self.corpora:
            raise ValueError("MixedCorpus requires at least one underlying corpus")

    def passages(self, *, target_total_tokens: int, seed: int) -> Iterator[str]:
        per = max(1, target_total_tokens // len(self.corpora))
        iters = [
            iter(c.passages(target_total_tokens=per * 4, seed=seed + i))
            for i, c in enumerate(self.corpora)
        ]
        emitted = 0
        rr = 0
        active = list(range(len(iters)))
        while active and emitted < target_total_tokens:
            i = active[rr % len(active)]
            try:
                p = next(iters[i])
            except StopIteration:
                active.remove(i)
                continue
            yield p
            emitted += count_tokens(p)
            rr += 1


def make_real_corpus(
    sources: tuple[str, ...] = ("wikipedia", "arxiv"),
) -> RealCorpus:
    """Return the best available real corpus for the requested sources.

    Tries each source in order; returns the first that loads successfully.
    Returns a :class:`MixedCorpus` if more than one loads. Raises
    :class:`RealCorpusUnavailable` if none can.
    """
    available: list[RealCorpus] = []
    errors: list[str] = []
    for src in sources:
        if src == "wikipedia":
            corpus: RealCorpus = HFWikipediaCorpus()
        elif src == "arxiv":
            corpus = HFArxivCorpus()
        else:
            errors.append(f"unknown source {src!r}")
            continue
        try:
            corpus._load_chunks()  # type: ignore[attr-defined]
            available.append(corpus)
        except RealCorpusUnavailable as exc:
            errors.append(f"{src}: {exc}")
            continue
    if not available:
        raise RealCorpusUnavailable("; ".join(errors) or "no sources requested")
    if len(available) == 1:
        return available[0]
    return MixedCorpus(corpora=tuple(available))


def build_real_haystack(
    *,
    target_tokens: int,
    needles: list[Needle],
    seed: int,
    corpus: RealCorpus,
    encoding: str = "o200k_base",
) -> str:
    """Tokenizer-exact haystack assembled from ``corpus`` passages with needles inserted.

    Algorithm:

    1. Pre-fetch enough passages from ``corpus`` to cover ``target_tokens``.
    2. Compute deterministic insertion indices for each needle so that the
       distance between consecutive needles is roughly equal across runs.
    3. Emit passages in the corpus order, splicing needles at the right
       boundary positions, until the cumulative token count reaches
       ``target_tokens - 30``.
    4. Append any unconsumed needles at the end (defensive: this lets
       tiny ``target_tokens`` budgets still satisfy "every needle appears
       exactly once").

    The returned haystack token count is guaranteed not to exceed
    ``target_tokens`` by more than the size of one passage.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    if not needles:
        # No needles requested — trivial corpus dump
        passages = list(corpus.passages(target_total_tokens=target_tokens, seed=seed))
        return "\n\n".join(passages)

    passages = list(corpus.passages(target_total_tokens=target_tokens, seed=seed))
    if not passages:
        raise RealCorpusUnavailable("corpus produced no passages")

    rng = random.Random(seed * 31 + len(needles))
    n_passages = len(passages)
    if n_passages < len(needles):
        # Cycle so every needle still has a slot
        passages = (passages * ((len(needles) // n_passages) + 1))[: max(n_passages, len(needles) * 2)]
        n_passages = len(passages)

    spacing = max(1, n_passages // (len(needles) + 1))
    needle_positions = sorted(
        {min(n_passages - 1, max(0, (i + 1) * spacing + rng.randint(-spacing // 4, spacing // 4)))
         for i in range(len(needles))}
    )
    while len(needle_positions) < len(needles):
        # Resolve duplicates by nudging
        for i in range(len(needles)):
            cand = (i + 1) * spacing
            if cand not in needle_positions and cand < n_passages:
                needle_positions.append(cand)
                if len(needle_positions) == len(needles):
                    break
        else:
            break
    needle_positions = sorted(needle_positions[: len(needles)])

    out: list[str] = []
    cur_tokens = 0
    next_needle = 0
    for i, p in enumerate(passages):
        if cur_tokens >= target_tokens - 30:
            break
        out.append(p)
        cur_tokens += count_tokens(p, encoding=encoding) + 2
        while next_needle < len(needle_positions) and i == needle_positions[next_needle]:
            sentence = needles[next_needle].sentence
            out.append(sentence)
            cur_tokens += count_tokens(sentence, encoding=encoding) + 2
            next_needle += 1

    while next_needle < len(needles):
        out.append(needles[next_needle].sentence)
        next_needle += 1

    return "\n\n".join(out)


__all__ = [
    "HFArxivCorpus",
    "HFWikipediaCorpus",
    "MixedCorpus",
    "RealCorpus",
    "RealCorpusUnavailable",
    "build_real_haystack",
    "make_real_corpus",
]
