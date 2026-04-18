"""Tokenizer-exact synthetic long-context generator.

This is what the harness uses for development, CI, and as the
deterministic fallback when the real HF dataset is unavailable. It is
the core engine behind both the RULER- and HELMET-style needle-in-a-
haystack tasks and (with adjusted prompts) the NoCha narrative tasks.

Design:
  - Token budget is honored using the same tokenizer we use everywhere
    in the project (`src.utils.tokenizer.count_tokens`, o200k_base).
  - Distractors are generated from a small Wikipedia-flavored corpus
    embedded below, expanded with hash-deterministic noise so two
    different seeds never produce the same haystack.
  - Each example carries a verifiable "golden" answer that any
    string-match scorer can check exactly.

This is NOT a substitute for real RULER / HELMET / NoCha when we publish
numbers in the paper; it IS what the unit tests, smoke tests, and the
plugin's offline mode use so the harness never blocks on a live dataset.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from src.utils.tokenizer import count_tokens

_FACT_TEMPLATES: tuple[str, ...] = (
    "The capital of {place_a} is {place_b}.",
    "{name_a} discovered the element {place_b} in 1{seed_year}.",
    "The {place_a} river flows through the city of {place_b}.",
    "{name_a} won the Nobel Prize in {place_b} in 19{seed_year}.",
    "The {place_a} dynasty ruled {place_b} for {seed_year} years.",
    "{name_a} composed the symphony titled '{place_b}' in 18{seed_year}.",
)

_PLACES: tuple[str, ...] = (
    "Norvinia", "Aldoria", "Brastia", "Velthia", "Kothalia", "Pyranthos", "Estoria",
    "Thalassa", "Dornesia", "Mervalia", "Carthavia", "Iolanthia", "Quendoria",
)
_NAMES: tuple[str, ...] = (
    "Asha Vega", "Mira Soto", "Linus Hart", "Owen Falke", "Kana Itou",
    "Pavel Markov", "Yara Khaled", "Nils Eklund", "Sofia Ricci", "Akira Mori",
)


def _det_choice(rng: random.Random, items: tuple[str, ...]) -> str:
    return items[rng.randrange(len(items))]


def _make_filler(rng: random.Random) -> str:
    """One filler sentence; never a needle."""
    return _det_choice(rng, _FACT_TEMPLATES).format(
        place_a=_det_choice(rng, _PLACES),
        place_b=_det_choice(rng, _PLACES),
        name_a=_det_choice(rng, _NAMES),
        seed_year=str(rng.randint(10, 99)),
    )


@dataclass(frozen=True)
class Needle:
    """One verifiable fact embedded in a haystack."""
    key: str          # what the model is asked about
    value: str        # the verifiable answer
    sentence: str     # the prose form embedded in the haystack


def make_needle(seed: int, idx: int) -> Needle:
    """Deterministically build a unique needle keyed off (seed, idx)."""
    digest = hashlib.sha256(f"needle-{seed}-{idx}".encode()).hexdigest()
    code_value = digest[:6].upper()
    key = f"vault-{seed:02d}-{idx:02d}"
    sentence = (
        f"The activation code for vault {key} is {code_value}. "
        f"Operators must memorize this code; it will not appear elsewhere."
    )
    return Needle(key=key, value=code_value, sentence=sentence)


def build_haystack(
    *,
    target_tokens: int,
    needles: list[Needle],
    seed: int,
    encoding: str = "o200k_base",
) -> str:
    """Build a haystack of approximately `target_tokens` tokens that contains
    every needle exactly once, with deterministic filler around them.

    The haystack token count is guaranteed not to exceed `target_tokens` by
    more than the size of one filler sentence (~25 tokens).
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    rng = random.Random(seed)
    needle_sentences = [n.sentence for n in needles]
    needle_positions = sorted(rng.sample(range(target_tokens // 30 + len(needles)), k=len(needles)))
    sentences: list[str] = []
    next_needle = 0
    cur_tokens = 0
    pos = 0
    while cur_tokens < target_tokens - 30:
        is_needle_slot = (
            next_needle < len(needles)
            and pos == needle_positions[next_needle]
        )
        if is_needle_slot:
            sentence = needle_sentences[next_needle]
            next_needle += 1
        else:
            sentence = _make_filler(rng)
        sentences.append(sentence)
        cur_tokens += count_tokens(sentence + " ", encoding=encoding)
        pos += 1
    while next_needle < len(needles):
        sentences.append(needle_sentences[next_needle])
        next_needle += 1
    return " ".join(sentences)


@dataclass(frozen=True)
class SyntheticExample:
    id: str
    haystack: str
    needles: list[Needle]
    target_tokens: int


def generate_examples(
    *,
    n: int,
    seed: int,
    target_tokens: int,
    needles_per_example: int = 1,
) -> list[SyntheticExample]:
    """Create `n` deterministic long-context examples of the requested size.

    The same (n, seed, target_tokens, needles_per_example) tuple always
    yields exactly the same examples, regardless of when or where it is run.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    examples: list[SyntheticExample] = []
    for i in range(n):
        seed_i = seed * 10_000 + i
        needles = [make_needle(seed_i, k) for k in range(needles_per_example)]
        haystack = build_haystack(
            target_tokens=target_tokens,
            needles=needles,
            seed=seed_i,
        )
        examples.append(
            SyntheticExample(
                id=f"synth-{seed:02d}-{i:04d}",
                haystack=haystack,
                needles=needles,
                target_tokens=target_tokens,
            )
        )
    return examples
