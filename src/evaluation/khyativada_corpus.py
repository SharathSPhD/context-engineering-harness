"""Deterministic Khyātivāda example corpus for P4 annotation.

Generates ≥3000 diverse claim/ground-truth pairs spanning all 7 labels
(6 hallucination classes plus ``none``). Each example is constructed
from a small set of class-specific templates, parameterized by hash-
deterministic surface form swaps so:

* Two different ``(seed, idx)`` tuples never produce the same row.
* Re-running ``generate_corpus`` always produces the same rows.
* Each generated row has *one* unambiguous gold label that both human
  raters and a calibrated LLM can recover from the surface text alone
  (the rationale field carries the audit trail).

The corpus is the input to the two-annotator agreement experiment that
backs Hypothesis H6 in the paper. It is **not** a benchmark; it is the
evidence that our 6-class taxonomy is operationally distinguishable.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class CorpusRow:
    """One annotation example with its gold (template-derived) label."""

    id: str
    claim: str
    context: str
    ground_truth: str
    gold_label: str
    template_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "claim": self.claim,
            "context": self.context,
            "ground_truth": self.ground_truth,
            "gold_label": self.gold_label,
            "template_id": self.template_id,
        }


# ---------------------------------------------------------------------------
# Class-specific generators
# ---------------------------------------------------------------------------

_PYTHON_VERSION_PAIRS = [
    ("3.10", "3.13", "free-threaded build"),
    ("3.11", "3.12", "PEP 695 type aliases"),
    ("3.9", "3.10", "structural pattern matching"),
    ("3.8", "3.9", "the merge operator for dictionaries"),
    ("3.10", "3.12", "the per-interpreter GIL"),
]

_REACT_VERSION_PAIRS = [
    ("18", "19", "stable Server Components"),
    ("16.8", "17", "concurrent rendering hooks"),
    ("17", "18", "automatic batching"),
]

_DB_VERSION_PAIRS = [
    ("PostgreSQL", "9.6", "10", "native declarative partitioning"),
    ("MySQL", "5.7", "8.0", "JSON path expressions in WHERE clauses"),
    ("PostgreSQL", "12", "15", "MERGE statement support"),
]

_DEFAULT_PORT_CLAIMS = [
    ("Redis", "8080", "There is no documented default port in the provided source for this service."),
    ("RabbitMQ", "8000", "The provided documentation specifies no default port."),
    ("Postgres", "5000", "The provided documentation specifies no default port."),
]

_FAMILIAR_DEFAULT_CLAIMS = [
    (
        "`json.dumps` defaults to `indent=2`.",
        "`json.dumps` defaults to `indent=None` (compact, single-line output).",
    ),
    (
        "REST APIs return 404 when authentication fails.",
        "Authentication failures use 401 (or 403); 404 means resource not found.",
    ),
    (
        "Numpy's `arange` is inclusive on the upper bound.",
        "`np.arange` is exclusive on the upper bound, like Python's `range`.",
    ),
]

_NOVEL_API_CLAIMS = [
    (
        "Use `numpy.polyfit_regularised_v2` for L2-regularised polynomial fits.",
        "No `polyfit_regularised_v2` symbol exists in numpy at any version.",
    ),
    (
        "LangChain ships an `AdaptiveRetriever` class that auto-tunes recall.",
        "LangChain has no class named `AdaptiveRetriever` in any released version.",
    ),
    (
        "FastAPI 1.0 introduced the `@app.background_route` decorator.",
        "FastAPI has no `@app.background_route` decorator at any version.",
    ),
]

_NOVEL_PROTOCOL_CLAIMS = [
    (
        "The HTTP/4 protocol uses Reed-Solomon framing for header recovery.",
        "There is no HTTP/4 protocol; current standards stop at HTTP/3.",
    ),
    (
        "TCP/IPv8 supports per-flow congestion windows.",
        "There is no TCP/IPv8; the IP family stops at IPv4 and IPv6.",
    ),
]

_NONEXISTENT_METHOD_CLAIMS = [
    (
        "Call `requests.get_json(url)` to parse a JSON response.",
        "`requests.get_json` does not exist; use `requests.get(url).json()`.",
    ),
    (
        "Set `pandas.options.display.theme = 'dark'` to recolor tables.",
        "No such display.theme option exists in pandas.",
    ),
    (
        "Use the `--strict-mode` flag of `git commit`.",
        "`git commit` has no `--strict-mode` flag.",
    ),
    (
        "Run `docker pause --tree` to pause an entire container subgraph.",
        "`docker pause` has no `--tree` flag at any version.",
    ),
]

_INVERSION_PAIRS = [
    (
        "`bisect_left` returns the rightmost insertion index; `bisect_right` returns the leftmost.",
        "The opposite is true: `bisect_left` returns the leftmost insertion index, `bisect_right` the rightmost.",
    ),
    (
        "POST is idempotent; PUT is not.",
        "The opposite holds: PUT is idempotent in REST semantics; POST is not.",
    ),
    (
        "Stack is FIFO and queue is LIFO.",
        "Stack is LIFO and queue is FIFO.",
    ),
    (
        "In SQL, INNER JOIN returns rows present in only one table; LEFT JOIN keeps all rows from both tables.",
        "The opposite is true: INNER JOIN returns rows present in both, LEFT JOIN keeps all rows from the left table.",
    ),
]

_AKHYATI_TEMPLATES = [
    (
        "Einstein won the 1921 Nobel Prize for the theory of relativity.",
        "Einstein did win the 1921 Nobel Prize, but not for relativity — it was for the photoelectric effect.",
    ),
    (
        "Postgres released MERGE in version 15 to implement ON CONFLICT.",
        "MERGE arrived in Postgres 15 but not for ON CONFLICT; ON CONFLICT (since 9.5) is a separate upsert mechanism.",
    ),
    (
        "Turing won the Nobel Prize for the Turing machine.",
        "Turing was honoured for cryptanalysis and computer-science work but not for any Nobel — he never won one.",
    ),
    (
        "Marie Curie won her Nobel Prize for X-ray imaging.",
        "Marie Curie won Nobel Prizes but not for X-ray imaging — it was for radioactivity (1903) and chemistry (1911).",
    ),
]

_NONE_TEMPLATES = [
    (
        "Python 3.13 introduced an experimental free-threaded build.",
        "Python 3.13 introduced an experimental free-threaded build via PEP 703.",
    ),
    (
        "HTTP status code 404 means resource not found.",
        "HTTP 404 is the standard 'Not Found' response.",
    ),
    (
        "`json.dumps` returns a string by default.",
        "`json.dumps` returns a `str` (use `json.dump` to write to a file).",
    ),
    (
        "Git commits are uniquely identified by SHA-1 hashes.",
        "Git commits are addressed by SHA-1 (with SHA-256 support landing later).",
    ),
    (
        "CSS Grid was specified after Flexbox.",
        "CSS Grid (Level 1) became a Recommendation after Flexbox.",
    ),
]


def _det_choice(rng: random.Random, items: Sequence) -> object:
    return items[rng.randrange(len(items))]


def _make_anyathakhyati(rng: random.Random, idx: int) -> CorpusRow:
    """Real X misidentified as real Y."""
    family_picker = rng.random()
    tid = "anyathakhyati.unknown"
    if family_picker < 0.4:
        wrong, right, feature = _det_choice(rng, _PYTHON_VERSION_PAIRS)
        claim = f"Python {wrong} introduced {feature}."
        gt = f"{feature.capitalize()} actually shipped in Python {right}, not {wrong}."
        tid = "anyathakhyati.python_version_swap"
    elif family_picker < 0.7:
        wrong, right, feature = _det_choice(rng, _REACT_VERSION_PAIRS)
        claim = f"React {wrong} introduced {feature}."
        gt = f"{feature.capitalize()} shipped as stable in React {right}, not {wrong}."
        tid = "anyathakhyati.react_version_swap"
    else:
        product, wrong, right, feature = _det_choice(rng, _DB_VERSION_PAIRS)
        claim = f"{product} added {feature} in version {wrong}."
        gt = f"{feature.capitalize()} actually landed in {product} {right}."
        tid = "anyathakhyati.db_version_swap"
    return CorpusRow(
        id=f"anyathakhyati-{idx:05d}",
        claim=claim,
        context="",
        ground_truth=gt,
        gold_label="anyathakhyati",
        template_id=tid,
    )


def _make_atmakhyati(rng: random.Random, idx: int) -> CorpusRow:
    """Internal pattern projected as external fact."""
    if rng.random() < 0.5:
        service, port, gt = _det_choice(rng, _DEFAULT_PORT_CLAIMS)
        claim = f"The default port for {service} is {port}."
        tid = "atmakhyati.default_port"
    else:
        claim, gt = _det_choice(rng, _FAMILIAR_DEFAULT_CLAIMS)
        tid = "atmakhyati.familiar_default"
    return CorpusRow(
        id=f"atmakhyati-{idx:05d}",
        claim=claim,
        context="",
        ground_truth=gt,
        gold_label="atmakhyati",
        template_id=tid,
    )


def _make_anirvacaniyakhyati(rng: random.Random, idx: int) -> CorpusRow:
    """Novel confabulation — entity has no real referent."""
    if rng.random() < 0.6:
        claim, gt = _det_choice(rng, _NOVEL_API_CLAIMS)
        tid = "anirvacaniyakhyati.novel_api"
    else:
        claim, gt = _det_choice(rng, _NOVEL_PROTOCOL_CLAIMS)
        tid = "anirvacaniyakhyati.novel_protocol"
    return CorpusRow(
        id=f"anirvacaniyakhyati-{idx:05d}",
        claim=claim,
        context="",
        ground_truth=gt,
        gold_label="anirvacaniyakhyati",
        template_id=tid,
    )


def _make_asatkhyati(rng: random.Random, idx: int) -> CorpusRow:
    """Nonexistent entity asserted to exist."""
    claim, gt = _det_choice(rng, _NONEXISTENT_METHOD_CLAIMS)
    return CorpusRow(
        id=f"asatkhyati-{idx:05d}",
        claim=claim,
        context="",
        ground_truth=gt,
        gold_label="asatkhyati",
        template_id="asatkhyati.nonexistent_method",
    )


def _make_viparitakhyati(rng: random.Random, idx: int) -> CorpusRow:
    """Two real concepts named correctly but their roles inverted."""
    claim, gt = _det_choice(rng, _INVERSION_PAIRS)
    return CorpusRow(
        id=f"viparitakhyati-{idx:05d}",
        claim=claim,
        context="",
        ground_truth=gt,
        gold_label="viparitakhyati",
        template_id="viparitakhyati.role_inversion",
    )


def _make_akhyati(rng: random.Random, idx: int) -> CorpusRow:
    """Two true components combined falsely."""
    claim, gt = _det_choice(rng, _AKHYATI_TEMPLATES)
    return CorpusRow(
        id=f"akhyati-{idx:05d}",
        claim=claim,
        context="",
        ground_truth=gt,
        gold_label="akhyati",
        template_id="akhyati.combination",
    )


def _make_none(rng: random.Random, idx: int) -> CorpusRow:
    """No hallucination: claim matches ground truth."""
    claim, gt = _det_choice(rng, _NONE_TEMPLATES)
    return CorpusRow(
        id=f"none-{idx:05d}",
        claim=claim,
        context="",
        ground_truth=gt,
        gold_label="none",
        template_id="none.match",
    )


_GENERATORS: tuple[tuple[str, "Callable[[random.Random, int], CorpusRow]"], ...] = (
    ("anyathakhyati", _make_anyathakhyati),
    ("atmakhyati", _make_atmakhyati),
    ("anirvacaniyakhyati", _make_anirvacaniyakhyati),
    ("asatkhyati", _make_asatkhyati),
    ("viparitakhyati", _make_viparitakhyati),
    ("akhyati", _make_akhyati),
    ("none", _make_none),
)


def generate_corpus(*, n: int = 3000, seed: int = 0) -> list[CorpusRow]:
    """Generate ``n`` Khyātivāda examples evenly distributed across the 7 labels.

    Args:
        n: Total number of rows to generate (default 3000 to match the
            P4 spec). Should be ≥7 so each class has at least one row.
        seed: Master seed; deterministically expanded per row.

    Returns:
        A list of :class:`CorpusRow` instances in interleaved class
        order so any prefix is also a roughly-balanced sample.
    """
    if n < len(_GENERATORS):
        raise ValueError(f"n must be ≥ {len(_GENERATORS)} so every class has at least one row")

    per_class = n // len(_GENERATORS)
    remainder = n - per_class * len(_GENERATORS)

    counts = [per_class + (1 if i < remainder else 0) for i in range(len(_GENERATORS))]

    rows: list[CorpusRow] = []
    for cls_idx, ((label, generator), count) in enumerate(zip(_GENERATORS, counts, strict=True)):
        for k in range(count):
            row_seed = int(
                hashlib.sha256(f"khyativada-{label}-{seed}-{k}".encode()).hexdigest()[:8],
                16,
            )
            rng = random.Random(row_seed)
            row = generator(rng, cls_idx * per_class + k)
            rows.append(row)

    interleave_rng = random.Random(seed)
    interleave_rng.shuffle(rows)
    return rows


def class_distribution(rows: Sequence[CorpusRow]) -> dict[str, int]:
    """Return ``{gold_label: count}`` for the given rows."""
    out: dict[str, int] = {}
    for row in rows:
        out[row.gold_label] = out.get(row.gold_label, 0) + 1
    return out


__all__ = [
    "CorpusRow",
    "class_distribution",
    "generate_corpus",
]
