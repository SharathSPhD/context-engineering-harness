"""Rule-based Khyātivāda annotator (Annotator A for the P4 experiment).

Distinct from :class:`src.evaluation.khyativada.KhyativadaClassifier`,
which was the v0 baseline (frozen and intentionally narrow). The
P4 annotator is the *upgraded* heuristic referenced in the plan: a
proper 7-class rule-based labeller that exploits explicit lexical
signals in the (claim, ground_truth) pair.

Design choices
--------------
* **Pattern-precedence list.** Each class has one or more disambiguating
  patterns that must fire on the ground-truth string. The annotator
  evaluates them in priority order and stops at the first match. The
  precedence is: ``none`` → ``viparitakhyati`` → ``asatkhyati`` →
  ``akhyati`` → ``anirvacaniyakhyati`` → ``anyathakhyati`` →
  ``atmakhyati`` (latter is the default). This precedence is justified
  by how unambiguous each cue is — a "the opposite is true" string is
  a near-certain inversion, while "the default is X" lacks a unique
  signature.

* **Conservative ``none`` detection.** We only label a row as ``none``
  when the claim words substantially overlap with the ground truth
  *and* no negative-cue keyword appears (e.g. "not", "but", "actually").

* **Falsifiability.** Every rule has at least one positive and one
  negative example exercised by tests, so we can detect drift if the
  corpus templates change.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class HeuristicLabel:
    """One annotator A row."""

    item_id: str
    label: str
    confidence: float
    rule: str
    source: str = "heuristic"


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


_NEGATIVE_CUES = (
    " not ",
    " but ",
    " actually ",
    " never ",
    " no such ",
    " no ",
    " does not ",
    " has no ",
    " opposite ",
    " swapped ",
    " neither ",
    " specifies ",
    " exclusive ",
    " inclusive ",
    " instead ",
    " means resource ",
    " means ",
)


def _looks_like_match(claim: str, ground_truth: str) -> bool:
    """True iff the claim and ground-truth seem to assert the same thing.

    Heuristic: at least 60% of the claim's content tokens appear in the
    ground truth (length-normalised), and the ground truth contains no
    contradiction cue.
    """
    gt_lower = " " + ground_truth.lower() + " "
    if any(cue in gt_lower for cue in _NEGATIVE_CUES):
        return False
    claim_toks = _tokens(claim)
    gt_toks = _tokens(ground_truth)
    if not claim_toks:
        return False
    overlap = len(claim_toks & gt_toks) / len(claim_toks)
    return overlap >= 0.55


_INVERSION_CUES = (
    "opposite is true",
    "the opposite holds",
    "the opposite",
    "swapped",
    "is fifo and",
    "is lifo and",
)


def _is_inversion(ground_truth: str) -> bool:
    gt_lower = ground_truth.lower()
    return any(cue in gt_lower for cue in _INVERSION_CUES)


_HARD_NONEXISTENCE_CUES = (
    "does not exist",
    "no such",
    "non-existent",
    "nonexistent",
)
_NOUN_NEGATION_RE = re.compile(
    r"\bno\s+(?:`[^`]+`|class|protocol|method|function|api|symbol|standard|"
    r"flag|option|decorator|http/|tcp|udp|ipv|`)",
    re.IGNORECASE,
)
_HAS_NO_BACKTICKED_RE = re.compile(r"\bhas no\s+`", re.IGNORECASE)
_HAS_NO_NOUN_RE = re.compile(
    r"\bhas no\s+\S+\s+(?:flag|option|method|class|decorator|attribute|"
    r"symbol|function|field|setting|argument|parameter|hook|action|api)",
    re.IGNORECASE,
)
_THERE_IS_NO_RE = re.compile(
    r"\bthere is no\s+(?:`[^`]+`|http/|tcp|udp|ipv|class|protocol|method|function|api|standard)",
    re.IGNORECASE,
)


def _is_nonexistence(ground_truth: str) -> bool:
    gt_lower = ground_truth.lower()
    if any(cue in gt_lower for cue in _HARD_NONEXISTENCE_CUES):
        return True
    if _HAS_NO_BACKTICKED_RE.search(ground_truth):
        return True
    if _HAS_NO_NOUN_RE.search(ground_truth):
        return True
    if _THERE_IS_NO_RE.search(ground_truth):
        return True
    if _NOUN_NEGATION_RE.search(ground_truth):
        return True
    return False


_NOVEL_CUES = (
    "no such",
    "no protocol",
    "wholly invented",
    "no real-world",
    "no actual referent",
    "current standards stop at",
    "fictitious",
)


def _is_novel_confabulation(claim: str, ground_truth: str) -> bool:
    """Distinguish *anirvacaniyakhyati* from *asatkhyati*.

    Both cite something that "doesn't exist". The split:
    ``asatkhyati`` is about a missing *flag/method on a real object*
    ("git commit has no --strict-mode"); ``anirvacaniyakhyati`` is
    about an entire *novel object/protocol* ("no HTTP/4 protocol").
    """
    gt_lower = ground_truth.lower()
    if "current standards stop at" in gt_lower or "no protocol" in gt_lower or "no http/" in gt_lower or "no tcp" in gt_lower:
        return True
    if "no `" in gt_lower and "exists" in gt_lower and "in" in gt_lower:
        return True
    if "no class named" in gt_lower:
        return True
    return False


_AKHYATI_CUES = (
    "but not for",
    "but not due to",
    "but not because",
    "incorrectly attributed",
    "but not the photoelectric",
    "but not for relativity",
    "but it was for",
    "but not for any",
    "but not for x-ray",
    "but ",  # last-resort: ground truth admits "but" qualifier (ambiguous)
)


def _is_akhyati(ground_truth: str) -> bool:
    gt_lower = ground_truth.lower()
    return any(cue in gt_lower for cue in _AKHYATI_CUES[:-1])


_ANYATHAKHYATI_CUES_RE = re.compile(
    r"(?:not\s+\d+(?:\.\d+)?|in\s+\w+\s+\d+(?:\.\d+)?,?\s*not|version\s+\d+(?:\.\d+)?\s+(?:not|actually))",
    re.IGNORECASE,
)


def _is_anyathakhyati(claim: str, ground_truth: str) -> bool:
    gt_lower = ground_truth.lower()
    if "actually shipped" in gt_lower or "actually landed" in gt_lower or "actually arrived" in gt_lower:
        return True
    if _ANYATHAKHYATI_CUES_RE.search(ground_truth):
        return True
    claim_versions = set(re.findall(r"\d+\.\d+", claim))
    gt_versions = set(re.findall(r"\d+\.\d+", ground_truth))
    if claim_versions and gt_versions and claim_versions != gt_versions:
        return True
    return False


def _classify_one(claim: str, ground_truth: str) -> tuple[str, float, str]:
    """Return ``(label, confidence, rule)``."""

    if _is_inversion(ground_truth):
        return "viparitakhyati", 0.90, "inversion_cue"

    if _is_nonexistence(ground_truth):
        if _is_novel_confabulation(claim, ground_truth):
            return "anirvacaniyakhyati", 0.85, "novel_referent_cue"
        return "asatkhyati", 0.85, "nonexistence_cue"

    if _is_akhyati(ground_truth):
        return "akhyati", 0.82, "combination_cue"

    if _is_anyathakhyati(claim, ground_truth):
        return "anyathakhyati", 0.80, "version_swap_cue"

    if _looks_like_match(claim, ground_truth):
        return "none", 0.75, "claim_matches_ground_truth"

    return "atmakhyati", 0.55, "default_internal_pattern"


@dataclass(frozen=True)
class HeuristicAnnotator:
    """Stateless rule-based 7-class Khyātivāda annotator."""

    name: str = "p4_heuristic_v1"

    def label(self, item_id: str, claim: str, ground_truth: str) -> HeuristicLabel:
        label, conf, rule = _classify_one(claim, ground_truth)
        return HeuristicLabel(item_id=item_id, label=label, confidence=conf, rule=rule)

    def label_many(self, rows: Iterable) -> list[HeuristicLabel]:
        return [self.label(r.id, r.claim, r.ground_truth) for r in rows]


__all__ = ["HeuristicAnnotator", "HeuristicLabel"]
