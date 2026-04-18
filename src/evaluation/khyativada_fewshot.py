"""Few-shot Khyātivāda classifier with structured output and heuristic guardrails.

Replaces the v0 ``KhyativadaClassifier.classify`` LLM path with:

1. A curated few-shot exemplar bank (≥3 per class, 21 total) drawn from
   the same 6-class taxonomy used in the paper:
   *anyathakhyati*, *atmakhyati*, *anirvacaniyakhyati*, *asatkhyati*,
   *viparitakhyati*, *akhyati*.
2. A strict JSON-only response contract enforced both by the system
   prompt and a parser that tolerates fenced code blocks and noise
   around the JSON object.
3. Heuristic guardrails that *override* the LLM when its prediction
   contradicts unambiguous textual evidence (e.g. ground truth says
   "does not exist" → the verdict must be ``asatkhyati``). Guardrails
   are explicitly tagged so the audit trail is unambiguous about which
   path produced the final label.
4. A graceful fallback chain — LLM → heuristic — so the classifier
   keeps producing labels even when the CLI is unavailable, mis-quoted,
   or rate-limited.

The classifier deliberately does *not* depend on
:class:`src.cli_bridge.ClaudeCLIClient` at import time; the client
factory is injected so tests, hooks, and the plugin runtime can wire
their own caller (e.g. a scheduler-aware wrapper).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.evaluation.khyativada import KhyativadaClass, KhyativadaClassifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KhyativadaPrediction:
    """Structured classifier verdict with full provenance."""

    label: str  # one of KhyativadaClass values, including "none"
    confidence: float
    rationale: str
    source: str  # "llm" | "heuristic" | "guardrail" | "fallback"
    llm_label: str | None = None
    heuristic_label: str | None = None
    agreement: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "class": self.label,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source": self.source,
            "llm_class": self.llm_label,
            "heuristic_class": self.heuristic_label,
            "agreement": self.agreement,
        }


@dataclass(frozen=True)
class _Exemplar:
    label: str
    claim: str
    ground_truth: str
    rationale: str


# Curated few-shot exemplar bank (3+ per class)
_EXEMPLARS: tuple[_Exemplar, ...] = (
    # anyathakhyati: real X misidentified as real Y
    _Exemplar(
        label="anyathakhyati",
        claim="The free GIL was removed in Python 3.10.",
        ground_truth="The optional free-threaded build (PEP 703) was added in Python 3.13, not 3.10.",
        rationale="Both Python 3.10 and 3.13 are real versions; one is misidentified as the other.",
    ),
    _Exemplar(
        label="anyathakhyati",
        claim="React 18 introduced server components.",
        ground_truth="React Server Components shipped as stable in React 19, not 18.",
        rationale="Two real React versions confused; the feature attribution is misplaced.",
    ),
    _Exemplar(
        label="anyathakhyati",
        claim="PostgreSQL added partitioning in version 9.6.",
        ground_truth="Native declarative partitioning landed in PostgreSQL 10.",
        rationale="Real adjacent versions, misattributed feature ownership.",
    ),
    # atmakhyati: internal pattern projected as external fact
    _Exemplar(
        label="atmakhyati",
        claim="The default port for this internal service is 8080.",
        ground_truth="The provided documentation specifies no default port for this service.",
        rationale="The model emits a habitual default that is not actually grounded in the source.",
    ),
    _Exemplar(
        label="atmakhyati",
        claim="`json.dumps` defaults to `indent=2`.",
        ground_truth="`json.dumps` defaults to `indent=None` (compact, single-line output).",
        rationale="The model projects a familiar formatter convention onto an unrelated API.",
    ),
    _Exemplar(
        label="atmakhyati",
        claim="REST APIs return 404 when authentication fails.",
        ground_truth="Authentication failures use 401 (or 403 for authorization); 404 means resource not found.",
        rationale="A learned pattern (HTTP 404 is common) is projected onto a different failure mode.",
    ),
    # anirvacaniyakhyati: novel confabulation with no real referent
    _Exemplar(
        label="anirvacaniyakhyati",
        claim="Use the `polyfit_regularised_v2` solver from numpy 2.1.",
        ground_truth="No `polyfit_regularised_v2` symbol exists in numpy at any version.",
        rationale="The named entity is wholly invented and cannot be located in any real reference.",
    ),
    _Exemplar(
        label="anirvacaniyakhyati",
        claim="The HTTP/4 protocol uses Reed-Solomon framing.",
        ground_truth="There is no HTTP/4 protocol; current standards stop at HTTP/3.",
        rationale="An entire fictitious protocol is asserted with elaborate but ungrounded detail.",
    ),
    _Exemplar(
        label="anirvacaniyakhyati",
        claim="LangChain ships an `AdaptiveRetriever` class.",
        ground_truth="LangChain has no class named `AdaptiveRetriever` in any released version.",
        rationale="Plausible-sounding novel API name with no actual referent in the library.",
    ),
    # asatkhyati: nonexistent entity asserted to exist
    _Exemplar(
        label="asatkhyati",
        claim="Call `requests.get_json(url)` to parse a JSON response.",
        ground_truth="`requests.get_json` does not exist; use `requests.get(url).json()`.",
        rationale="The asserted method has no referent at all in the requests library.",
    ),
    _Exemplar(
        label="asatkhyati",
        claim="Set `pandas.options.display.theme = 'dark'` to recolor tables.",
        ground_truth="No such display.theme option exists in pandas.",
        rationale="The configuration key is non-existent; ground truth is explicit about non-existence.",
    ),
    _Exemplar(
        label="asatkhyati",
        claim="Use the `--strict-mode` flag of `git commit`.",
        ground_truth="`git commit` has no `--strict-mode` flag.",
        rationale="The flag is fabricated; ground truth indicates outright non-existence.",
    ),
    # viparitakhyati: systematic inversion of A and B
    _Exemplar(
        label="viparitakhyati",
        claim="`bisect_left` returns the rightmost insertion index; `bisect_right` returns the leftmost.",
        ground_truth="The opposite is true: `bisect_left` returns the leftmost insertion index, `bisect_right` the rightmost.",
        rationale="The two real functions are correctly named but their behaviours are swapped.",
    ),
    _Exemplar(
        label="viparitakhyati",
        claim="POST is idempotent; PUT is not.",
        ground_truth="The opposite holds: PUT is idempotent in REST semantics; POST is not.",
        rationale="Two real verbs are correctly identified but their semantic roles are inverted.",
    ),
    _Exemplar(
        label="viparitakhyati",
        claim="Stack is FIFO and queue is LIFO.",
        ground_truth="Stack is LIFO and queue is FIFO.",
        rationale="Both data structures exist; their behaviours are systematically swapped.",
    ),
    # akhyati: two true components combined falsely
    _Exemplar(
        label="akhyati",
        claim="Einstein won the 1921 Nobel Prize for the theory of relativity.",
        ground_truth="Einstein did win the 1921 Nobel Prize, but for the photoelectric effect, not relativity.",
        rationale="Year, prize, and theory are individually correct; the *causal* combination is wrong.",
    ),
    _Exemplar(
        label="akhyati",
        claim="Postgres released MERGE in version 15 to support upserts via ON CONFLICT.",
        ground_truth="MERGE arrived in Postgres 15, but ON CONFLICT (introduced in 9.5) is a separate upsert mechanism.",
        rationale="Both facts are individually true but their causal/syntactic linking is fabricated.",
    ),
    _Exemplar(
        label="akhyati",
        claim="Turing won the Nobel Prize for the Turing machine.",
        ground_truth="Turing was honoured for cryptanalysis and computer-science work, but he never won a Nobel Prize.",
        rationale="The Turing machine and Turing's reputation are real; the prize attribution joins them falsely.",
    ),
    # No-hallucination ("none") exemplars to make the model willing to abstain
    _Exemplar(
        label="none",
        claim="Python 3.13 introduced an experimental free-threaded build.",
        ground_truth="Python 3.13 introduced an experimental free-threaded build via PEP 703.",
        rationale="Claim matches the ground truth.",
    ),
    _Exemplar(
        label="none",
        claim="HTTP status code 404 means resource not found.",
        ground_truth="HTTP 404 is the standard 'Not Found' response.",
        rationale="The claim is factually correct.",
    ),
    _Exemplar(
        label="none",
        claim="`json.dumps` returns a string by default.",
        ground_truth="`json.dumps` returns a `str` (use `json.dump` to write to a file).",
        rationale="The claim matches the documented behaviour.",
    ),
)


_VALID_LABELS = {c.value for c in KhyativadaClass}


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_INLINE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_structured_response(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM response.

    Tolerates Markdown fences, leading prose, and trailing commentary so a
    minor formatting drift never trips the classifier into the heuristic
    fallback unnecessarily. Raises ``ValueError`` if no parsable object
    is found.
    """
    if not text:
        raise ValueError("empty response")
    fenced = _FENCED_JSON_RE.search(text)
    candidates: list[str] = []
    if fenced:
        candidates.append(fenced.group(1))
    inline = _INLINE_JSON_RE.search(text)
    if inline:
        candidates.append(inline.group(0))
    candidates.append(text.strip())
    last_exc: Exception | None = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
    raise ValueError(f"no JSON object found in response (last error: {last_exc})")


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    label = payload.get("class") or payload.get("label")
    if label not in _VALID_LABELS:
        raise ValueError(f"invalid class {label!r}; expected one of {sorted(_VALID_LABELS)}")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"invalid confidence {confidence!r}; expected float in [0, 1]")
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    return {"class": label, "confidence": float(confidence), "rationale": rationale.strip()}


def _apply_guardrails(
    claim: str,
    ground_truth: str,
    llm_label: str,
) -> tuple[str, str | None]:
    """Apply heuristic overrides; return ``(final_label, override_reason)``."""
    gt_lower = ground_truth.lower()
    if any(token in gt_lower for token in ("does not exist", "no such", "nonexistent", "non-existent")):
        if llm_label != "asatkhyati":
            return "asatkhyati", "ground_truth_asserts_nonexistence"
    if any(token in gt_lower for token in ("opposite is true", "the opposite holds", "swapped")):
        if llm_label != "viparitakhyati":
            return "viparitakhyati", "ground_truth_asserts_inversion"
    if any(token in gt_lower for token in ("but not for", "but not due to", "incorrectly attributed")):
        if llm_label != "akhyati":
            return "akhyati", "ground_truth_asserts_relational_combination"
    return llm_label, None


def _format_exemplar_block(exemplars: list[_Exemplar]) -> str:
    lines: list[str] = []
    for ex in exemplars:
        lines.append(
            "Claim: "
            + ex.claim
            + "\nGround truth: "
            + ex.ground_truth
            + "\n"
            + json.dumps(
                {"class": ex.label, "confidence": 0.9, "rationale": ex.rationale},
                ensure_ascii=False,
            )
        )
    return "\n\n".join(lines)


def _select_exemplars(per_class: int, *, seed: int) -> list[_Exemplar]:
    """Deterministically sample ``per_class`` exemplars per label (label order preserved)."""
    import random

    rng = random.Random(seed)
    by_label: dict[str, list[_Exemplar]] = {}
    for ex in _EXEMPLARS:
        by_label.setdefault(ex.label, []).append(ex)
    chosen: list[_Exemplar] = []
    for label in (
        "anyathakhyati",
        "atmakhyati",
        "anirvacaniyakhyati",
        "asatkhyati",
        "viparitakhyati",
        "akhyati",
        "none",
    ):
        bucket = by_label.get(label, [])
        if not bucket:
            continue
        rng.shuffle(bucket)
        chosen.extend(bucket[: max(1, per_class)])
    return chosen


_SYSTEM_PROMPT = """You are a Khyātivāda hallucination classifier. You assign exactly one of these
6 fault classes (or "none" if the claim is not hallucinated):

- anyathakhyati: a real entity X is misidentified as another real entity Y.
- atmakhyati: an internal habit / training pattern is projected as an external fact.
- anirvacaniyakhyati: the entity is wholly novel — no real-world or documented referent.
- asatkhyati: a non-existent entity is asserted to exist.
- viparitakhyati: two real concepts are correctly named but their roles are systematically inverted.
- akhyati: two individually true components are joined into a false combined claim.

Output strictly one JSON object with keys: class, confidence (0..1), rationale (one sentence).
No prose outside the JSON. No markdown fences."""


ClientFactory = Callable[[], Any]


class FewShotKhyativadaClassifier:
    """Few-shot Khyātivāda classifier with structured output + guardrails."""

    def __init__(
        self,
        client_factory: ClientFactory | None = None,
        *,
        model: str = "claude-haiku-4-5",
        n_shots_per_class: int = 2,
        max_tokens: int = 256,
        seed: int = 0,
        heuristic: KhyativadaClassifier | None = None,
    ) -> None:
        self._client_factory = client_factory
        self.model = model
        self.n_shots_per_class = n_shots_per_class
        self.max_tokens = max_tokens
        self.seed = seed
        self._heuristic = heuristic or KhyativadaClassifier()

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from src.cli_bridge import ClaudeCLIClient

        return ClaudeCLIClient()

    def build_prompt(self, claim: str, context: str, ground_truth: str) -> str:
        exemplars = _select_exemplars(self.n_shots_per_class, seed=self.seed)
        return (
            "Examples:\n\n"
            + _format_exemplar_block(exemplars)
            + "\n\nNow classify:\n"
            f"Claim: {claim}\nContext provided: {context or 'none'}\nGround truth: {ground_truth}\n"
            "Respond with one JSON object only."
        )

    def classify(self, claim: str, context: str, ground_truth: str) -> KhyativadaPrediction:
        heuristic_result = self._heuristic.classify_heuristic(claim, ground_truth)
        heuristic_label = (
            heuristic_result["class"].value
            if isinstance(heuristic_result["class"], KhyativadaClass)
            else str(heuristic_result["class"])
        )

        try:
            client = self._client()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self.build_prompt(claim, context, ground_truth)}],
                seed=self.seed,
            )
            payload = _parse_structured_response(response.content[0].text)
            payload = _validate_payload(payload)
        except Exception as exc:
            logger.info("LLM classification failed (%s); falling back to heuristic", exc)
            return KhyativadaPrediction(
                label=heuristic_label,
                confidence=float(heuristic_result.get("confidence", 0.5)),
                rationale=str(heuristic_result.get("rationale", "")),
                source="heuristic",
                llm_label=None,
                heuristic_label=heuristic_label,
                agreement=None,
            )

        llm_label = str(payload["class"])
        final_label, override = _apply_guardrails(claim, ground_truth, llm_label)
        if override is not None:
            return KhyativadaPrediction(
                label=final_label,
                confidence=max(0.6, float(payload["confidence"])),
                rationale=f"guardrail-override:{override}; llm-said={llm_label}",
                source="guardrail",
                llm_label=llm_label,
                heuristic_label=heuristic_label,
                agreement=(llm_label == heuristic_label),
            )

        return KhyativadaPrediction(
            label=llm_label,
            confidence=float(payload["confidence"]),
            rationale=str(payload["rationale"]),
            source="llm",
            llm_label=llm_label,
            heuristic_label=heuristic_label,
            agreement=(llm_label == heuristic_label),
        )

    def batch_classify(self, examples: list[dict]) -> list[KhyativadaPrediction]:
        return [
            self.classify(
                e.get("claim", ""),
                e.get("context", ""),
                e.get("ground_truth", ""),
            )
            for e in examples
        ]


__all__ = [
    "FewShotKhyativadaClassifier",
    "KhyativadaPrediction",
]
