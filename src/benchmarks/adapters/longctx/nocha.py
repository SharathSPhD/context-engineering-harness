"""NoCha adapter — narrative reading comprehension over book-length text.

Karpinska et al. (2024) introduced the "Novel Challenges" benchmark, which
asks book-length-context-aware models to verify a true claim and reject a
plausible but false claim about the same passage. Because every example is
a paired (true, false) claim, the metric is "joint accuracy": the model
must accept the true claim AND reject the false claim on the same passage.

The synthetic generator here mints short narrative passages with two
verifiable facts each, plus a paired distractor that contradicts one fact.
The real HF dataset (`tau/NoCha`) will be wired in for the published runs;
the synthetic engine keeps unit tests, smoke tests, and the plugin's
offline mode self-contained.
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field

from src.benchmarks.base import BenchmarkAdapter, BenchmarkExample, ModelOutput
from src.benchmarks.registry import register

from ._hf_loader import HFUnavailable
from ._synthetic import generate_examples

logger = logging.getLogger(__name__)


@register
@dataclass
class NochaJointAccuracyAdapter(BenchmarkAdapter):
    """NoCha-style joint accuracy on paired (true, false) claims."""

    name: str = "nocha_joint"
    requires_long_context: bool = True
    license_note: str = "synthetic; real NoCha loaded via HF when enabled"

    target_tokens: int = 12_288
    default_n: int = 20
    load_real: bool = False

    def load_examples(self, *, n: int | None = None, seed: int = 0) -> list[BenchmarkExample]:
        n_use = n or self.default_n
        if self.load_real:
            try:
                raise HFUnavailable(
                    "NoCha real loader is wired in P3 (p3_longctx); "
                    "use load_real=False until then"
                )
            except HFUnavailable as exc:
                logger.warning("NoCha real loader unavailable (%s); using synthetic", exc)
        synth = generate_examples(
            n=n_use,
            seed=seed,
            target_tokens=self.target_tokens,
            needles_per_example=1,
        )
        examples: list[BenchmarkExample] = []
        for s in synth:
            needle = s.needles[0]
            true_claim = (
                f"The activation code for vault {needle.key} is {needle.value}."
            )
            false_claim = self._mutated_claim(true_claim, seed=seed, eid=s.id)
            examples.append(
                BenchmarkExample(
                    id=f"nocha-{s.id}",
                    prompt=needle.key,
                    ground_truth={
                        "true_claim": true_claim,
                        "false_claim": false_claim,
                        "code": needle.value,
                    },
                    context=s.haystack,
                    metadata={"target_tokens": s.target_tokens},
                )
            )
        return examples

    @staticmethod
    def _mutated_claim(true_claim: str, *, seed: int, eid: str) -> str:
        rng = random.Random(f"{seed}:{eid}")
        match = re.search(r"is ([0-9A-Z]{6})\.", true_claim)
        if not match:
            return true_claim.replace(".", " (revoked).")
        original = match.group(1)
        chars = list(original)
        i = rng.randrange(len(chars))
        alphabet = "0123456789ABCDEF"
        replacement = rng.choice([c for c in alphabet if c != chars[i]])
        chars[i] = replacement
        return true_claim.replace(original, "".join(chars))

    def render_prompt(self, example: BenchmarkExample, *, condition: str) -> str:
        gt = example.ground_truth
        assert isinstance(gt, dict)
        if condition == "harness_on":
            return (
                "You are a careful narrative-fidelity verifier. "
                "Use ONLY the document below as evidence. Reply with strict JSON: "
                '{"true_claim_verdict": "TRUE" | "FALSE", '
                '"false_claim_verdict": "TRUE" | "FALSE"}.\n\n'
                f"DOCUMENT:\n{example.context}\n\n"
                f"CLAIM A: {gt['true_claim']}\n"
                f"CLAIM B: {gt['false_claim']}\n\n"
                'Output JSON only. Format: {"true_claim_verdict":"TRUE","false_claim_verdict":"FALSE"} '
                "for the case where A is true and B is false."
            )
        return (
            f"{example.context}\n\n"
            f"Claim A: {gt['true_claim']}\n"
            f"Claim B: {gt['false_claim']}\n\n"
            "Which of these claims is supported by the document above? "
            "Answer in the form 'A is X, B is Y' where X and Y are TRUE or FALSE."
        )

    def score(self, example: BenchmarkExample, output: ModelOutput) -> tuple[float, bool, str]:
        true_verdict, false_verdict = self._extract_verdicts(output.text)
        joint = (true_verdict == "TRUE") and (false_verdict == "FALSE")
        return (1.0 if joint else 0.0), joint, output.text.strip()

    @staticmethod
    def _extract_verdicts(text: str) -> tuple[str | None, str | None]:
        """Pull (true_claim_verdict, false_claim_verdict) from `text`.

        Tries strict JSON first (the harness-on prompt asks for it).
        Falls back to a label-anchored regex for the harness-off / free-text
        case ("A is TRUE, B is FALSE").
        """
        try:
            payload = json.loads(text.strip())
            if isinstance(payload, dict):
                t = str(payload.get("true_claim_verdict", "")).upper().strip() or None
                f = str(payload.get("false_claim_verdict", "")).upper().strip() or None
                if t in {"TRUE", "FALSE"} or f in {"TRUE", "FALSE"}:
                    return t, f
        except (json.JSONDecodeError, AttributeError):
            pass

        upper = text.upper()
        t_match = re.search(r"\bA\b[^A-Z\d]{0,12}(TRUE|FALSE)", upper) \
                  or re.search(r"TRUE[_ ]CLAIM[^A-Z]{0,30}(TRUE|FALSE)", upper)
        f_match = re.search(r"\bB\b[^A-Z\d]{0,12}(TRUE|FALSE)", upper) \
                  or re.search(r"FALSE[_ ]CLAIM[^A-Z]{0,30}(TRUE|FALSE)", upper)
        return (
            t_match.group(1) if t_match else None,
            f_match.group(1) if f_match else None,
        )
