"""ModelCaller implementations for P6-A.

Two implementations:

  * `MockHarnessCaller`   — deterministic stand-in for the `claude` CLI.
    Honours the harness contract: the structured "harness_on" prompt
    nudges the simulator toward higher recall on long-context payloads;
    the unstructured "harness_off" prompt makes the simulator leak
    distractors into its answer. Effect size is parameterised so the
    end-to-end runner + stats can be unit-tested at known ground truth.

  * `LiveCLICaller`        — wraps `tools.dev.scheduler.CLIBudgetScheduler`,
    so every call is cached, rate-limited, and ledger-tracked. Used only
    when `--live` is passed; never imported at unit-test time.

Both implementations satisfy `src.benchmarks.base.ModelCaller`:

    def __call__(*, prompt, model, max_tokens, system="", seed=None) -> ModelOutput
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from src.benchmarks.base import ModelOutput

logger = logging.getLogger(__name__)


# --- mock model ---------------------------------------------------------


@dataclass
class MockHarnessCaller:
    """Deterministic harness simulator.

    The simulator inspects (system + prompt) for "harness_on" cues —
    presence of a structured retriever instruction — and biases its
    extraction accordingly. It also reads the prompt for embedded
    activation codes (the synthetic generator inserts them inline as
    "activation code is XXXXXX") and either copies them out (treatment)
    or returns a noisy nearby distractor (baseline) at a configurable
    error rate.

    This is a *behaviour stub* — it does NOT make any network calls and
    is fully reproducible per (model, seed, prompt) hash.
    """

    treatment_recall: float = 0.92
    baseline_recall: float = 0.55
    haiku_penalty: float = 0.10  # weaker model has lower recall on long context
    max_keys_returned: int = 8

    _CODE_RE: re.Pattern = re.compile(
        r"activation code(?: for vault [A-Za-z0-9-]+)? is ([A-Z0-9]{6})",
        re.IGNORECASE,
    )
    _ALL_CODE_RE: re.Pattern = re.compile(r"\b([A-Z0-9]{6})\b")

    def __call__(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        system: str = "",
        seed: int | None = None,
    ) -> ModelOutput:
        full = (system or "") + "\n" + prompt
        is_treatment = self._is_treatment(full)
        recall = self.treatment_recall if is_treatment else self.baseline_recall
        if "haiku" in model.lower():
            recall = max(0.0, recall - self.haiku_penalty)

        # Find every (vault_id → code) pair printed in the haystack.
        pairs = self._extract_pairs(prompt)

        # Find which vault the user actually asked about.
        asked = self._asked_vault(prompt)

        # Recall multi-key answers (HELMET-recall, RULER-multi):
        if "list" in prompt.lower() and "vault" in prompt.lower() and len(pairs) > 1:
            kept = self._sample_keys(pairs, recall, model, seed, prompt)
            answer = "\n".join(f"{k}={v}" for k, v in kept[: self.max_keys_returned])
            return self._build(answer, full)

        # Single-key recall:
        if asked and asked in pairs:
            if self._roll(recall, model, seed, prompt):
                return self._build(pairs[asked], full)
            # Failure mode: emit a distractor code (baseline-style noise).
            distractors = [v for k, v in pairs.items() if k != asked]
            if distractors:
                idx = self._stable_int(model, seed, prompt) % len(distractors)
                return self._build(distractors[idx], full)
            return self._build("", full)

        # Fallback: return the first code we see (usually wrong on baseline).
        if pairs:
            first_value = next(iter(pairs.values()))
            return self._build(first_value if is_treatment else "ZZZZZZ", full)
        return self._build("", full)

    # --- helpers ----------------------------------------------------

    @staticmethod
    def _is_treatment(text: str) -> bool:
        """Detect harness_on cues inserted by adapters' render_prompt()."""
        cues = (
            "structured retriever",
            "retrieval-augmented assistant",
            "focused long-context retriever",
            "focused long-context",
            "answer with the single requested code",
        )
        low = text.lower()
        return any(c in low for c in cues)

    def _sample_keys(
        self,
        pairs: dict[str, str],
        recall: float,
        model: str,
        seed: int | None,
        prompt: str,
    ) -> list[tuple[str, str]]:
        kept: list[tuple[str, str]] = []
        for i, (k, v) in enumerate(pairs.items()):
            if self._roll(recall, model, seed, prompt + f"|{i}"):
                kept.append((k, v))
        return kept

    @classmethod
    def _extract_pairs(cls, prompt: str) -> dict[str, str]:
        """Return {vault_id: code} for every needle the synthetic generator printed."""
        # The synthetic generator writes:
        #   "The activation code for vault vault-XX-YY is ABC123."
        pat = re.compile(
            r"activation code for vault ([A-Za-z0-9-]+) is ([A-Z0-9]{6})",
            re.IGNORECASE,
        )
        out: dict[str, str] = {}
        for m in pat.finditer(prompt):
            out[m.group(1).upper()] = m.group(2).upper()
        return out

    @staticmethod
    def _asked_vault(prompt: str) -> str:
        """Pull the vault id the user asked about (best-effort regex)."""
        # Look in the trailing ANSWER section first (where the question is asked).
        m = re.search(r"vault ([A-Za-z0-9-]+)", prompt[-400:], re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"vault ([A-Za-z0-9-]+)", prompt, re.IGNORECASE)
        return m.group(1).upper() if m else ""

    @staticmethod
    def _stable_int(model: str, seed: int | None, prompt: str) -> int:
        h = hashlib.sha1(
            f"{model}|{seed}|{prompt[:512]}|{prompt[-256:]}".encode("utf-8")
        ).digest()
        return int.from_bytes(h[:8], "big")

    @classmethod
    def _roll(cls, prob: float, model: str, seed: int | None, prompt: str) -> bool:
        """Deterministic Bernoulli given (model, seed, prompt)."""
        u = (cls._stable_int(model, seed, prompt) % 1_000_000) / 1_000_000.0
        return u < prob

    @staticmethod
    def _build(text: str, full_input: str) -> ModelOutput:
        in_tok = max(1, len(full_input) // 4)
        out_tok = max(1, len(text) // 4)
        return ModelOutput(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=0.0,
            metadata={"caller": "MockHarnessCaller"},
        )


# --- live model ---------------------------------------------------------


class LiveCLICaller:
    """Wraps `tools.dev.scheduler.CLIBudgetScheduler.submit` as a `ModelCaller`.

    Imported lazily so the offline / CI path does not need `subprocess` or
    a working `claude` binary on PATH.
    """

    def __init__(self, scheduler) -> None:  # type: CLIBudgetScheduler
        self._sched = scheduler

    def __call__(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        system: str = "",
        seed: int | None = None,
    ) -> ModelOutput:
        result = self._sched.submit(
            model=model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            seed=seed,
        )
        return ModelOutput(
            text=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=float(result.latency_ms),
            metadata={
                "caller": "LiveCLICaller",
                "cache_hit": result.cache_hit,
                "regime": result.regime,
                "attempts": result.attempts,
                "prompt_hash": result.prompt_hash,
            },
        )


def make_caller(mode: str, *, scheduler=None) -> "callable":
    """Factory used by the CLI runner script."""
    if mode == "mock":
        return MockHarnessCaller()
    if mode == "live":
        if scheduler is None:
            raise ValueError("live mode requires a CLIBudgetScheduler instance")
        return LiveCLICaller(scheduler)
    raise ValueError(f"unknown caller mode {mode!r}; expected 'mock' or 'live'")
