"""Per-token surprise scoring for event-boundary detection.

Background
----------
:class:`src.compaction.detector.EventBoundaryDetector` consumes a stream
of per-token surprise values and flags positions where surprise spikes
above a threshold. Until now the harness only had a hand-supplied
surprise vector — fine for unit tests, useless for real agent traces.

This module supplies real per-token surprise scoring backed by a small
language model (Qwen3-1.7B or, when memory is tight, Qwen3-0.6B), with
a graceful fallback chain so the harness never silently degrades:

1. ``vllm`` — preferred for production runs. Loads ``Qwen/Qwen3-1.7B``
   by default and asks for ``prompt_logprobs`` so we get the negative
   log-probability of every token in the prompt without generating
   anything new.
2. ``transformers`` — used when ``vllm`` is unavailable but a CUDA /
   MPS device + ``torch`` are. Performs a single forward pass and reads
   token-level cross-entropy.
3. ``heuristic`` — always available; computes surprise from in-context
   token-frequency (Zipf-style). Deterministic, zero dependencies, used
   in CI and on the plain-CPU developer laptop. Never reported as a
   real LM probability; the backend name is always recorded in
   :attr:`SurpriseProfile.backend`.

The factory :func:`make_surprise_scorer` resolves the backend based on
the ``CEH_SURPRISE_BACKEND`` environment variable
(``auto`` | ``vllm`` | ``hf`` | ``heuristic``). It never raises if a
model backend is missing — it always returns *some* scorer and emits a
log line so the caller can confirm what they got.

Boundaries
----------
:func:`event_boundaries_from_text` is a one-shot convenience that
combines ``score_text`` + smoothing + threshold into a list of token
indices, suitable for direct use by
:class:`EventBoundaryDetector.detect_from_surprises`.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenSurprise:
    """Surprise for a single token."""

    token: str
    token_id: int
    nll: float  # negative log probability, base e
    normalised: float  # NLL squashed to [0, 1] for threshold comparisons


@dataclass(frozen=True)
class SurpriseProfile:
    """Per-token surprise scores for one piece of text."""

    backend: str
    model: str | None
    tokens: list[TokenSurprise]

    @property
    def normalised(self) -> list[float]:
        return [t.normalised for t in self.tokens]

    @property
    def nll(self) -> list[float]:
        return [t.nll for t in self.tokens]


class SurpriseScorer(Protocol):
    backend_name: str
    model_name: str | None

    def score_text(self, text: str) -> SurpriseProfile: ...


def _squash_nll(nll: float, *, scale: float = 6.0) -> float:
    """Map an NLL value (>=0) to [0, 1] via a saturating curve.

    NLL of 0 -> surprise 0.0
    NLL of `scale` -> surprise ~= 0.63
    NLL of 3*scale -> surprise ~= 0.95
    """
    if nll <= 0.0:
        return 0.0
    return 1.0 - math.exp(-nll / scale)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _whitespace_tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


# Heuristic backend
class HeuristicSurpriseScorer:
    """Zipf-style in-context surprise. Deterministic. No model required.

    Per-token surprise is ``-log2(P)`` where ``P`` is the empirical
    relative frequency of the token in the (already-seen prefix +1)
    of the input. Repeated tokens decay toward low surprise; the first
    occurrence of a rare token registers a spike. This is *not* a real
    language-model surprise — it is a tractable proxy used for tests,
    CI, and offline mode. Treat boundaries it detects as illustrative,
    not authoritative.
    """

    backend_name = "heuristic"
    model_name = None

    def __init__(self, *, scale: float = 4.0) -> None:
        self._scale = scale

    def score_text(self, text: str) -> SurpriseProfile:
        if not text:
            return SurpriseProfile(self.backend_name, None, [])
        tokens = _whitespace_tokenise(text)
        if not tokens:
            return SurpriseProfile(self.backend_name, None, [])
        counts: Counter[str] = Counter()
        out: list[TokenSurprise] = []
        for i, tok in enumerate(tokens):
            counts[tok] += 1
            seen = i + 1
            p = counts[tok] / seen
            nll = -math.log(p)  # natural log so it composes with squash
            out.append(
                TokenSurprise(
                    token=tok,
                    token_id=hash(tok) & 0xFFFF,
                    nll=nll,
                    normalised=_squash_nll(nll, scale=self._scale),
                )
            )
        return SurpriseProfile(self.backend_name, None, out)


# vLLM backend
class VLLMSurpriseScorer:
    """Real per-token surprise from a Qwen3 model via vLLM.

    Preferred backend for production. Uses ``LLM.generate`` with
    ``SamplingParams(prompt_logprobs=1, max_tokens=1)`` and reads the
    log-probability of each *prompt* token from the returned object.
    The first token has no preceding context and is given an NLL of
    0.0 so it never dominates downstream smoothing.

    Construction lazily imports ``vllm`` and ``transformers``.
    """

    backend_name = "vllm"

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-1.7B",
        fallback_model: str = "Qwen/Qwen3-0.6B",
        max_model_len: int | None = None,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.45,
        scale: float = 6.0,
    ) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only when vllm absent
            raise RuntimeError(
                "vllm not installed; install vllm or use HFSurpriseScorer/HeuristicSurpriseScorer"
            ) from exc

        self.model_name = model
        self._fallback_model = fallback_model
        self._scale = scale
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._gpu_memory_utilization = gpu_memory_utilization
        self._llm = None  # lazy-init so import side-effects stay light

    def _ensure_llm(self):
        if self._llm is not None:
            return
        from vllm import LLM

        attempts = [self.model_name]
        if self._fallback_model and self._fallback_model != self.model_name:
            attempts.append(self._fallback_model)

        last_exc: Exception | None = None
        for candidate in attempts:
            try:
                logger.info("Loading vLLM model %s", candidate)
                kwargs = {
                    "model": candidate,
                    "dtype": self._dtype,
                    "gpu_memory_utilization": self._gpu_memory_utilization,
                }
                if self._max_model_len is not None:
                    kwargs["max_model_len"] = self._max_model_len
                self._llm = LLM(**kwargs)
                self.model_name = candidate
                return
            except Exception as exc:
                logger.warning("vLLM load failed for %s: %s", candidate, exc)
                last_exc = exc
        raise RuntimeError(f"vLLM failed to load any of {attempts}") from last_exc

    def score_text(self, text: str) -> SurpriseProfile:
        self._ensure_llm()
        from vllm import SamplingParams

        sampling = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0.0)
        out = self._llm.generate([text], sampling, use_tqdm=False)[0]
        prompt_logprobs = out.prompt_logprobs or []
        prompt_token_ids = out.prompt_token_ids or []
        tokenizer = self._llm.get_tokenizer()

        tokens: list[TokenSurprise] = []
        for idx, (tok_id, lp_dict) in enumerate(zip(prompt_token_ids, prompt_logprobs)):
            tok_text = tokenizer.decode([tok_id])
            if idx == 0 or lp_dict is None:
                nll = 0.0
            else:
                # vLLM returns either a plain float or {token_id: Logprob}
                if isinstance(lp_dict, dict):
                    entry = lp_dict.get(tok_id)
                    lp = getattr(entry, "logprob", entry) if entry is not None else 0.0
                else:
                    lp = float(lp_dict)
                nll = -float(lp)
            tokens.append(
                TokenSurprise(
                    token=tok_text,
                    token_id=int(tok_id),
                    nll=nll,
                    normalised=_squash_nll(nll, scale=self._scale),
                )
            )
        return SurpriseProfile(self.backend_name, self.model_name, tokens)


# Hugging Face transformers backend
class HFSurpriseScorer:
    """Per-token surprise via ``transformers`` cross-entropy.

    Slower than vLLM but does not require a vLLM-compatible install.
    Lazy-imports ``torch`` and ``transformers`` so unrelated code paths
    never pay the import cost.
    """

    backend_name = "hf"

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-0.6B",
        device: str | None = None,
        scale: float = 6.0,
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers/torch not installed; install transformers torch or use HeuristicSurpriseScorer"
            ) from exc
        self.model_name = model
        self._device = device
        self._scale = scale
        self._tokenizer = None
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        logger.info("Loading HF model %s on %s", self.model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

    def score_text(self, text: str) -> SurpriseProfile:
        self._ensure_model()
        import torch
        import torch.nn.functional as F

        ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)
        with torch.no_grad():
            logits = self._model(ids).logits[0]  # (seq, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        seq = ids[0]
        tokens: list[TokenSurprise] = []
        for i, tok_id in enumerate(seq.tolist()):
            tok_text = self._tokenizer.decode([tok_id])
            if i == 0:
                nll = 0.0
            else:
                lp = log_probs[i - 1, tok_id].item()
                nll = -float(lp)
            tokens.append(
                TokenSurprise(
                    token=tok_text,
                    token_id=int(tok_id),
                    nll=nll,
                    normalised=_squash_nll(nll, scale=self._scale),
                )
            )
        return SurpriseProfile(self.backend_name, self.model_name, tokens)


# Factory and helpers
def make_surprise_scorer(
    backend: str | None = None,
    *,
    model: str | None = None,
    fallback_model: str | None = None,
) -> SurpriseScorer:
    """Resolve a surprise scorer based on availability and env config.

    The chosen backend depends on the resolved value of ``backend``:

    * ``auto`` (default) — try ``vllm`` first, then ``hf``, then
      ``heuristic``. Logs which one was selected.
    * ``vllm`` / ``hf`` / ``heuristic`` — forces that backend.

    Environment variable ``CEH_SURPRISE_BACKEND`` overrides ``backend``
    when neither the argument nor a non-default backend was passed.
    """
    chosen = (backend or os.getenv("CEH_SURPRISE_BACKEND") or "auto").lower()

    if chosen == "heuristic":
        logger.info("Surprise backend: heuristic (forced)")
        return HeuristicSurpriseScorer()

    if chosen in ("vllm", "auto"):
        try:
            kwargs = {}
            if model:
                kwargs["model"] = model
            if fallback_model:
                kwargs["fallback_model"] = fallback_model
            scorer = VLLMSurpriseScorer(**kwargs)
            logger.info("Surprise backend: vllm (model=%s)", scorer.model_name)
            return scorer
        except RuntimeError as exc:
            if chosen == "vllm":
                raise
            logger.info("vLLM unavailable (%s); falling back", exc)

    if chosen in ("hf", "auto"):
        try:
            kwargs = {}
            if model:
                kwargs["model"] = model
            scorer = HFSurpriseScorer(**kwargs)
            logger.info("Surprise backend: hf (model=%s)", scorer.model_name)
            return scorer
        except RuntimeError as exc:
            if chosen == "hf":
                raise
            logger.info("HF transformers unavailable (%s); falling back to heuristic", exc)

    logger.info("Surprise backend: heuristic (auto-selected)")
    return HeuristicSurpriseScorer()


def smooth(values: Iterable[float], *, window: int = 5) -> list[float]:
    """Centered moving-average smoothing for a sequence."""
    seq = list(values)
    if not seq:
        return []
    out: list[float] = []
    for i in range(len(seq)):
        lo = max(0, i - window // 2)
        hi = min(len(seq), i + window // 2 + 1)
        chunk = seq[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def event_boundaries_from_text(
    text: str,
    *,
    scorer: SurpriseScorer | None = None,
    threshold: float = 0.75,
    smoothing_window: int = 5,
) -> list[int]:
    """End-to-end: text -> per-token surprise -> smoothed -> boundary indices."""
    scorer = scorer or make_surprise_scorer()
    profile = scorer.score_text(text)
    smoothed = smooth(profile.normalised, window=smoothing_window)
    return [i for i, v in enumerate(smoothed) if v >= threshold]
