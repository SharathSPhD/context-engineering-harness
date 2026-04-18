"""Benchmark abstract base + value types.

Every benchmark adapter (RULER, HELMET, NoCha, HaluEval, TruthfulQA,
FACTS-Grounding, SWE-bench Verified, the live case study, …) is a subclass
of `BenchmarkAdapter`. The runner only ever sees this abstract surface.

Design constraints:
  - Adapters must be deterministic for a given (example, model_output).
  - `run_example` returns the raw model output (no scoring); scoring is done
    inside `score(...)` so we can rescore archived raw outputs without re-billing.
  - Adapters never reach for env vars; the runner injects the model caller.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Protocol


@dataclass
class ModelOutput:
    """What the model produced for one example. Adapter-agnostic."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class ModelCaller(Protocol):
    """Pluggable callable. Implementations: CLIBudgetScheduler.invoke (dev),
    raw subprocess wrapper (plugin smoke), or a fixture function (tests)."""

    def __call__(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        system: str = "",
        seed: int | None = None,
    ) -> ModelOutput: ...


@dataclass
class BenchmarkExample:
    """One input row from a benchmark."""
    id: str
    prompt: str
    ground_truth: str | dict | list
    metadata: dict = field(default_factory=dict)
    context: str = ""  # optional long-context payload


@dataclass
class BenchmarkResult:
    """One model outcome on one example. Adapter sets `score` and `correct`."""
    example_id: str
    condition: str  # "treatment" | "baseline" or any free-form label
    seed: int
    prediction: str
    score: float
    correct: bool
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    """Aggregate of N BenchmarkResult under one (adapter, model, condition)."""
    adapter_name: str
    model: str
    condition: str
    seed: int
    results: list[BenchmarkResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.results]

    @property
    def mean_score(self) -> float:
        s = self.scores
        return sum(s) / len(s) if s else 0.0

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)


class BenchmarkAdapter(ABC):
    """Abstract base for every benchmark adapter."""

    name: str = ""
    requires_long_context: bool = False
    license_note: str = ""

    @abstractmethod
    def load_examples(self, *, n: int | None = None, seed: int = 0) -> list[BenchmarkExample]:
        """Return up to `n` examples, deterministic per `seed`."""

    @abstractmethod
    def render_prompt(self, example: BenchmarkExample, *, condition: str) -> str:
        """Return the prompt string sent to the model under a given condition."""

    @abstractmethod
    def score(self, example: BenchmarkExample, output: ModelOutput) -> tuple[float, bool, str]:
        """Score one model output against ground truth.

        Returns: (numeric_score, correct_bool, normalized_prediction_text)
        """

    def system_prompt(self, *, condition: str) -> str:
        """Optional system prompt; default empty. Override per adapter."""
        return ""

    def iter_examples(
        self, *, n: int | None = None, seed: int = 0
    ) -> Iterator[BenchmarkExample]:
        yield from self.load_examples(n=n, seed=seed)
