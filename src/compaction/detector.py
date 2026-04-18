from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.compaction.surprise import SurpriseProfile, SurpriseScorer


class EventBoundaryDetector:
    """Detects event boundaries in agent sessions.

    Inspired by Zacks et al. (2007) event segmentation theory: boundaries occur
    at prediction-failure moments, not at arbitrary token thresholds.

    The detector is purely numeric — it consumes a per-token surprise vector
    and returns boundary indices. To turn raw text into surprise vectors,
    pair it with a :class:`src.compaction.surprise.SurpriseScorer` (vLLM,
    HF transformers, or a heuristic fallback) via :meth:`detect_in_text`.
    """

    def __init__(self, surprise_threshold: float = 0.75):
        if not 0.0 < surprise_threshold <= 1.0:
            raise ValueError("surprise_threshold must be in (0, 1]")
        self.surprise_threshold = surprise_threshold

    def detect_from_surprises(self, surprises: list[float]) -> list[int]:
        """Return indices where per-token surprise exceeds threshold (event boundaries)."""
        return [i for i, s in enumerate(surprises) if s >= self.surprise_threshold]

    def detect_from_signals(self, task_switch: bool, surprise_spike: bool) -> bool:
        """Returns True if either signal indicates a boundary."""
        return task_switch or surprise_spike

    def window_average_surprise(self, surprises: list[float], window_size: int = 5) -> list[float]:
        """Sliding window average for smoothed boundary detection."""
        if not surprises:
            return []
        averaged = []
        for i in range(len(surprises)):
            start = max(0, i - window_size // 2)
            end = min(len(surprises), i + window_size // 2 + 1)
            averaged.append(sum(surprises[start:end]) / (end - start))
        return averaged

    def detect_in_text(
        self,
        text: str,
        scorer: "SurpriseScorer | None" = None,
        *,
        smoothing_window: int = 5,
    ) -> tuple[list[int], "SurpriseProfile"]:
        """Score ``text`` with ``scorer`` and return ``(boundaries, profile)``.

        ``boundaries`` are token indices where the smoothed surprise crosses
        :attr:`surprise_threshold`. ``profile`` is the raw
        :class:`SurpriseProfile` so callers can audit which backend / model
        produced the numbers (this is the difference between a real LM
        boundary and the heuristic-fallback boundary).
        """
        from src.compaction.surprise import make_surprise_scorer, smooth

        scorer = scorer or make_surprise_scorer()
        profile = scorer.score_text(text)
        smoothed = smooth(profile.normalised, window=smoothing_window)
        boundaries = [i for i, v in enumerate(smoothed) if v >= self.surprise_threshold]
        return boundaries, profile
