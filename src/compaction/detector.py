class EventBoundaryDetector:
    """Detects event boundaries in agent sessions.

    Inspired by Zacks et al. (2007) event segmentation theory: boundaries occur
    at prediction-failure moments, not at arbitrary token thresholds.
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
