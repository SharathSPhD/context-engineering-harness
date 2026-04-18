from src.avacchedaka.store import ContextStore
from src.avacchedaka.query import AvacchedakaQuery
from src.compaction.detector import EventBoundaryDetector


class BoundaryTriggeredCompactor:
    """Compacts context at event boundaries rather than fixed token thresholds.

    Inspired by hippocampal sharp-wave ripple replay during quiescent periods
    (Buzsaki 2015): consolidation happens at natural cognitive boundaries,
    not arbitrarily.
    """

    def __init__(self, store: ContextStore, compress_threshold: float = 0.3):
        self.store = store
        self.compress_threshold = compress_threshold

    def compact_at_boundary(self, qualificand: str = "", task_context: str = "") -> list[str]:
        """Compress low-precision elements at a detected event boundary.

        Args:
            qualificand: Optional avacchedaka qualificand scope for targeted compaction.
            task_context: Optional condition string to scope which elements to compact.
        """
        return self.store.compress(self.compress_threshold)

    def threshold_compact(self, token_count: int, token_threshold: int, qualificand: str = "") -> list[str]:
        """Baseline: compress when token count exceeds a fixed threshold.

        Args:
            qualificand: Optional avacchedaka qualificand to scope the compaction.
        """
        if token_count >= token_threshold:
            return self.store.compress(self.compress_threshold)
        return []


class BoundaryTriggeredSession:
    """Manages a session that compacts at event boundaries."""

    def __init__(
        self,
        store: ContextStore,
        detector: EventBoundaryDetector,
        compress_threshold: float = 0.3,
    ):
        self.store = store
        self.detector = detector
        self.compactor = BoundaryTriggeredCompactor(store, compress_threshold)
        self.compaction_events: list[dict] = []

    def process_surprises(self, surprises: list[float], step: int = 0) -> list[str]:
        """Check for boundaries in surprise sequence; compact if found."""
        boundaries = self.detector.detect_from_surprises(surprises)
        if boundaries:
            compressed = self.compactor.compact_at_boundary()
            self.compaction_events.append({
                "step": step,
                "boundary_indices": boundaries,
                "compressed_ids": compressed,
            })
            return compressed
        return []
