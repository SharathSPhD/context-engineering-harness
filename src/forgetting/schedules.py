import dataclasses
from datetime import datetime
from src.avacchedaka.store import ContextStore


class NoForgetting:
    """Baseline: retain all elements indefinitely."""

    def __init__(self, store: ContextStore):
        self.store = store

    def apply(self) -> list[str]:
        return []


class FixedCompaction:
    """Baseline: keep only the N most recent non-sublated elements."""

    def __init__(self, store: ContextStore, keep_newest: int = 50):
        self.store = store
        self.keep_newest = keep_newest

    def apply(self) -> list[str]:
        elements = sorted(
            [e for e in self.store._elements.values() if e.sublated_by is None],
            key=lambda e: e.timestamp,
            reverse=True,
        )
        to_remove = elements[self.keep_newest:]
        removed = []
        for e in to_remove:
            self.store._elements[e.id] = dataclasses.replace(e, precision=0.0)
            removed.append(e.id)
        return removed


class RecencyWeightedForgetting:
    """Decay precision based on age; remove elements that decay below threshold."""

    def __init__(self, store: ContextStore, decay_factor: float = 0.9):
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError("decay_factor must be in (0, 1]")
        self.store = store
        self.decay_factor = decay_factor

    def apply(self) -> list[str]:
        now = datetime.utcnow()
        removed = []
        for eid, e in list(self.store._elements.items()):
            if e.sublated_by is not None:
                continue
            age_hours = (now - e.timestamp).total_seconds() / 3600
            decayed = e.precision * (self.decay_factor ** age_hours)
            if decayed < 0.3:
                self.store._elements[eid] = dataclasses.replace(e, precision=0.0)
                removed.append(eid)
        return removed


class RewardWeightedForgetting:
    """Retain elements with high task-relevance salience; remove the rest."""

    def __init__(self, store: ContextStore, reward_key: str = "task_relevance", keep_threshold: float = 0.5):
        self.store = store
        self.reward_key = reward_key
        self.keep_threshold = keep_threshold

    def apply(self) -> list[str]:
        removed = []
        for eid, e in list(self.store._elements.items()):
            if e.sublated_by is not None:
                continue
            reward = e.salience.get(self.reward_key, e.precision)
            if reward < self.keep_threshold:
                self.store._elements[eid] = dataclasses.replace(e, precision=0.0)
                removed.append(eid)
        return removed


class BadhaFirstForgetting:
    """Clear sublated (cancelled) elements first - badha principle.

    Analogous to Richards & Frankland (2017): neurogenesis-driven clearance
    targets the most outdated/contradicted traces first.
    """

    def __init__(self, store: ContextStore):
        self.store = store

    def apply(self) -> list[str]:
        removed = []
        for eid, e in list(self.store._elements.items()):
            if e.sublated_by is not None:
                self.store._elements[eid] = dataclasses.replace(e, precision=0.0)
                removed.append(eid)
        return removed
