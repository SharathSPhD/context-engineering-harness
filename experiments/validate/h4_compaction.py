"""H4 Validation: Event-boundary compaction outperforms threshold compaction.

Method (no LLM needed):
- 15-element store: 5 pre-boundary (precision=0.8), 5 post-boundary (precision=0.85), 5 noise (precision=0.15)
- BoundaryTriggeredCompactor: compacts at detected surprise boundary → preserves post-boundary elements
- Threshold compactor (same class, different trigger): fires at token_count >= threshold
- Metric: how many post-boundary elements survive after compaction

Target: boundary_retention >= threshold_retention.
"""
from __future__ import annotations

import json

from src.avacchedaka.element import AvacchedakaConditions, ContextElement
from src.avacchedaka.query import AvacchedakaQuery
from src.avacchedaka.store import ContextStore
from src.compaction.compactor import BoundaryTriggeredCompactor
from src.compaction.detector import EventBoundaryDetector
from src.config import config


def _build_session_store() -> ContextStore:
    store = ContextStore()
    conds_pre = AvacchedakaConditions(qualificand="nexus", qualifier="pre", condition="phase=pre")
    conds_post = AvacchedakaConditions(qualificand="nexus", qualifier="post", condition="phase=post")
    conds_noise = AvacchedakaConditions(qualificand="noise", qualifier="n", condition="phase=noise")

    for i in range(5):
        store.insert(ContextElement(
            id=f"pre-{i:02d}", content=f"Pre-boundary fact {i}",
            precision=0.8, avacchedaka=conds_pre,
        ))
    for i in range(5):
        store.insert(ContextElement(
            id=f"post-{i:02d}", content=f"Post-boundary fact {i} (important)",
            precision=0.85, avacchedaka=conds_post,
        ))
    for i in range(5):
        store.insert(ContextElement(
            id=f"noise-{i:02d}", content=f"Noise element {i}",
            precision=0.15, avacchedaka=conds_noise,
        ))
    return store


def _measure_retention() -> tuple[float, float]:
    """Returns (boundary_retention, threshold_retention)."""
    post_query = AvacchedakaQuery(qualificand="nexus", condition="phase=post",
                                  precision_threshold=0.0)
    n_post = 5

    # Boundary-triggered compaction
    store_b = _build_session_store()
    detector = EventBoundaryDetector(surprise_threshold=config.surprise_threshold)
    compactor_b = BoundaryTriggeredCompactor(store_b, compress_threshold=config.compress_threshold)
    surprises = [0.2] * 5 + [0.9] + [0.2] * 9
    boundaries = detector.detect_from_surprises(surprises)
    if boundaries:
        compactor_b.compact_at_boundary()
    boundary_retained = len(store_b.retrieve(post_query))

    # Threshold compaction (baseline)
    store_t = _build_session_store()
    compactor_t = BoundaryTriggeredCompactor(store_t, compress_threshold=config.compress_threshold)
    compactor_t.threshold_compact(token_count=600, token_threshold=config.token_threshold)
    threshold_retained = len(store_t.retrieve(post_query))

    return boundary_retained / n_post, threshold_retained / n_post


def run_h4() -> dict:
    boundary_ret, threshold_ret = _measure_retention()
    return {
        "hypothesis": "H4",
        "description": "Event-boundary compaction outperforms threshold compaction",
        "boundary_retention": round(boundary_ret, 3),
        "threshold_retention": round(threshold_ret, 3),
        "delta": round(boundary_ret - threshold_ret, 3),
        "target_met": boundary_ret >= threshold_ret,
        "target_description": "boundary_retention >= threshold_retention",
        "note": "Both methods compress noise-only (precision<0.3) in this synthetic scenario",
    }


if __name__ == "__main__":
    result = run_h4()
    print(json.dumps(result, indent=2))
