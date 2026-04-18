"""NexusAPI synthetic validation corpus.

Scenario: a fictional Python web service (NexusAPI) whose documentation
undergoes controlled shifts in auth policy (JWT expiry 24h→1h),
database version (PG14→PG16), and rate limiting (100→50 req/min).

This corpus exercises all 7 hypotheses with real-world-like conflicts,
schema congruence variations, and event boundaries.
"""
from __future__ import annotations

from src.avacchedaka.element import AvacchedakaConditions, ContextElement
from src.avacchedaka.store import ContextStore

# ── Document corpus ──────────────────────────────────────────────────────────

NEXUSAPI_DOCS: dict[str, dict] = {
    # --- AUTH (shifts from 24h to 1h expiry) ---
    "auth_pre": {
        "id": "auth-pre-001",
        "content": "JWT tokens expire after 24 hours. Refresh tokens are valid for 30 days.",
        "domain": "web_security",
        "precision": 0.88,
        "phase": "pre_shift",
        "qualificand": "auth",
        "qualifier": "expiry",
        "condition": "task_type=code_review AND phase=pre_shift",
    },
    "auth_post": {
        "id": "auth-post-001",
        "content": "JWT tokens now expire after 1 hour (policy updated 2026-03). Refresh window is 50 minutes.",
        "domain": "web_security",
        "precision": 0.95,
        "phase": "post_shift",
        "qualificand": "auth",
        "qualifier": "expiry",
        "condition": "task_type=code_review AND phase=post_shift",
    },
    # --- DATABASE (PG14 → PG16) ---
    "db_pre": {
        "id": "db-pre-001",
        "content": "NexusAPI uses PostgreSQL 14 with two read replicas.",
        "domain": "infrastructure",
        "precision": 0.85,
        "phase": "pre_shift",
        "qualificand": "database",
        "qualifier": "version",
        "condition": "task_type=code_review AND phase=pre_shift",
    },
    "db_post": {
        "id": "db-post-001",
        "content": "Database upgraded to PostgreSQL 16 (migration completed 2026-02).",
        "domain": "infrastructure",
        "precision": 0.93,
        "phase": "post_shift",
        "qualificand": "database",
        "qualifier": "version",
        "condition": "task_type=code_review AND phase=post_shift",
    },
    # --- RATE LIMITING (100 → 50 req/min) ---
    "rate_pre": {
        "id": "rate-pre-001",
        "content": "API rate limit: 100 requests per minute per user.",
        "domain": "infrastructure",
        "precision": 0.87,
        "phase": "pre_shift",
        "qualificand": "rate_limiting",
        "qualifier": "limit",
        "condition": "task_type=code_review AND phase=pre_shift",
    },
    "rate_post": {
        "id": "rate-post-001",
        "content": "Rate limit reduced to 50 requests per minute after DDoS incident (2026-01).",
        "domain": "infrastructure",
        "precision": 0.92,
        "phase": "post_shift",
        "qualificand": "rate_limiting",
        "qualifier": "limit",
        "condition": "task_type=code_review AND phase=post_shift",
    },
    # --- STABLE DOCS ---
    "password_policy": {
        "id": "pwd-001",
        "content": "Passwords must be minimum 12 characters with uppercase, lowercase, digit, and symbol.",
        "domain": "web_security",
        "precision": 0.97,
        "phase": "stable",
        "qualificand": "password",
        "qualifier": "policy",
        "condition": "task_type=code_review",
    },
    "endpoint_docs": {
        "id": "ep-001",
        "content": "POST /api/v2/auth/token returns a JWT. GET /api/v2/users/:id returns user profile.",
        "domain": "web_security",
        "precision": 0.90,
        "phase": "stable",
        "qualificand": "endpoints",
        "qualifier": "paths",
        "condition": "task_type=code_review",
    },
    "csrf_docs": {
        "id": "csrf-001",
        "content": "CSRF tokens are single-use random values bound to the session cookie.",
        "domain": "web_security",
        "precision": 0.91,
        "phase": "stable",
        "qualificand": "csrf",
        "qualifier": "mechanism",
        "condition": "task_type=code_review",
    },
    "cors_docs": {
        "id": "cors-001",
        "content": "CORS preflight uses OPTIONS method. Allowed origins are configured in settings.py.",
        "domain": "web_security",
        "precision": 0.89,
        "phase": "stable",
        "qualificand": "cors",
        "qualifier": "preflight",
        "condition": "task_type=code_review",
    },
}

# ── Question-answer pairs ─────────────────────────────────────────────────────

QA_PAIRS: list[dict] = [
    {
        "question": "How long are JWT tokens valid in NexusAPI?",
        "pre_shift_answer": "24 hours",
        "post_shift_answer": "1 hour",
        "qualificand": "auth",
        "condition": "task_type=code_review",
    },
    {
        "question": "What database version does NexusAPI use?",
        "pre_shift_answer": "PostgreSQL 14",
        "post_shift_answer": "PostgreSQL 16",
        "qualificand": "database",
        "condition": "task_type=code_review",
    },
    {
        "question": "What is the API rate limit for NexusAPI?",
        "pre_shift_answer": "100 requests per minute",
        "post_shift_answer": "50 requests per minute",
        "qualificand": "rate_limiting",
        "condition": "task_type=code_review",
    },
]

# Distractors for schema-congruence tests (unrelated to web security)
INCONGRUENT_DISTRACTORS: list[str] = [
    "The Amazon rainforest covers approximately 5.5 million km².",
    "Piano has 88 keys in standard concert configuration.",
    "Shakespeare wrote 37 plays and 154 sonnets.",
    "Water boils at 100°C at sea level.",
    "Beethoven was deaf when he composed his Ninth Symphony.",
    "The Eiffel Tower was completed in 1889.",
    "Honey never spoils due to its low moisture content.",
    "The Great Wall of China is not visible from space.",
]


# ── Store builders ────────────────────────────────────────────────────────────

def _make_element(doc: dict) -> ContextElement:
    return ContextElement(
        id=doc["id"],
        content=doc["content"],
        precision=doc["precision"],
        avacchedaka=AvacchedakaConditions(
            qualificand=doc["qualificand"],
            qualifier=doc["qualifier"],
            condition=doc["condition"],
        ),
        provenance="nexusapi_corpus",
    )


def build_pre_shift_store() -> ContextStore:
    """Store populated with pre-shift NexusAPI documentation only."""
    store = ContextStore()
    for key, doc in NEXUSAPI_DOCS.items():
        if doc["phase"] in ("pre_shift", "stable"):
            store.insert(_make_element(doc))
    return store


def build_post_shift_store(store: ContextStore) -> None:
    """Mutate store in-place: sublate pre-shift elements, insert post-shift.

    Implements the bādha principle — pre-shift elements are sublated (not deleted).
    """
    post_shift_map: dict[str, str] = {}
    for key, doc in NEXUSAPI_DOCS.items():
        if doc["phase"] == "post_shift":
            elem = _make_element(doc)
            store.insert(elem)
            post_shift_map[doc["qualificand"]] = elem.id

    for key, doc in NEXUSAPI_DOCS.items():
        if doc["phase"] == "pre_shift":
            q = doc["qualificand"]
            if q in post_shift_map:
                try:
                    store.sublate(doc["id"], post_shift_map[q])
                except KeyError:
                    pass
