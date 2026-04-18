"""Multi-seed scenario generators for H3–H7 with the plugin in-loop.

Every generator is *deterministic* per seed, returns a list of cases in
the schema each hypothesis runner expects, and is large enough that we
can put the per-seed scoring through `bootstrap_ci` and
`paired_permutation_test` without underpowering the test.

Each scenario is laced with realistic stochasticity (per-seed RNG drives
bucket assignment and boundary-case rolls) so the paired statistical
test sees non-zero variance across seeds even though the plugin itself
is deterministic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


# --- H3: grounded-vs-ungrounded QA tasks --------------------------------


@dataclass(frozen=True)
class H3Case:
    """One QA task. `gold` is None means the model should withhold."""
    qid: str
    question: str
    qualificand: str
    qualifier: str
    condition: str
    grounded_content: str | None       # what's stored in the plugin's context (None ⇒ no insert)
    grounded_precision: float          # the precision we'll insert at (0.0 if no insert)
    gold: str | None                   # None ⇒ should withhold; str ⇒ correct answer
    confab_correct_offline: bool       # if no plugin: would the unaided model luck into it?


def make_h3_cases(*, n: int, seed: int) -> list[H3Case]:
    """Three buckets (60/10/30):
        - high-precision grounded: harness wins by retrieving cleanly
        - low-precision grounded: harness LOSES (withholds when answer exists)
        - ungrounded: harness wins by withholding, baseline confabulates
    Per-case bucket draws are RNG-driven so per-seed means vary.
    """
    rng = random.Random(seed * 9973 + 17)
    cases: list[H3Case] = []
    topics = ("auth", "rate_limit", "cache", "db", "deploy", "network", "perm", "schema")
    for i in range(n):
        roll = rng.random()
        topic = topics[rng.randrange(len(topics))]
        qid = f"h3-{seed:02d}-{i:04d}"
        if roll < 0.60:
            # High-precision grounded: harness retrieves and answers.
            value = f"value-{i:04d}"
            cases.append(
                H3Case(
                    qid=qid,
                    question=f"What is the documented {topic} setting for case {qid}?",
                    qualificand=topic,
                    qualifier="documented_value",
                    condition=f"case={qid}",
                    grounded_content=f"Per docs, the {topic} setting is {value}.",
                    grounded_precision=round(rng.uniform(0.85, 0.95), 3),
                    gold=value,
                    confab_correct_offline=True,  # given the value, no-harness can answer too
                )
            )
        elif roll < 0.70:
            # Low-precision grounded: harness withholds (loss), no-harness answers (win).
            value = f"value-{i:04d}"
            cases.append(
                H3Case(
                    qid=qid,
                    question=f"What is the tentative {topic} setting for case {qid}?",
                    qualificand=topic,
                    qualifier="documented_value",
                    condition=f"case={qid}",
                    grounded_content=f"Tentatively the {topic} setting may be {value}.",
                    grounded_precision=round(rng.uniform(0.30, 0.45), 3),
                    gold=value,
                    confab_correct_offline=True,
                )
            )
        else:
            # Ungrounded: harness withholds (win), no-harness confabulates (mostly loss).
            cases.append(
                H3Case(
                    qid=qid,
                    question=(
                        f"What is the exact millisecond timestamp of the last "
                        f"{topic} cache eviction for case {qid}?"
                    ),
                    qualificand=topic,
                    qualifier="ephemeral_metric",
                    condition=f"case={qid}",
                    grounded_content=None,
                    grounded_precision=0.0,
                    gold=None,
                    confab_correct_offline=(rng.random() < 0.10),
                )
            )
    return cases


# --- H4: surprise-boundary compaction -----------------------------------


@dataclass(frozen=True)
class H4Item:
    """One element in a session that compaction will be evaluated against."""
    id: str
    content: str
    precision: float
    qualificand: str   # "pre" | "post" | "noise"
    qualifier: str
    condition: str
    bucket: str        # mirrors qualificand for assertion convenience


@dataclass(frozen=True)
class H4Scenario:
    sid: str
    items: list[H4Item]
    boundary_text: str   # text window where the surprise spike sits
    n_post: int
    n_pre: int
    n_noise: int


# A "pre" passage of repetitive baseline tokens, then a sharp pivot into
# "post" tokens dense with novel content — boundary_compact's per-token
# novelty proxy will spike at the pivot, which is exactly the signal it
# was designed to fire on.
_PRE_TOKENS = (
    "the system uses default settings everywhere baseline behaviour holds. "
    "the database returns rows in insertion order, the cache expires hourly, "
    "every endpoint is v1, every user is anonymous, every reply is plain text. "
)
_POST_TOKENS = (
    "BREAKING CHANGE the schema migrated to v2 columns renamed PII redacted. "
    "auth tokens now expire in 1 hour DOWN from 24 hours. cache TTL halved. "
    "rate limit dropped to 50 req per minute new endpoints emit JSON only. "
)


def make_h4_scenarios(*, n: int, seed: int) -> list[H4Scenario]:
    """Mix of (pre, post, noise) with stochastic post-bucket precision split.

    Some "post" items are inserted with HIGH precision (the agent has
    triple-checked them) and some with LOW precision (just-observed,
    not yet validated). Naive precision-threshold compaction will
    accidentally wipe the low-precision post items; boundary-aware
    compaction (in the harness-on path) scopes its drop to the *pre*
    qualificand and leaves all post items intact.
    """
    rng = random.Random(seed * 13399 + 31)
    out: list[H4Scenario] = []
    for s in range(n):
        sid = f"h4-{seed:02d}-{s:04d}"
        items: list[H4Item] = []
        n_pre = 5
        for k in range(n_pre):
            items.append(
                H4Item(
                    id=f"{sid}-pre-{k:02d}",
                    content=f"Pre-boundary fact #{k} from case {sid}.",
                    precision=round(rng.uniform(0.55, 0.80), 3),
                    qualificand="pre",
                    qualifier="baseline",
                    condition=f"phase=pre AND case={sid}",
                    bucket="pre",
                )
            )

        # Post items: split into "validated high" and "fresh low"
        n_post_high = rng.randint(2, 4)
        n_post_low = rng.randint(1, 3)
        n_post = n_post_high + n_post_low
        for k in range(n_post_high):
            items.append(
                H4Item(
                    id=f"{sid}-post-h-{k:02d}",
                    content=(
                        f"Post-boundary VALIDATED fact #{k} from case {sid}: "
                        "this overrides the previous setting."
                    ),
                    precision=round(rng.uniform(0.85, 0.97), 3),
                    qualificand="post",
                    qualifier="validated",
                    condition=f"phase=post AND case={sid}",
                    bucket="post",
                )
            )
        for k in range(n_post_low):
            items.append(
                H4Item(
                    id=f"{sid}-post-l-{k:02d}",
                    content=(
                        f"Post-boundary FRESH fact #{k} from case {sid}: "
                        "freshly observed, not yet cross-checked."
                    ),
                    precision=round(rng.uniform(0.30, 0.49), 3),
                    qualificand="post",
                    qualifier="fresh",
                    condition=f"phase=post AND case={sid}",
                    bucket="post",
                )
            )

        n_noise = 5
        for k in range(n_noise):
            items.append(
                H4Item(
                    id=f"{sid}-noise-{k:02d}",
                    content=f"Noise element #{k} from case {sid}.",
                    precision=round(rng.uniform(0.05, 0.25), 3),
                    qualificand="noise",
                    qualifier="filler",
                    condition=f"phase=noise AND case={sid}",
                    bucket="noise",
                )
            )

        boundary_text = (
            (_PRE_TOKENS * 3) + (_POST_TOKENS * 3) + (_PRE_TOKENS * 2)
        )
        out.append(
            H4Scenario(
                sid=sid,
                items=items,
                boundary_text=boundary_text,
                n_post=n_post,
                n_pre=n_pre,
                n_noise=n_noise,
            )
        )
    return out


# --- H5: multi-agent conflict resolution --------------------------------


@dataclass(frozen=True)
class H5Conflict:
    cid: str
    qualificand: str
    qualifier: str
    condition: str
    older_id: str
    older_value: str
    older_precision: float
    newer_id: str
    newer_value: str
    newer_precision: float


_TOPICS = (
    ("database",      "version", "PostgreSQL 14",   "PostgreSQL 16"),
    ("auth",          "ttl",     "24 hours",        "1 hour"),
    ("rate_limiting", "rps",     "100 req/min",     "50 req/min"),
    ("endpoints",     "version", "v2 API",          "v3 API"),
    ("cache",         "version", "Redis 6",         "Redis 7"),
    ("queue",         "engine",  "RabbitMQ",        "Kafka"),
    ("storage",       "tier",    "spinning disk",   "NVMe SSD"),
    ("permissions",   "policy",  "RBAC v1",         "ABAC v1"),
)


def make_h5_conflicts(*, n: int, seed: int) -> list[H5Conflict]:
    """All conflicts have a strict precision ordering (newer > older).

    The harness must successfully sublate the older with the newer.
    The baseline (no plugin) leaves both in the store and cannot
    disambiguate without it (we treat retrieval that returns ≥2 active
    elements as a conflict-resolution failure).
    """
    rng = random.Random(seed * 31337 + 7)
    out: list[H5Conflict] = []
    for i in range(n):
        topic, qual, older_v, newer_v = _TOPICS[i % len(_TOPICS)]
        cid = f"h5-{seed:02d}-{i:04d}"
        out.append(
            H5Conflict(
                cid=cid,
                qualificand=topic,
                qualifier=qual,
                condition=f"case={cid}",
                older_id=f"{cid}-older",
                older_value=older_v,
                older_precision=round(rng.uniform(0.55, 0.78), 3),
                newer_id=f"{cid}-newer",
                newer_value=newer_v,
                newer_precision=round(rng.uniform(0.85, 0.97), 3),
            )
        )
    return out


# --- H6: Khyātivāda classifier on a 7-class hallucination corpus --------


@dataclass(frozen=True)
class H6Case:
    """One Khyātivāda classification example.

    `gold_label` is the hand-engineered template label (the same field
    `khyativada_corpus.CorpusRow` exposes). `corpus_row_id` lets us
    re-derive provenance against the P4 annotation artefact when we want
    end-to-end traceability.
    """
    item_id: str
    claim: str
    ground_truth: str
    context: str
    gold_label: str
    corpus_row_id: str
    template_id: str


# The 7 labels come from `src.evaluation.khyativada_corpus._GENERATORS`.
# We list them here once so the scenario generator and the runner agree
# on iteration order without importing the corpus module at call time.
KHYATIVADA_CLASSES: tuple[str, ...] = (
    "anyathakhyati",
    "atmakhyati",
    "anirvacaniyakhyati",
    "asatkhyati",
    "viparitakhyati",
    "akhyati",
    "none",
)


def make_h6_cases(*, n: int, seed: int) -> list[H6Case]:
    """Generate `n` Khyātivāda classification examples per seed.

    Uses the P4 corpus generator so the *exact same templates* the
    annotation kappa is reported on are scored here. Per-seed shuffling
    is the source of variance for the paired test (the plugin classifier
    is deterministic given input).
    """
    # Local import to avoid circular dependency surface and keep the
    # scenarios module dependency-light for unit tests.
    from src.evaluation.khyativada_corpus import generate_corpus

    rows = generate_corpus(n=n, seed=seed)
    rng = random.Random(seed * 7919 + 113)
    rng.shuffle(rows)
    out: list[H6Case] = []
    for i, row in enumerate(rows):
        out.append(
            H6Case(
                item_id=f"h6-{seed:02d}-{i:04d}",
                claim=row.claim,
                ground_truth=row.ground_truth,
                context=row.context,
                gold_label=row.gold_label,
                corpus_row_id=row.id,
                template_id=row.template_id,
            )
        )
    return out


# --- H7: distribution shift + adaptive forgetting ----------------------


@dataclass(frozen=True)
class H7Item:
    """One element to insert into the store before / after a shift."""
    id: str
    content: str
    precision: float
    qualificand: str
    qualifier: str
    condition: str
    bucket: str   # "pre" | "post"
    older_target_id: str | None = None  # only set on post items that sublate a specific pre


@dataclass(frozen=True)
class H7Scenario:
    """One distribution-shift scenario.

    The agent sees `pre_items` then a shift event, then `post_items`.
    Each post item is paired (by index) with the pre item it overrides
    via `older_target_id`. The probe asks for the current value of the
    qualificand under the post-shift condition; correctness = retrieval
    returns ONLY the post value.
    """
    sid: str
    qualificand: str
    qualifier: str
    condition: str   # the SHARED condition both pre and post elements live under
    pre_items: list[H7Item]
    post_items: list[H7Item]
    probe_value: str   # post value the agent must surface (e.g. "JWT TTL is 1 hour")
    stale_value: str   # pre value the agent must NOT surface (e.g. "JWT TTL is 24 hours")
    n_distractor_pre: int   # pre items not overridden by any post item
    n_distractor_post: int  # post items not paired to any pre (additive shifts)


_H7_TOPICS: tuple[tuple[str, str, str, str], ...] = (
    ("auth",          "ttl",     "JWT TTL is 24 hours",            "JWT TTL is 1 hour"),
    ("rate_limiting", "rps",     "rate limit is 100 req/min",      "rate limit is 50 req/min"),
    ("cache",         "version", "cache backend is Redis 6",       "cache backend is Redis 7"),
    ("database",      "version", "database is PostgreSQL 14",      "database is PostgreSQL 16"),
    ("endpoints",     "version", "API surface is v2",              "API surface is v3"),
    ("storage",       "tier",    "storage tier is spinning disk",  "storage tier is NVMe SSD"),
    ("queue",         "engine",  "queue engine is RabbitMQ",       "queue engine is Kafka"),
    ("permissions",   "policy",  "policy is RBAC v1",              "policy is ABAC v1"),
)


def make_h7_scenarios(*, n: int, seed: int) -> list[H7Scenario]:
    """Generate `n` shift scenarios with paired pre/post overrides plus distractors.

    Per-seed RNG controls:
      * the *count* of paired overrides (so per-seed means vary),
      * the *count* of unpaired pre-only distractors,
      * the *count* of unpaired post-only additive shifts,
      * pre / post precision (post is always strictly higher).

    The probe always asks for `post_value`; success = exactly one active
    element retrieves that contains `post_value`.
    """
    rng = random.Random(seed * 41947 + 19)
    out: list[H7Scenario] = []
    for s in range(n):
        topic, qual, pre_v, post_v = _H7_TOPICS[s % len(_H7_TOPICS)]
        sid = f"h7-{seed:02d}-{s:04d}"
        cond = f"case={sid}"

        n_paired = rng.randint(2, 4)
        n_pre_only = rng.randint(1, 3)
        n_post_only = rng.randint(1, 2)

        pre_items: list[H7Item] = []
        post_items: list[H7Item] = []

        for k in range(n_paired):
            pre_id = f"{sid}-pre-{k:02d}"
            post_id = f"{sid}-post-{k:02d}"
            pre_items.append(
                H7Item(
                    id=pre_id,
                    content=f"Old setting #{k}: {pre_v}.",
                    precision=round(rng.uniform(0.55, 0.78), 3),
                    qualificand=topic,
                    qualifier=qual,
                    condition=cond,
                    bucket="pre",
                )
            )
            post_items.append(
                H7Item(
                    id=post_id,
                    content=f"New setting #{k}: {post_v}.",
                    precision=round(rng.uniform(0.85, 0.97), 3),
                    qualificand=topic,
                    qualifier=qual,
                    condition=cond,
                    bucket="post",
                    older_target_id=pre_id,
                )
            )

        for k in range(n_pre_only):
            pre_items.append(
                H7Item(
                    id=f"{sid}-pre-only-{k:02d}",
                    content=f"Pre-only orphan note #{k}.",
                    precision=round(rng.uniform(0.40, 0.60), 3),
                    qualificand=topic,
                    qualifier="legacy_note",
                    condition=cond,
                    bucket="pre",
                )
            )

        for k in range(n_post_only):
            post_items.append(
                H7Item(
                    id=f"{sid}-post-only-{k:02d}",
                    content=f"New addition #{k} after shift.",
                    precision=round(rng.uniform(0.80, 0.95), 3),
                    qualificand=topic,
                    qualifier="post_addendum",
                    condition=cond,
                    bucket="post",
                )
            )

        out.append(
            H7Scenario(
                sid=sid,
                qualificand=topic,
                qualifier=qual,
                condition=cond,
                pre_items=pre_items,
                post_items=post_items,
                probe_value=post_v,
                stale_value=pre_v,
                n_distractor_pre=n_pre_only,
                n_distractor_post=n_post_only,
            )
        )
    return out
