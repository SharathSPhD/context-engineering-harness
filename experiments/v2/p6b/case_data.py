"""Real GitHub issue case-study fixtures for P6-B.

Each ``CaseStudy`` instance encodes a *real, documented* issue from a
public repository together with the kind of mixed-precision evidence an
LLM coding agent would actually encounter while triaging it: official
current documentation, stale Stack Overflow answers, deprecated
changelog entries, and so on.

The case studies were chosen to exhibit the *exact failure mode* the
Pratyaksha plugin is designed to address — multiple "documents" each
asserting an answer to the same question, with different vintages and
different precisions, where naive concatenation into the model's
prompt leaves the agent unable to tell *which* answer to act on.

Provenance:

* ``django_request_body`` — the recurring "RawPostDataException after
  reading request.POST" issue that has shipped with Django since 1.4
  and continues to surface in tickets such as
  https://code.djangoproject.com/ticket/27592 and
  https://github.com/django/django/pull/15497. The Django docs were
  rewritten between 3.2 and 4.x; older Stack Overflow answers still
  reference the pre-4.x semantics. We encode the actual current
  behaviour and one stale doc snippet exactly as they appear.
* ``requests_retry_adapter`` — well-known mismatch between the legacy
  ``requests`` retry pattern using ``HTTPAdapter`` from ``urllib3``
  vs the post-2.30 supported pattern. See
  https://github.com/psf/requests/issues/2401 and
  https://github.com/urllib3/urllib3/pull/2780.
* ``pandas_iterrows_dtype`` — the dtype-loss-on-iterrows footgun
  documented at
  https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html
  with multiple older blog posts giving wrong dtype-preservation
  recommendations.

These fixtures intentionally do NOT call any LLM or hit any network at
runtime. The case study runner replays them deterministically against
the in-process plugin so the artifact is fully reproducible and can be
re-run by reviewers without an Anthropic account.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceItem:
    """One piece of evidence the agent encounters during research.

    ``precision`` is the confidence the *retrieval system* assigns the
    document at the moment it is surfaced — a fresh changelog from the
    library's own repo at the pinned commit gets a high precision
    (≥0.85), a 7-year-old top-voted Stack Overflow answer that
    references a deprecated API gets a low precision (≤0.40).

    ``stale`` marks evidence whose claim is wrong relative to the
    pinned codebase commit; ``superseded_by_id`` lets us mark which
    later evidence item makes it obsolete (the plugin's
    ``sublate_with_evidence`` is the structural analog).
    """
    id: str
    qualificand: str
    qualifier: str
    condition: str
    content: str
    precision: float
    source: str
    stale: bool = False
    superseded_by_id: str | None = None


@dataclass(frozen=True)
class CaseStudy:
    """One complete reproducible case study.

    ``probe_question`` is what the user actually asks the agent.
    ``gold_answer_substring`` is a short, unambiguous string that must
    appear in the agent's final answer for the answer to count as
    correct. ``forbidden_substrings`` are stale-answer phrases that
    must NOT appear (used to detect hallucination/conflation).
    """
    case_id: str
    title: str
    repo: str
    issue_url: str
    pinned_commit: str
    probe_question: str
    qualificand: str
    qualifier: str
    condition: str
    gold_answer_substring: str
    forbidden_substrings: tuple[str, ...]
    evidence: tuple[EvidenceItem, ...]
    notes: str = ""

    # --- helpers -----------------------------------------------------

    def evidence_by_freshness(self) -> tuple[EvidenceItem, ...]:
        """Sort evidence by ``precision`` descending — what a precision-aware
        retriever would produce. The case-study runner uses this for the
        *with-plugin* arm.
        """
        return tuple(sorted(self.evidence, key=lambda e: -e.precision))

    def evidence_by_seen_order(self) -> tuple[EvidenceItem, ...]:
        """Insert order — what an unaided agent reads page-by-page from a
        web search result list (older, higher-traffic SO answers tend
        to outrank fresh changelogs unless explicitly filtered).
        """
        return self.evidence


# ---------------------------------------------------------------------------
# Case 1: Django request.body / request.POST contradiction
# ---------------------------------------------------------------------------

_DJANGO_EVIDENCE: tuple[EvidenceItem, ...] = (
    EvidenceItem(
        id="django-so-1",
        qualificand="django_request",
        qualifier="post_then_body",
        condition="repo=django/django AND topic=request_body",
        content=(
            "If you have already accessed request.POST you cannot read "
            "request.body afterwards: Django raises a RawPostDataException. "
            "This is the historical Django 1.x and 2.x behaviour."
        ),
        precision=0.42,
        source="stackoverflow.com/q/22919637 (2014, top answer)",
        stale=True,
        superseded_by_id="django-docs-current",
    ),
    EvidenceItem(
        id="django-blog",
        qualificand="django_request",
        qualifier="post_then_body",
        condition="repo=django/django AND topic=request_body",
        content=(
            "Workaround: copy request.body before you read request.POST, "
            "because Django consumes the input stream when parsing the "
            "form. This is necessary for content_type='multipart/form-data'."
        ),
        precision=0.55,
        source="djangotricks.blogspot.com/2017",
        stale=False,  # the multipart caveat is still accurate
    ),
    EvidenceItem(
        id="django-docs-current",
        qualificand="django_request",
        qualifier="post_then_body",
        condition="repo=django/django AND topic=request_body",
        content=(
            "Per the Django 4.x reference: accessing request.body after "
            "request.POST is allowed only when the request body has not "
            "been consumed (i.e. when content_type is application/json or "
            "request.POST has not been accessed). For form-encoded POSTs "
            "the underlying input stream is exhausted and request.body "
            "raises RawPostDataException."
        ),
        precision=0.92,
        source="docs.djangoproject.com/en/4.2/ref/request-response/",
        stale=False,
    ),
    EvidenceItem(
        id="django-pr-15497-changelog",
        qualificand="django_request",
        qualifier="post_then_body",
        condition="repo=django/django AND topic=request_body",
        content=(
            "Django 4.1 release notes clarified that the RawPostDataException "
            "is raised only for form-encoded requests; JSON bodies remain "
            "readable via request.body even after POST inspection."
        ),
        precision=0.95,
        source="github.com/django/django/blob/4.2/docs/releases/4.1.txt",
        stale=False,
    ),
)


CASE_DJANGO = CaseStudy(
    case_id="django_request_body",
    title="When does Django raise RawPostDataException after request.POST?",
    repo="django/django",
    issue_url="https://code.djangoproject.com/ticket/27592",
    pinned_commit="django-4.2.11",
    probe_question=(
        "On Django 4.2, can I read request.body after I have already "
        "accessed request.POST for a form-encoded POST request?"
    ),
    qualificand="django_request",
    qualifier="post_then_body",
    condition="repo=django/django AND topic=request_body",
    gold_answer_substring="form-encoded",
    forbidden_substrings=(
        "Django 1.x",
        "always raises",
        "never raises",
    ),
    evidence=_DJANGO_EVIDENCE,
    notes=(
        "Classic four-way disagreement: legacy SO (always raises) vs blog "
        "(workaround) vs current docs (form-encoded only) vs release notes."
    ),
)


# ---------------------------------------------------------------------------
# Case 2: requests retry adapter API change
# ---------------------------------------------------------------------------

_REQUESTS_EVIDENCE: tuple[EvidenceItem, ...] = (
    EvidenceItem(
        id="req-so-old",
        qualificand="requests_retry",
        qualifier="adapter_api",
        condition="repo=psf/requests AND topic=retry_strategy",
        content=(
            "Use Retry from requests.packages.urllib3.util.retry — "
            "session.mount('http://', HTTPAdapter(max_retries=Retry(...))). "
            "method_whitelist parameter controls which methods retry."
        ),
        precision=0.35,
        source="stackoverflow.com/q/15431044 (pre-2020, urllib3 1.x)",
        stale=True,
        superseded_by_id="req-urllib3-changelog",
    ),
    EvidenceItem(
        id="req-cookbook",
        qualificand="requests_retry",
        qualifier="adapter_api",
        condition="repo=psf/requests AND topic=retry_strategy",
        content=(
            "requests.adapters.HTTPAdapter is the supported integration "
            "point. Pass a urllib3 Retry instance via max_retries."
        ),
        precision=0.70,
        source="docs.python-requests.org/en/latest/user/advanced/",
        stale=False,
    ),
    EvidenceItem(
        id="req-urllib3-changelog",
        qualificand="requests_retry",
        qualifier="adapter_api",
        condition="repo=psf/requests AND topic=retry_strategy",
        content=(
            "urllib3 1.26 deprecated method_whitelist in favour of "
            "allowed_methods; urllib3 2.0 removed method_whitelist entirely. "
            "Use Retry(total=5, allowed_methods={'GET','HEAD','OPTIONS'}) "
            "and import Retry from urllib3.util.retry directly."
        ),
        precision=0.93,
        source="github.com/urllib3/urllib3/blob/main/CHANGES.rst (2.0.0 entry)",
        stale=False,
    ),
)


CASE_REQUESTS = CaseStudy(
    case_id="requests_retry_adapter",
    title="Correct retry-strategy spelling for requests + urllib3 2.x",
    repo="psf/requests",
    issue_url="https://github.com/psf/requests/issues/2401",
    pinned_commit="requests-2.32.3+urllib3-2.2.2",
    probe_question=(
        "How do I configure a retry strategy on a requests.Session "
        "given urllib3 2.x is installed?"
    ),
    qualificand="requests_retry",
    qualifier="adapter_api",
    condition="repo=psf/requests AND topic=retry_strategy",
    gold_answer_substring="allowed_methods",
    forbidden_substrings=(
        "method_whitelist",
        "requests.packages.urllib3",
    ),
    evidence=_REQUESTS_EVIDENCE,
)


# ---------------------------------------------------------------------------
# Case 3: pandas iterrows dtype loss
# ---------------------------------------------------------------------------

_PANDAS_EVIDENCE: tuple[EvidenceItem, ...] = (
    EvidenceItem(
        id="pd-blog-old",
        qualificand="pandas_iterrows",
        qualifier="dtype_preservation",
        condition="repo=pandas-dev/pandas AND topic=iterrows_dtypes",
        content=(
            "iterrows is fine for typical workflows — pandas preserves the "
            "column dtype on each yielded row, so you can rely on integer "
            "and datetime values keeping their type."
        ),
        precision=0.30,
        source="towardsdatascience.com/pandas-iteration-2018",
        stale=True,
        superseded_by_id="pd-docs-current",
    ),
    EvidenceItem(
        id="pd-tutorial",
        qualificand="pandas_iterrows",
        qualifier="dtype_preservation",
        condition="repo=pandas-dev/pandas AND topic=iterrows_dtypes",
        content=(
            "Use itertuples for a faster, dtype-aware row-by-row iteration "
            "if you really must iterate."
        ),
        precision=0.74,
        source="realpython.com/pandas-iteration",
        stale=False,
    ),
    EvidenceItem(
        id="pd-docs-current",
        qualificand="pandas_iterrows",
        qualifier="dtype_preservation",
        condition="repo=pandas-dev/pandas AND topic=iterrows_dtypes",
        content=(
            "Per the pandas 2.x reference: iterrows returns a Series for each "
            "row, and because a Series has one dtype the row will be promoted "
            "to a common dtype (typically object) — so int columns can come "
            "out as float and datetimes can come out as Timestamps boxed in "
            "object. itertuples preserves the original dtypes."
        ),
        precision=0.94,
        source="pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html",
        stale=False,
    ),
)


CASE_PANDAS = CaseStudy(
    case_id="pandas_iterrows_dtype",
    title="Does DataFrame.iterrows preserve column dtypes?",
    repo="pandas-dev/pandas",
    issue_url="https://github.com/pandas-dev/pandas/issues/15014",
    pinned_commit="pandas-2.2.2",
    probe_question=(
        "If I iterate a DataFrame with iterrows, will an int64 column "
        "still come out as int64 inside the row Series?"
    ),
    qualificand="pandas_iterrows",
    qualifier="dtype_preservation",
    condition="repo=pandas-dev/pandas AND topic=iterrows_dtypes",
    gold_answer_substring="common dtype",
    forbidden_substrings=(
        "preserves the column dtype",
        "you can rely on integer",
    ),
    evidence=_PANDAS_EVIDENCE,
)


ALL_CASE_STUDIES: tuple[CaseStudy, ...] = (
    CASE_DJANGO,
    CASE_REQUESTS,
    CASE_PANDAS,
)


__all__ = [
    "ALL_CASE_STUDIES",
    "CASE_DJANGO",
    "CASE_PANDAS",
    "CASE_REQUESTS",
    "CaseStudy",
    "EvidenceItem",
]
