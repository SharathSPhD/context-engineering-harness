"""Unit tests for `experiments.v2.p6a.callers.MockHarnessCaller`.

The mock caller is the deterministic stand-in used by CI for the entire
P6-A multi-seed × multi-model re-run pipeline. If it ever drifts, the
end-to-end runner stops producing reproducible numbers — so we pin its
behaviour with explicit checks here.
"""
from __future__ import annotations

from experiments.v2.p6a.callers import MockHarnessCaller


def _haystack(vault_id: str = "vault-00-00", code: str = "ABCDEF") -> str:
    """Synthesise a minimal RULER-style haystack."""
    return (
        "The capital of Norvinia is Aldoria. "
        f"The activation code for vault {vault_id} is {code}. "
        "Operators must memorize this code; it will not appear elsewhere. "
        "The Pyranthos river flows through the city of Estoria. "
    )


def test_treatment_recovers_correct_code_for_asked_vault() -> None:
    caller = MockHarnessCaller(treatment_recall=1.0, baseline_recall=0.0)
    haystack = _haystack(vault_id="vault-00-00", code="ABCDEF")
    prompt = (
        "You are a focused long-context retriever. "
        f"=== DOCUMENT ===\n{haystack}\n=== END DOCUMENT ===\n"
        "What is the activation code for vault vault-00-00?"
    )
    out = caller(prompt=prompt, model="claude-sonnet-4-6", max_tokens=64, seed=0)
    assert out.text == "ABCDEF"
    assert out.metadata["caller"] == "MockHarnessCaller"


def test_baseline_returns_distractor_or_wrong_code() -> None:
    caller = MockHarnessCaller(treatment_recall=0.99, baseline_recall=0.0)
    # Baseline-style prompt: no structured retriever cues.
    haystack = (
        f"The activation code for vault vault-00-00 is ABCDEF. "
        f"The activation code for vault vault-00-01 is QQQQQQ."
    )
    prompt = (
        f"{haystack}\n\n"
        "Question: What is the activation code for vault vault-00-00?\n"
        "Answer:"
    )
    out = caller(prompt=prompt, model="claude-sonnet-4-6", max_tokens=64, seed=0)
    assert out.text != "ABCDEF"
    assert out.text in ("QQQQQQ", "")


def test_recall_for_multikey_listing_proportional_to_recall() -> None:
    pairs = [
        ("vault-00-00", "AAAAAA"),
        ("vault-00-01", "BBBBBB"),
        ("vault-00-02", "CCCCCC"),
        ("vault-00-03", "DDDDDD"),
        ("vault-00-04", "EEEEEE"),
    ]
    haystack = " ".join(
        f"The activation code for vault {k} is {v}." for k, v in pairs
    )
    keys = ", ".join(k for k, _ in pairs)
    prompt = (
        "You are a structured retriever. "
        "List ALL activation codes for the requested vaults.\n\n"
        f"DOCUMENT:\n{haystack}\n\n"
        f"REQUESTED VAULTS: {keys}\n\n"
        "ANSWERS:"
    )

    high = MockHarnessCaller(treatment_recall=1.0, baseline_recall=0.0)
    out_high = high(prompt=prompt, model="claude-sonnet-4-6", max_tokens=128, seed=0)
    assert all(v in out_high.text for _, v in pairs)

    low = MockHarnessCaller(treatment_recall=0.0, baseline_recall=0.0)
    out_low = low(prompt=prompt, model="claude-sonnet-4-6", max_tokens=128, seed=0)
    assert all(v not in out_low.text for _, v in pairs)


def test_haiku_underperforms_sonnet_at_same_recall() -> None:
    caller = MockHarnessCaller(
        treatment_recall=0.6, baseline_recall=0.4, haiku_penalty=0.5
    )
    pairs = [(f"vault-00-{i:02d}", f"CODE{i:02d}") for i in range(20)]
    haystack = " ".join(
        f"The activation code for vault {k} is {v}." for k, v in pairs
    )

    sonnet_hits = 0
    haiku_hits = 0
    for k, v in pairs:
        prompt = (
            "You are a focused long-context retriever. "
            f"=== DOCUMENT ===\n{haystack}\n=== END DOCUMENT ===\n"
            f"What is the activation code for vault {k}?"
        )
        out_s = caller(prompt=prompt, model="claude-sonnet-4-6", max_tokens=64, seed=42)
        out_h = caller(prompt=prompt, model="claude-haiku-4-5", max_tokens=64, seed=42)
        if out_s.text == v:
            sonnet_hits += 1
        if out_h.text == v:
            haiku_hits += 1
    # Sonnet should not be strictly worse than Haiku given the penalty.
    assert sonnet_hits >= haiku_hits


def test_caller_is_deterministic_per_input_tuple() -> None:
    caller = MockHarnessCaller()
    prompt = (
        "You are a focused long-context retriever. "
        "The activation code for vault vault-00-00 is ABCDEF. "
        "What is the activation code for vault vault-00-00?"
    )
    a = caller(prompt=prompt, model="claude-sonnet-4-6", max_tokens=64, seed=7)
    b = caller(prompt=prompt, model="claude-sonnet-4-6", max_tokens=64, seed=7)
    assert a.text == b.text
    assert a.input_tokens == b.input_tokens
    assert a.output_tokens == b.output_tokens
