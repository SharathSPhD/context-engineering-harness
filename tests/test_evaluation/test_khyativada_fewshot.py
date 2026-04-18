"""Tests for src.evaluation.khyativada_fewshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from src.evaluation.khyativada import KhyativadaClass, KhyativadaClassifier
from src.evaluation.khyativada_fewshot import (
    FewShotKhyativadaClassifier,
    KhyativadaPrediction,
    _apply_guardrails,
    _parse_structured_response,
    _select_exemplars,
    _validate_payload,
)


@dataclass
class _StubResponse:
    text: str

    @property
    def content(self):
        return [self]


@dataclass
class _StubMessages:
    next_text: str
    seen_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs):
        self.seen_kwargs = kwargs
        return _StubResponse(self.next_text)


@dataclass
class _StubClient:
    messages: _StubMessages


def _client_factory(text: str):
    msgs = _StubMessages(next_text=text)
    return lambda: _StubClient(messages=msgs), msgs


# Pure helpers
class TestParseStructuredResponse:
    def test_plain_object(self):
        out = _parse_structured_response('{"class": "asatkhyati", "confidence": 0.9, "rationale": "x"}')
        assert out["class"] == "asatkhyati"

    def test_fenced_json(self):
        out = _parse_structured_response(
            'Sure thing!\n```json\n{"class": "akhyati", "confidence": 0.7, "rationale": "x"}\n```\n'
        )
        assert out["class"] == "akhyati"

    def test_inline_json_with_surrounding_prose(self):
        out = _parse_structured_response(
            'My answer: {"class": "atmakhyati", "confidence": 0.6, "rationale": "x"}'
        )
        assert out["class"] == "atmakhyati"

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            _parse_structured_response("totally not JSON at all")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_structured_response("")


class TestValidatePayload:
    def test_valid_payload_passes(self):
        out = _validate_payload({"class": "asatkhyati", "confidence": 0.9, "rationale": "ok"})
        assert out["class"] == "asatkhyati"

    def test_label_alias_accepted(self):
        out = _validate_payload({"label": "akhyati", "confidence": 0.5, "rationale": "ok"})
        assert out["class"] == "akhyati"

    def test_unknown_class_rejected(self):
        with pytest.raises(ValueError):
            _validate_payload({"class": "wrong", "confidence": 0.5, "rationale": "x"})

    def test_out_of_range_confidence_rejected(self):
        with pytest.raises(ValueError):
            _validate_payload({"class": "atmakhyati", "confidence": 1.5, "rationale": "x"})

    def test_empty_rationale_rejected(self):
        with pytest.raises(ValueError):
            _validate_payload({"class": "atmakhyati", "confidence": 0.5, "rationale": "   "})

    def test_none_label_accepted(self):
        out = _validate_payload({"class": "none", "confidence": 0.5, "rationale": "no hallucination"})
        assert out["class"] == "none"


class TestGuardrails:
    def test_nonexistence_guardrail_overrides(self):
        label, reason = _apply_guardrails(
            claim="use foo()", ground_truth="foo() does not exist", llm_label="atmakhyati"
        )
        assert label == "asatkhyati"
        assert reason == "ground_truth_asserts_nonexistence"

    def test_inversion_guardrail_overrides(self):
        label, reason = _apply_guardrails(
            claim="A is X, B is Y",
            ground_truth="The opposite is true: A is Y, B is X.",
            llm_label="anyathakhyati",
        )
        assert label == "viparitakhyati"

    def test_relational_guardrail_overrides(self):
        label, reason = _apply_guardrails(
            claim="X for Y", ground_truth="Yes for X but not for Y", llm_label="atmakhyati"
        )
        assert label == "akhyati"

    def test_no_override_when_llm_already_matches(self):
        label, reason = _apply_guardrails(
            claim="x", ground_truth="x does not exist", llm_label="asatkhyati"
        )
        assert label == "asatkhyati"
        assert reason is None

    def test_no_guardrail_signal_returns_llm_label(self):
        label, reason = _apply_guardrails(
            claim="anything", ground_truth="completely unrelated", llm_label="atmakhyati"
        )
        assert label == "atmakhyati"
        assert reason is None


class TestExemplarSelection:
    def test_returns_at_least_one_per_class(self):
        chosen = _select_exemplars(per_class=2, seed=0)
        labels = {ex.label for ex in chosen}
        assert {"anyathakhyati", "atmakhyati", "anirvacaniyakhyati", "asatkhyati", "viparitakhyati", "akhyati", "none"} <= labels

    def test_deterministic_under_same_seed(self):
        a = _select_exemplars(per_class=2, seed=7)
        b = _select_exemplars(per_class=2, seed=7)
        assert [(ex.label, ex.claim) for ex in a] == [(ex.label, ex.claim) for ex in b]


# End-to-end classifier tests
class TestClassifierFewShotPath:
    def test_llm_path_returns_prediction(self):
        factory, msgs = _client_factory(
            '{"class": "asatkhyati", "confidence": 0.91, "rationale": "no such method"}'
        )
        clf = FewShotKhyativadaClassifier(client_factory=factory, n_shots_per_class=1)
        pred = clf.classify(
            claim="use requests.get_json()",
            context="",
            ground_truth="requests.get_json does not exist; use response.json()",
        )
        # Guardrail and LLM agree → source should be "llm"
        assert pred.source == "llm"
        assert pred.label == "asatkhyati"
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.llm_label == "asatkhyati"
        assert pred.heuristic_label == "asatkhyati"
        assert pred.agreement is True

    def test_guardrail_overrides_llm(self):
        factory, _ = _client_factory(
            '{"class": "atmakhyati", "confidence": 0.55, "rationale": "model said pattern"}'
        )
        clf = FewShotKhyativadaClassifier(client_factory=factory)
        pred = clf.classify(
            claim="use requests.get_json()",
            context="",
            ground_truth="requests.get_json does not exist",
        )
        assert pred.source == "guardrail"
        assert pred.label == "asatkhyati"
        assert pred.llm_label == "atmakhyati"
        assert "guardrail-override" in pred.rationale

    def test_invalid_llm_response_falls_back_to_heuristic(self):
        factory, _ = _client_factory("totally not JSON")
        clf = FewShotKhyativadaClassifier(client_factory=factory)
        pred = clf.classify(
            claim="use foo()",
            context="",
            ground_truth="foo() does not exist",
        )
        assert pred.source == "heuristic"
        assert pred.label == "asatkhyati"

    def test_client_exception_falls_back_to_heuristic(self):
        def bad_factory():
            raise RuntimeError("CLI down")

        clf = FewShotKhyativadaClassifier(client_factory=bad_factory)
        pred = clf.classify(
            claim="x",
            context="",
            ground_truth="x does not exist",
        )
        assert pred.source == "heuristic"
        assert pred.label == "asatkhyati"

    def test_prompt_passes_seed_and_system(self):
        factory, msgs = _client_factory(
            '{"class": "atmakhyati", "confidence": 0.5, "rationale": "ok"}'
        )
        clf = FewShotKhyativadaClassifier(client_factory=factory, seed=42)
        clf.classify(claim="x", context="", ground_truth="y")
        assert msgs.seen_kwargs is not None
        assert msgs.seen_kwargs.get("seed") == 42
        assert "Khyātivāda" in msgs.seen_kwargs.get("system", "")

    def test_prompt_includes_few_shot_exemplars(self):
        factory, msgs = _client_factory(
            '{"class": "atmakhyati", "confidence": 0.5, "rationale": "ok"}'
        )
        clf = FewShotKhyativadaClassifier(client_factory=factory, n_shots_per_class=1)
        clf.classify(claim="x", context="", ground_truth="y")
        prompt = msgs.seen_kwargs["messages"][0]["content"]
        assert "Examples:" in prompt
        # All six fault classes plus "none" should appear in the few-shot block
        for label in (
            "anyathakhyati",
            "atmakhyati",
            "anirvacaniyakhyati",
            "asatkhyati",
            "viparitakhyati",
            "akhyati",
            "none",
        ):
            assert label in prompt

    def test_batch_classify(self):
        factory, _ = _client_factory(
            '{"class": "atmakhyati", "confidence": 0.5, "rationale": "ok"}'
        )
        clf = FewShotKhyativadaClassifier(client_factory=factory)
        examples = [
            {"claim": "a", "context": "", "ground_truth": "z"},
            {"claim": "use foo()", "context": "", "ground_truth": "foo() does not exist"},
        ]
        preds = clf.batch_classify(examples)
        assert len(preds) == 2
        assert isinstance(preds[0], KhyativadaPrediction)
        # Second example should be guardrail-overridden to asatkhyati
        assert preds[1].label == "asatkhyati"

    def test_as_dict_round_trip(self):
        factory, _ = _client_factory(
            '{"class": "atmakhyati", "confidence": 0.6, "rationale": "ok"}'
        )
        clf = FewShotKhyativadaClassifier(client_factory=factory)
        pred = clf.classify(claim="x", context="", ground_truth="y")
        d = pred.as_dict()
        assert d["class"] == "atmakhyati"
        assert d["source"] == "llm"
        assert "rationale" in d
