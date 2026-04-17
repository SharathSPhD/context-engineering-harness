import sys
import types
from unittest.mock import MagicMock

import pytest
from src.evaluation.khyativada import KhyativadaClassifier, KhyativadaClass


def test_classifier_has_six_classes():
    assert len(KhyativadaClassifier.CLASSES) == 6


def test_all_six_class_names_present():
    expected = {"anyathakhyati", "atmakhyati", "anirvacaniyakhyati", "asatkhyati", "viparitakhyati", "akhyati"}
    assert set(KhyativadaClassifier.CLASSES) == expected


def test_anyathakhyati_version_mismatch():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(
        claim="Python's GIL was removed in version 3.10",
        ground_truth="Python's GIL was removed in version 3.13",
    )
    assert result["class"] == KhyativadaClass.anyathakhyati


def test_asatkhyati_nonexistent_api():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(
        claim="Use the requests.get_json() method to parse responses",
        ground_truth="requests.get_json() does not exist; use response.json() instead",
    )
    assert result["class"] == KhyativadaClass.asatkhyati


def test_akhyati_relational_combination_error():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(
        claim="Einstein won the Nobel Prize in 1921 for his theory of relativity",
        ground_truth="Einstein won in 1921 but not for relativity — he won for the photoelectric effect",
    )
    assert result["class"] == KhyativadaClass.akhyati


def test_atmakhyati_default_fallback():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(
        claim="The standard port for this service is 8080",
        ground_truth="No source confirms a standard port for this service",
    )
    assert result["class"] == KhyativadaClass.atmakhyati


def test_classify_returns_required_keys():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(claim="something", ground_truth="something else")
    assert "class" in result
    assert "confidence" in result
    assert "rationale" in result


def test_confidence_in_valid_range():
    clf = KhyativadaClassifier()
    result = clf.classify_heuristic(
        claim="Python's GIL was removed in version 3.10",
        ground_truth="Python's GIL was removed in version 3.13",
    )
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_no_api_key_uses_heuristic():
    clf = KhyativadaClassifier()
    result = clf.classify(claim="something", context="", ground_truth="something does not exist")
    assert result["class"] == KhyativadaClass.asatkhyati


def test_classify_with_api_key_uses_llm(monkeypatch):
    fake_anthropic = types.ModuleType("anthropic")

    class FakeClient:
        def __init__(self, api_key=""):
            self.messages = self
            self._api_key = api_key

        def create(self, **kwargs):
            block = MagicMock()
            block.text = '{"class": "anyathakhyati", "confidence": 0.9, "rationale": "mocked"}'
            resp = MagicMock()
            resp.content = [block]
            return resp

    fake_anthropic.Anthropic = FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    clf = KhyativadaClassifier()
    result = clf.classify(
        claim="Python 3.10", context="docs", ground_truth="Python 3.13", api_key="test-key"
    )
    assert result["class"] == "anyathakhyati"
    assert result["confidence"] == 0.9


def test_batch_classify_returns_list():
    clf = KhyativadaClassifier()
    examples = [
        {"claim": "requests.get_json()", "context": "", "ground_truth": "does not exist"},
        {"claim": "ver 3.10", "context": "", "ground_truth": "version 3.13"},
    ]
    results = clf.batch_classify(examples)
    assert len(results) == 2
    assert all("class" in r for r in results)
