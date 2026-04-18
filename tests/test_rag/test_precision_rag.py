import pytest
from src.rag.conflicting_qa import ConflictingSourceQA
from src.rag.precision_rag import PrecisionWeightedRAG
from src.rag.baselines import VanillaRAG


def test_conflicting_qa_has_two_sources():
    ex = ConflictingSourceQA.build_example("Q?", "correct", "wrong", 0.9, 0.3)
    assert len(ex.sources) == 2
    assert ex.correct_answer == "correct"


def test_conflicting_qa_sources_have_required_keys():
    ex = ConflictingSourceQA.build_example("Q?", "correct", "wrong", 0.9, 0.3)
    for src in ex.sources:
        assert "content" in src
        assert "precision" in src
        assert "answer" in src
        assert "is_correct" in src


def test_precision_rag_prefers_high_precision_source():
    rag = PrecisionWeightedRAG()
    ex = ConflictingSourceQA.build_example("Q?", "correct", "wrong", 0.9, 0.2)
    selected = rag.select_sources(ex.sources, top_k=1)
    assert selected[0]["answer"] == "correct"


def test_precision_rag_sorted_descending():
    rag = PrecisionWeightedRAG()
    sources = [
        {"content": "a", "precision": 0.5, "answer": "a"},
        {"content": "b", "precision": 0.9, "answer": "b"},
        {"content": "c", "precision": 0.7, "answer": "c"},
    ]
    selected = rag.select_sources(sources)
    assert selected[0]["precision"] == 0.9
    assert selected[-1]["precision"] == 0.5


def test_detect_conflict_true_when_answers_differ_and_gap_small():
    rag = PrecisionWeightedRAG()
    sources = [
        {"content": "a", "precision": 0.8, "answer": "24h"},
        {"content": "b", "precision": 0.75, "answer": "1h"},
    ]
    assert rag.detect_conflict(sources) is True


def test_detect_conflict_false_when_gap_large():
    rag = PrecisionWeightedRAG()
    sources = [
        {"content": "a", "precision": 0.95, "answer": "24h"},
        {"content": "b", "precision": 0.2, "answer": "1h"},
    ]
    assert rag.detect_conflict(sources) is False


def test_detect_conflict_false_when_answers_same():
    rag = PrecisionWeightedRAG()
    sources = [
        {"content": "a", "precision": 0.8, "answer": "24h"},
        {"content": "b", "precision": 0.75, "answer": "24h"},
    ]
    assert rag.detect_conflict(sources) is False


def test_build_prompt_includes_conflict_note():
    rag = PrecisionWeightedRAG()
    sources = [
        {"content": "says X", "precision": 0.8, "answer": "X"},
        {"content": "says Y", "precision": 0.75, "answer": "Y"},
    ]
    prompt = rag.build_prompt("Q?", sources)
    assert "conflict" in prompt.lower() or "uncertainty" in prompt.lower()


def test_build_prompt_no_conflict_note_when_no_conflict():
    rag = PrecisionWeightedRAG()
    sources = [
        {"content": "says X", "precision": 0.95, "answer": "X"},
        {"content": "says X too", "precision": 0.9, "answer": "X"},
    ]
    prompt = rag.build_prompt("Q?", sources)
    assert "conflict" not in prompt.lower()


def test_vanilla_rag_preserves_order():
    rag = VanillaRAG()
    sources = [{"content": f"s{i}", "precision": float(i) / 10, "answer": str(i)} for i in range(5)]
    selected = rag.select_sources(sources, top_k=3)
    assert [s["answer"] for s in selected] == ["0", "1", "2"]
