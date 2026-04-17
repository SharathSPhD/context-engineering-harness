from src.agents.manas import ManasOutput
from src.avacchedaka.query import AvacchedakaQuery


def test_manas_output_fields():
    out = ManasOutput(
        candidate_ids=["a", "b"],
        uncertainty=0.7,
        recommended_queries=[AvacchedakaQuery(qualificand="auth", condition="task_type=qa")],
        reasoning_sketch="Could be X or Y.",
    )
    assert out.uncertainty == 0.7
    assert len(out.recommended_queries) == 1
    assert out.reasoning_sketch == "Could be X or Y."


def test_buddhi_output_withhold():
    from src.agents.buddhi import BuddhiOutput
    out = BuddhiOutput(answer=None, confidence=0.3)
    assert out.answer is None
    assert out.confidence < 0.6


def test_buddhi_output_defaults():
    from src.agents.buddhi import BuddhiOutput
    out = BuddhiOutput(answer="42", confidence=0.9)
    assert out.sublated == []
    assert out.reasoning_trace == ""
    assert out.khyativada_flags == []
