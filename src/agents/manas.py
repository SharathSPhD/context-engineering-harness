import json
from dataclasses import dataclass, field
from src.cli_bridge import get_client
from src.avacchedaka.query import AvacchedakaQuery
from src.config import config


@dataclass
class ManasOutput:
    candidate_ids: list[str]
    uncertainty: float
    recommended_queries: list[AvacchedakaQuery]
    reasoning_sketch: str


class ManasAgent:
    """Broad attention, surfaces candidates, does NOT commit to answers.
    Analogous to manas: indecisive, sensory-bound mental activity."""

    SYSTEM = (
        "You are the manas stage of a two-stage reasoning system. "
        "Your role: surface candidate information relevant to the question. Do NOT commit to a final answer. "
        "Output JSON with keys: candidate_summary (str), uncertainty (float 0-1), "
        "recommended_queries (list of {qualificand, condition} dicts), reasoning_sketch (str)."
    )

    def __init__(self, api_key: str = "", model: str = ""):
        self.client = get_client(api_key)
        self.model = model or config.fast_model

    def run(self, question: str, context_window: str, task_context: str, qualificand: str) -> ManasOutput:
        messages = [
            {
                "role": "user",
                "content": f"Context:\n{context_window}\n\nQuestion: {question}\nTask: {task_context}",
            }
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self.SYSTEM,
            messages=messages,
        )
        try:
            raw = json.loads(response.content[0].text)
        except (json.JSONDecodeError, IndexError):
            raw = {}
        queries = [
            AvacchedakaQuery(
                qualificand=q.get("qualificand", qualificand),
                condition=q.get("condition", task_context),
            )
            for q in raw.get("recommended_queries", [])
        ]
        return ManasOutput(
            candidate_ids=[],
            uncertainty=float(raw.get("uncertainty", 0.9)),
            recommended_queries=queries or [AvacchedakaQuery(qualificand=qualificand, condition=task_context)],
            reasoning_sketch=raw.get("reasoning_sketch", raw.get("candidate_summary", "")),
        )
