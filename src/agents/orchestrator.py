"""ManusBuddhi orchestrator — two-stage agent loop with Sakshi invariant.

v2 fix (G9): the SakshiPrefix is now passed through to each stage as an
explicit system-message addendum rather than inlined into the user-visible
context window. The witness invariant must be a property of the agent's
identity, not just another retrievable element.
"""
from __future__ import annotations

from src.agents.buddhi import BuddhiAgent, BuddhiOutput
from src.agents.manas import ManasAgent, ManasOutput
from src.agents.sakshi import SakshiPrefix
from src.avacchedaka.query import AvacchedakaQuery
from src.avacchedaka.store import ContextStore
from src.config import config

DEFAULT_SAKSHI = SakshiPrefix(
    "This system conducts rigorous, grounded reasoning. "
    "It withholds answers when evidence is insufficient rather than fabricating."
)


class ManusBuddhiOrchestrator:
    def __init__(
        self,
        api_key: str = "",
        store: ContextStore | None = None,
        sakshi: SakshiPrefix = DEFAULT_SAKSHI,
        manas_model: str = "",
        buddhi_model: str = "",
        manas_max_tokens: int = 512,
        buddhi_max_tokens: int = 1024,
    ):
        if store is None:
            store = ContextStore()
        self.store = store
        self.sakshi = sakshi
        self.manas = ManasAgent(api_key, manas_model or config.fast_model, max_tokens=manas_max_tokens)
        self.buddhi = BuddhiAgent(api_key, buddhi_model or config.smart_model, max_tokens=buddhi_max_tokens)

    def run(self, question: str, task_context: str, qualificand: str) -> BuddhiOutput:
        query = AvacchedakaQuery(qualificand=qualificand, condition=task_context)
        retrieved_elements = self.store.retrieve(query, max_elements=10)
        initial_context = self.store.to_context_window(query, max_tokens=2048)

        manas_out = self.manas.run(
            question=question,
            context_window=initial_context,
            task_context=task_context,
            qualificand=qualificand,
            sakshi_invariant=self.sakshi.content,
        )
        manas_out = ManasOutput(
            candidate_ids=[e.id for e in retrieved_elements],
            uncertainty=manas_out.uncertainty,
            recommended_queries=manas_out.recommended_queries,
            reasoning_sketch=manas_out.reasoning_sketch,
        )

        additional_parts = []
        for rec_query in manas_out.recommended_queries[:3]:
            additional_parts.append(self.store.to_context_window(rec_query, max_tokens=1024))
        enriched_context = initial_context
        if additional_parts:
            enriched_context += "\n" + "\n".join(additional_parts)

        return self.buddhi.run(
            question=question,
            context_window=enriched_context,
            manas_sketch=manas_out.reasoning_sketch,
            uncertainty=manas_out.uncertainty,
            candidate_ids=manas_out.candidate_ids,
            sakshi_invariant=self.sakshi.content,
        )
