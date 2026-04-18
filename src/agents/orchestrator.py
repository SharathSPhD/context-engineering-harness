from src.agents.manas import ManasAgent, ManasOutput
from src.agents.buddhi import BuddhiAgent, BuddhiOutput
from src.agents.sakshi import SakshiPrefix
from src.avacchedaka.store import ContextStore
from src.avacchedaka.query import AvacchedakaQuery

DEFAULT_SAKSHI = SakshiPrefix(
    "This system conducts rigorous, grounded reasoning. "
    "It withholds answers when evidence is insufficient rather than fabricating."
)


class ManusBuddhiOrchestrator:
    def __init__(
        self,
        api_key: str = "",
        store: ContextStore = None,
        sakshi: SakshiPrefix = DEFAULT_SAKSHI,
        manas_model: str = "claude-haiku-4-5",
        buddhi_model: str = "claude-sonnet-4-6",
    ):
        if store is None:
            store = ContextStore()
        self.store = store
        self.sakshi = sakshi
        self.manas = ManasAgent(api_key, manas_model)
        self.buddhi = BuddhiAgent(api_key, buddhi_model)

    def run(self, question: str, task_context: str, qualificand: str) -> BuddhiOutput:
        query = AvacchedakaQuery(qualificand=qualificand, condition=task_context)
        retrieved_elements = self.store.retrieve(query, max_elements=10)
        initial_context = self.store.to_context_window(query, max_tokens=2048)

        manas_out = self.manas.run(
            question=question,
            context_window=initial_context,
            task_context=task_context,
            qualificand=qualificand,
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

        sakshi_block = f"[SAKSHI INVARIANT]\n{self.sakshi.content}\n[/SAKSHI INVARIANT]"
        enriched_context = sakshi_block + "\n\n" + enriched_context

        return self.buddhi.run(
            question=question,
            context_window=enriched_context,
            manas_sketch=manas_out.reasoning_sketch,
            uncertainty=manas_out.uncertainty,
            candidate_ids=manas_out.candidate_ids,
        )
