from src.agents.manas import ManasAgent
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
        api_key: str,
        store: ContextStore,
        sakshi: SakshiPrefix = DEFAULT_SAKSHI,
        manas_model: str = "claude-haiku-4-5-20251001",
        buddhi_model: str = "claude-sonnet-4-6",
    ):
        self.store = store
        self.sakshi = sakshi
        self.manas = ManasAgent(api_key, manas_model)
        self.buddhi = BuddhiAgent(api_key, buddhi_model)

    def run(self, question: str, task_context: str, qualificand: str) -> BuddhiOutput:
        query = AvacchedakaQuery(qualificand=qualificand, condition=task_context)
        initial_context = self.store.to_context_window(query, max_tokens=2048)

        manas_out = self.manas.run(
            question=question,
            context_window=initial_context,
            task_context=task_context,
            qualificand=qualificand,
        )

        additional_parts = []
        for rec_query in manas_out.recommended_queries[:3]:
            additional_parts.append(self.store.to_context_window(rec_query, max_tokens=1024))
        enriched_context = initial_context + "\n" + "\n".join(additional_parts)

        return self.buddhi.run(
            question=question,
            context_window=enriched_context,
            manas_sketch=manas_out.reasoning_sketch,
            uncertainty=manas_out.uncertainty,
        )
