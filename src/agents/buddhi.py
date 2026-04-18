import json
from dataclasses import dataclass, field
from src.cli_bridge import get_client
from src.config import config


@dataclass
class BuddhiOutput:
    answer: str | None
    confidence: float
    sublated: list[str] = field(default_factory=list)
    reasoning_trace: str = ""
    khyativada_flags: list[str] = field(default_factory=list)


class BuddhiAgent:
    """Narrow attention, discriminates, commits to answer or explicitly withholds.
    Analogous to buddhi: discriminative, decisive faculty."""

    SYSTEM = (
        "You are the buddhi stage of a two-stage reasoning system. "
        "Given candidate reasoning and context, commit to a final answer OR explicitly withhold "
        "if evidence is insufficient. "
        "Output JSON: {\"answer\": str or null, \"confidence\": 0-1, \"reasoning_trace\": str, "
        "\"sublated_candidates\": [str], \"khyativada_flags\": [str]}. "
        "If confidence < 0.6 and evidence is weak, set answer to null. Never fabricate."
    )

    def __init__(self, api_key: str = "", model: str = "", max_tokens: int = 1024):
        self.client = get_client(api_key)
        self.model = model or config.smart_model
        self.max_tokens = max_tokens

    def run(
        self,
        question: str,
        context_window: str,
        manas_sketch: str,
        uncertainty: float,
        candidate_ids: list[str] | None = None,
        sakshi_invariant: str = "",
    ) -> BuddhiOutput:
        system = self.SYSTEM
        if sakshi_invariant:
            system = (
                f"{self.SYSTEM}\n\n"
                f"<sakshi_prefix>\n{sakshi_invariant}\n</sakshi_prefix>"
            )
        content = (
            f"Context:\n{context_window}\n\n"
            f"Candidate element IDs surfaced by manas: {candidate_ids or []}\n\n"
            f"Manas sketch (uncommitted):\n{manas_sketch}\n\n"
            f"Manas uncertainty: {uncertainty:.2f}\n\n"
            f"Question: {question}"
        )
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        try:
            raw = json.loads(response.content[0].text)
        except (json.JSONDecodeError, IndexError):
            return BuddhiOutput(answer=None, confidence=0.0, reasoning_trace="parse error")
        return BuddhiOutput(
            answer=raw.get("answer"),
            confidence=float(raw.get("confidence", 0.0)),
            sublated=raw.get("sublated_candidates", []),
            reasoning_trace=raw.get("reasoning_trace", ""),
            khyativada_flags=raw.get("khyativada_flags", []),
        )
