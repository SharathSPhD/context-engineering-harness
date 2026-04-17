import json
from dataclasses import dataclass, field
import anthropic


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

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def run(
        self,
        question: str,
        context_window: str,
        manas_sketch: str,
        uncertainty: float,
    ) -> BuddhiOutput:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_window}\n\n"
                    f"Manas sketch (uncommitted):\n{manas_sketch}\n\n"
                    f"Manas uncertainty: {uncertainty:.2f}\n\n"
                    f"Question: {question}"
                ),
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
            return BuddhiOutput(answer=None, confidence=0.0, reasoning_trace="parse error")
        return BuddhiOutput(
            answer=raw.get("answer"),
            confidence=float(raw.get("confidence", 0.0)),
            sublated=raw.get("sublated_candidates", []),
            reasoning_trace=raw.get("reasoning_trace", ""),
            khyativada_flags=raw.get("khyativada_flags", []),
        )
