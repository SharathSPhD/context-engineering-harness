import json
import re
from enum import Enum


class KhyativadaClass(str, Enum):
    anyathakhyati = "anyathakhyati"
    atmakhyati = "atmakhyati"
    anirvacaniyakhyati = "anirvacaniyakhyati"
    asatkhyati = "asatkhyati"
    viparitakhyati = "viparitakhyati"
    akhyati = "akhyati"
    none = "none"


class KhyativadaClassifier:
    CLASSES = [c.value for c in KhyativadaClass if c != KhyativadaClass.none]

    def classify_heuristic(self, claim: str, ground_truth: str) -> dict:
        """Rule-based heuristic classifier. Used for unit-testable detection and fast batch labeling."""
        gt_lower = ground_truth.lower()

        # asatkhyati: ground truth says something does not exist
        if "does not exist" in gt_lower or "nonexistent" in gt_lower or "no such" in gt_lower:
            return {
                "class": KhyativadaClass.asatkhyati,
                "confidence": 0.85,
                "rationale": "Ground truth indicates the referenced entity does not exist.",
            }

        # akhyati: relational error — true components combined falsely
        # "but not for" captures "he won the prize but not for X" pattern
        if "but not for" in gt_lower or "but not due to" in gt_lower or "incorrectly attributed" in gt_lower:
            return {
                "class": KhyativadaClass.akhyati,
                "confidence": 0.75,
                "rationale": "Both components may be true but their combination is false.",
            }

        # anyathakhyati: version/identifier mismatch
        claim_nums = re.findall(r'\d+\.\d+', claim)
        gt_nums = re.findall(r'\d+\.\d+', ground_truth)
        if claim_nums and gt_nums and set(claim_nums) != set(gt_nums):
            return {
                "class": KhyativadaClass.anyathakhyati,
                "confidence": 0.80,
                "rationale": "Version/identifier mismatch — real entity misidentified.",
            }

        return {
            "class": KhyativadaClass.atmakhyati,
            "confidence": 0.5,
            "rationale": "Default: likely internal pattern projection without grounding.",
        }

    def classify(self, claim: str, context: str, ground_truth: str, api_key: str = "") -> dict:
        """LLM-based classifier using Claude. Falls back to heuristic if no api_key."""
        if not api_key:
            return self.classify_heuristic(claim, ground_truth)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""Classify this hallucination into exactly one of these 6 khyātivāda types:
- anyathakhyati: real entity misidentified as another real entity
- atmakhyati: internal pattern projected as external fact
- anirvacaniyakhyati: novel confabulation (neither real nor derivable)
- asatkhyati: nonexistent entity asserted to exist
- viparitakhyati: systematic inversion/reversal of A and B
- akhyati: two true claims combined into a false one

Claim: {claim}
Context provided: {context or 'none'}
Ground truth: {ground_truth}

Respond with JSON only: {{"class": "<type>", "confidence": <0-1>, "rationale": "<one sentence>"}}"""
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            result = json.loads(response.content[0].text)
            required = {"class", "confidence", "rationale"}
            if not required.issubset(result.keys()):
                raise ValueError(f"LLM response missing keys: {result}")
            return result
        except (json.JSONDecodeError, ValueError, KeyError):
            return self.classify_heuristic(claim, ground_truth)

    def batch_classify(self, examples: list[dict], api_key: str = "") -> list[dict]:
        return [
            self.classify(
                e.get("claim", ""), e.get("context", ""), e.get("ground_truth", ""), api_key
            )
            for e in examples
        ]
