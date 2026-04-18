from dataclasses import dataclass


@dataclass
class QAExample:
    question: str
    correct_answer: str
    sources: list[dict]


class ConflictingSourceQA:
    @staticmethod
    def build_example(
        question: str,
        correct_answer: str,
        incorrect_answer: str,
        correct_source_precision: float,
        incorrect_source_precision: float,
    ) -> QAExample:
        return QAExample(
            question=question,
            correct_answer=correct_answer,
            sources=[
                {
                    "content": f"According to this source: {correct_answer}",
                    "precision": correct_source_precision,
                    "answer": correct_answer,
                    "is_correct": True,
                },
                {
                    "content": f"According to this source: {incorrect_answer}",
                    "precision": incorrect_source_precision,
                    "answer": incorrect_answer,
                    "is_correct": False,
                },
            ],
        )
