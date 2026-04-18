from dataclasses import dataclass, field
from src.avacchedaka.element import ContextElement, AvacchedakaConditions
from src.avacchedaka.store import ContextStore


@dataclass
class ShiftedTask:
    pre_shift_elements: list[ContextElement]
    post_shift_elements: list[ContextElement]
    question: str
    pre_shift_answer: str
    post_shift_answer: str


class DistributionShiftBenchmark:
    """Builds a long-horizon benchmark with a controlled distribution shift.

    Analogous to H7's 'codebase that changes API conventions midstream'.
    """

    def build_jwt_shift(self) -> ShiftedTask:
        conds_pre = AvacchedakaConditions(qualificand="auth", qualifier="expiry", condition="phase=pre_shift")
        conds_post = AvacchedakaConditions(qualificand="auth", qualifier="expiry", condition="phase=post_shift")
        pre = [
            ContextElement(id="pre-001", content="JWT tokens expire after 24 hours.", precision=0.9, avacchedaka=conds_pre),
            ContextElement(id="pre-002", content="Token refresh window is 23 hours.", precision=0.85, avacchedaka=conds_pre),
        ]
        post = [
            ContextElement(id="post-001", content="JWT tokens now expire after 1 hour (policy updated).", precision=0.95, avacchedaka=conds_post),
            ContextElement(id="post-002", content="Token refresh window is 50 minutes.", precision=0.9, avacchedaka=conds_post),
        ]
        return ShiftedTask(
            pre_shift_elements=pre,
            post_shift_elements=post,
            question="How long do JWT tokens last?",
            pre_shift_answer="24 hours",
            post_shift_answer="1 hour",
        )

    def build_store_pre_shift(self, task: ShiftedTask) -> ContextStore:
        store = ContextStore()
        for e in task.pre_shift_elements:
            store.insert(e)
        return store

    def apply_shift(self, store: ContextStore, task: ShiftedTask) -> None:
        """Apply the distribution shift: sublate pre-shift elements, insert post-shift."""
        for post_elem in task.post_shift_elements:
            store.insert(post_elem)
        for pre_elem in task.pre_shift_elements:
            matching_post = task.post_shift_elements[0]
            store.sublate(pre_elem.id, matching_post.id)
