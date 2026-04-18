import dataclasses
from src.avacchedaka.element import ContextElement
from src.avacchedaka.query import AvacchedakaQuery


class ContextStore:
    def __init__(self):
        self._elements: dict[str, ContextElement] = {}

    def insert(self, element: ContextElement) -> None:
        if element.sublated_by is not None and element.precision > 0:
            raise ValueError(
                f"Element {element.id!r}: sublated_by is set but precision > 0 violates the sublation invariant"
            )
        self._elements[element.id] = element

    def get(self, element_id: str) -> ContextElement | None:
        return self._elements.get(element_id)

    def retrieve(
        self,
        query: AvacchedakaQuery,
        precision_threshold: float | None = None,
        max_elements: int | None = None,
    ) -> list[ContextElement]:
        threshold = precision_threshold if precision_threshold is not None else query.precision_threshold
        limit = max_elements if max_elements is not None else query.max_elements
        candidates = [
            e for e in self._elements.values()
            if e.sublated_by is None
            and e.precision >= threshold
            and query.matches(e)
        ]
        candidates.sort(key=lambda e: e.precision, reverse=True)
        return candidates[:limit]

    def sublate(self, element_id: str, by_element_id: str) -> None:
        if element_id not in self._elements:
            raise KeyError(f"Element {element_id} not found")
        elem = self._elements[element_id]
        self._elements[element_id] = dataclasses.replace(
            elem, sublated_by=by_element_id, precision=0.0
        )

    def compress(self, precision_threshold: float = 0.3) -> list[str]:
        """Set precision=0.0 on non-sublated elements below threshold. Returns their IDs."""
        compressed = []
        for eid, elem in list(self._elements.items()):
            if elem.sublated_by is None and 0 < elem.precision < precision_threshold:
                self._elements[eid] = dataclasses.replace(elem, precision=0.0)
                compressed.append(eid)
        return compressed

    def to_context_window(self, query: AvacchedakaQuery, max_tokens: int = 4096) -> str:
        """Assembles retrieved elements into a context string for injection into Claude API messages."""
        elements = self.retrieve(query)
        parts = []
        total_chars = 0
        char_budget = max_tokens * 4
        for e in elements:
            block = f"[{e.avacchedaka.qualificand}|precision={e.precision:.2f}] {e.content}"
            if total_chars + len(block) > char_budget:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n".join(parts)
