"""ContextStore — the typed, sublatable knowledge container.

v2 invariants (tightened from v0):
  - `insert` raises on duplicate id by default; pass `overwrite=True` to upsert.
  - `compress` honors optional `qualificand` and `task_context` to scope which
    elements are eligible for compression (G7).
  - `to_context_window` budgets in tokenizer-exact tokens, not characters (G12).
"""
from __future__ import annotations

import dataclasses
import logging

from src.avacchedaka.element import ContextElement
from src.avacchedaka.query import AvacchedakaQuery
from src.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class ContextStore:
    def __init__(self) -> None:
        self._elements: dict[str, ContextElement] = {}

    def insert(self, element: ContextElement, *, overwrite: bool = False) -> None:
        """Insert a new element. Raises if id already exists unless overwrite=True."""
        if element.sublated_by is not None and element.precision > 0:
            raise ValueError(
                f"Element {element.id!r}: sublated_by is set but precision > 0 violates the sublation invariant"
            )
        if element.id in self._elements and not overwrite:
            raise ValueError(
                f"Element {element.id!r} already exists; pass overwrite=True to replace"
            )
        self._elements[element.id] = element

    def get(self, element_id: str) -> ContextElement | None:
        return self._elements.get(element_id)

    def __len__(self) -> int:
        return len(self._elements)

    def __contains__(self, element_id: str) -> bool:
        return element_id in self._elements

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

    def compress(
        self,
        precision_threshold: float = 0.3,
        *,
        qualificand: str = "",
        task_context: str = "",
    ) -> list[str]:
        """Set precision=0.0 on non-sublated elements below threshold.

        Optional scoping (G7):
          - `qualificand`: only consider elements with this avacchedaka qualificand.
          - `task_context`: AND-conjunctive condition; element's condition tokens
            must be a superset of the task_context tokens.

        Returns the list of compressed element ids.
        """
        need_tokens = (
            {t.strip() for t in task_context.split(" AND ")} if task_context else set()
        )
        compressed: list[str] = []
        for eid, elem in list(self._elements.items()):
            if elem.sublated_by is not None:
                continue
            if not (0 < elem.precision < precision_threshold):
                continue
            if qualificand and elem.avacchedaka.qualificand != qualificand:
                continue
            if need_tokens:
                have_tokens = {
                    t.strip() for t in elem.avacchedaka.condition.split(" AND ")
                }
                if not need_tokens.issubset(have_tokens):
                    continue
            self._elements[eid] = dataclasses.replace(elem, precision=0.0)
            compressed.append(eid)
        return compressed

    def to_context_window(
        self,
        query: AvacchedakaQuery,
        max_tokens: int = 4096,
        *,
        encoding: str = "o200k_base",
    ) -> str:
        """Assemble retrieved elements into a context string within `max_tokens`.

        Budgets in tokenizer-exact tokens (G12). When the next element would
        exceed `max_tokens`, the assembly stops; we never emit a partial block.
        """
        elements = self.retrieve(query)
        parts: list[str] = []
        used = 0
        for e in elements:
            block = f"[{e.avacchedaka.qualificand}|precision={e.precision:.2f}] {e.content}"
            block_tokens = count_tokens(block, encoding=encoding)
            # Account for the joining newline between blocks.
            extra = 1 if parts else 0
            if used + block_tokens + extra > max_tokens:
                break
            parts.append(block)
            used += block_tokens + extra
        return "\n".join(parts)
