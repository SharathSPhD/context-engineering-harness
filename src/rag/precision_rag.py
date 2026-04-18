class PrecisionWeightedRAG:
    def select_sources(self, sources: list[dict], top_k: int = 3) -> list[dict]:
        """Return top_k sources sorted by precision descending."""
        return sorted(sources, key=lambda s: s.get("precision", 0.5), reverse=True)[:top_k]

    def detect_conflict(self, sources: list[dict]) -> bool:
        """True if top-2 sources give different answers and precision gap < 0.3."""
        if len(sources) < 2:
            return False
        top2 = self.select_sources(sources, top_k=2)
        answers_differ = top2[0].get("answer") != top2[1].get("answer")
        gap = abs(top2[0].get("precision", 0.0) - top2[1].get("precision", 0.0))
        return answers_differ and gap < 0.3

    def build_prompt(self, question: str, sources: list[dict]) -> str:
        selected = self.select_sources(sources)
        conflict = self.detect_conflict(sources)
        source_text = "\n".join(
            f"[Source precision={s['precision']:.2f}] {s['content']}" for s in selected
        )
        conflict_note = "\nNote: Sources conflict. Express appropriate uncertainty." if conflict else ""
        return f"Sources:\n{source_text}{conflict_note}\n\nQuestion: {question}\nAnswer:"
