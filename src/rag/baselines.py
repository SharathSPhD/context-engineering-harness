class VanillaRAG:
    """Baseline: top-k retrieval with no precision weighting."""

    def select_sources(self, sources: list[dict], top_k: int = 3) -> list[dict]:
        return sources[:top_k]

    def build_prompt(self, question: str, sources: list[dict]) -> str:
        selected = self.select_sources(sources)
        source_text = "\n".join(f"[Source] {s['content']}" for s in selected)
        return f"Sources:\n{source_text}\n\nQuestion: {question}\nAnswer:"
