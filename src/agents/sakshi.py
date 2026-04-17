class SakshiPrefix:
    """Witness-invariant frozen summary. Never rewritten by context operations.
    Provides the stable reference frame against which all reasoning occurs."""

    def __init__(self, content: str):
        self._content = content

    @property
    def content(self) -> str:
        return self._content

    def as_system_message(self) -> dict:
        return {"role": "user", "content": f"<sakshi_prefix>\n{self._content}\n</sakshi_prefix>"}
