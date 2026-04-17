from dataclasses import dataclass


@dataclass
class AvacchedakaQuery:
    qualificand: str
    condition: str
    qualifier: str = ""
    precision_threshold: float = 0.5
    max_elements: int = 20

    def matches(self, element) -> bool:
        """True if element's avacchedaka qualificand matches and all condition tokens are present."""
        if element.avacchedaka.qualificand != self.qualificand:
            return False
        if not self.condition:
            return True
        for token in self.condition.split(" AND "):
            if token.strip() not in element.avacchedaka.condition:
                return False
        return True
