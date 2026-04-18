from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AvacchedakaConditions:
    qualificand: str
    qualifier: str
    condition: str
    relation: str = "inherence"


@dataclass
class ContextElement:
    id: str
    content: str
    precision: float
    avacchedaka: AvacchedakaConditions
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance: str = ""
    sublated_by: str | None = None
    salience: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.precision <= 1.0:
            raise ValueError("precision must be between 0 and 1")
