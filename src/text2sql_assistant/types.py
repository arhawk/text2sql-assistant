from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class Example:
    question: str
    sql: str
    split: str = "train"
    source: str = "sample"
    template_sql: str | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(data.pop("extra"))
        return data


@dataclass(slots=True)
class Prediction:
    mode: str
    question: str
    sql: str
    score: float
    matched_question: str
    matched_sql: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalSummary:
    mode: str
    total: int
    exact_match: int
    accuracy: float
    artifact_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
