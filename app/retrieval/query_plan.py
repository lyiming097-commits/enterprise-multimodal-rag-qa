from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QueryPlan:
    """All retrieval inputs produced during query preprocessing."""

    original_query: str
    rewritten_query: str
    expanded_queries: list[str] = field(default_factory=list)
    hypothetical_document: str | None = None

    @classmethod
    def from_payload(cls, original_query: str, payload: dict[str, Any]) -> QueryPlan:
        original = original_query.strip()
        rewritten = str(payload.get("rewritten_query", "")).strip() or original
        raw_expanded = payload.get("expanded_queries", [])
        if not isinstance(raw_expanded, list):
            raw_expanded = []
        expanded = [
            value
            for value in _unique_nonempty(str(item) for item in raw_expanded)
            if value not in {original, rewritten}
        ][:3]
        hypothetical = str(payload.get("hypothetical_document", "")).strip() or None
        return cls(
            original_query=original,
            rewritten_query=rewritten,
            expanded_queries=expanded,
            hypothetical_document=hypothetical,
        )

    @classmethod
    def fallback(cls, question: str) -> QueryPlan:
        question = question.strip()
        return cls(original_query=question, rewritten_query=question)

    @property
    def lexical_queries(self) -> list[str]:
        return _unique_nonempty([self.original_query, self.rewritten_query, *self.expanded_queries])

    @property
    def dense_queries(self) -> list[str]:
        return self.lexical_queries

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique_nonempty(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in result:
            result.append(clean)
    return result
