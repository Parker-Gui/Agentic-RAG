"""Choose short source records for the current research question."""

import re

from app.agents.evidence import source_location
from app.agents.models import Evidence


def _terms(text: str) -> set[str]:
    chars = re.sub(r"\s+", "", text.lower())
    return {chars[i : i + 2] for i in range(len(chars) - 1)}


def select_evidence(
    question: str, evidence: list[Evidence], limit: int = 12, *, minimum_overlap: int = 0
) -> list[Evidence]:
    query_terms = _terms(question)
    return sorted(
        (item for item in evidence if len(query_terms & _terms(item.title + item.content[:500])) >= minimum_overlap),
        key=lambda item: (len(query_terms & _terms(item.title + item.content[:500])), item.id),
        reverse=True,
    )[:limit]


def evidence_prompt(evidence: list[Evidence], *, max_chars: int | None = 1000) -> str:
    return "\n\n".join(
        f"[{item.id}] {item.title} | 日期: {item.published_at or '未知'} | "
        f"来源: {source_location(item.model_dump())}\n{item.content[:max_chars]}"
        for item in evidence
    )
