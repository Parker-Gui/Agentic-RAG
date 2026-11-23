"""Stable, serializable records shared by the outer research workflow."""

from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class TerminationReason(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MODEL_BUDGET = "model_budget"
    TOOL_BUDGET = "tool_budget"
    TIME_BUDGET = "time_budget"
    REVIEW_BUDGET = "review_budget"
    EXECUTION_ERROR = "execution_error"


class Subtask(BaseModel):
    id: str
    goal: str
    status: str = "pending"
    evidence_ids: list[str] = Field(default_factory=list)
    attempts: int = 0


class MemoryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    quote: str = Field(min_length=1, max_length=2000)


class MemoryResult(BaseModel):
    status: Literal["pending", "saved", "not_saved", "uncertain"]
    message: str
    memory_ids: list[UUID] = Field(default_factory=list)
    attempted: bool = False
    reason: Literal["clarification", "unavailable", "write_uncertain", "budget"] | None = None


class Evidence(BaseModel):
    id: str
    subtask_id: str
    tool: str
    title: str
    content: str
    url: str | None = None
    article_id: int | None = None
    document_id: str | None = None
    revision_id: str | None = None
    format: str | None = None
    start: int | None = None
    end: int | None = None
    locator: dict[str, Any] = Field(default_factory=dict)
    body_hash: str | None = None
    truncated: bool = False
    update_state: str | None = None
    published_at: str | None = None
    source_kind: str | None = None
    source_task_id: str | None = None


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class Review(BaseModel):
    approved: bool
    missing_subtasks: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""


class Budget(BaseModel):
    max_model_calls: int = 12
    max_tool_calls: int = 12
    max_review_retries: int = 2
    max_seconds: float = 180.0
    model_calls: int = 0
    tool_calls: int = 0
    review_retries: int = 0
    elapsed_seconds: float = 0.0
    summary_calls: int = 0
    usage: list[dict[str, Any]] = Field(default_factory=list)
