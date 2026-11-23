"""HTTP shapes for research tasks and explicit profile memory changes."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.agents.models import MemoryResult


class AgentTaskCreate(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    conversation_id: UUID | None = None


TaskStatus = Literal["running", "interrupted", "completed", "stopped", "failed", "abandoned"]


class TaskRecoveryRead(BaseModel):
    can_resume: bool = False
    blocked_reason: str | None = None
    budget_status: Literal["known", "uncertain", "unavailable"] = "unavailable"
    pending_nodes: list[str] = Field(default_factory=list)


class AgentTaskSummaryRead(BaseModel):
    task_id: UUID
    conversation_id: UUID | None = None
    question: str
    status: TaskStatus = "running"
    termination: str | None = None
    pause_requested: bool = False
    recovery: TaskRecoveryRead = Field(default_factory=TaskRecoveryRead)


class AgentTaskList(BaseModel):
    items: list[AgentTaskSummaryRead]
    total: int


class AgentTaskRead(AgentTaskSummaryRead):
    answer: str | None = None
    plan: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    review: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    memory_result: MemoryResult | None = None


class ConversationTurnRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: UUID
    question: str
    answer: str | None
    status: str
    evidence: list[dict[str, Any]]
    created_at: datetime


class ConversationRead(BaseModel):
    id: UUID
    items: list[ConversationTurnRead]
    total: int


class ProfileMemoryCreate(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class ProfileMemoryUpdate(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class ProfileMemoryRead(BaseModel):
    id: UUID
    memory: str


class ProfileMemoryList(BaseModel):
    items: list[ProfileMemoryRead]
