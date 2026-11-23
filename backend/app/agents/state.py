"""Outer graph state. Deep Agents keeps tool messages within each execute call."""

from typing import Any, TypedDict


class TaskState(TypedDict, total=False):
    question: str
    user_id: str
    task_id: str
    conversation_id: str
    conversation_history: str
    profile: list[str]
    memory_only: bool
    memory_request: dict[str, Any] | None
    memory_result: dict[str, Any] | None
    plan: list[dict[str, Any]]
    current_index: int
    evidence: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    draft: str
    answer: str
    review: dict[str, Any]
    budget: dict[str, Any]
    termination: str
    tool_errors: list[str]
    trace: list[dict[str, Any]]
    next_step: str
