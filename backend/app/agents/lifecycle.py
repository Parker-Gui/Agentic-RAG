"""Lifecycle projections; LangGraph remains the owner of execution state."""

import math
from typing import Any, Literal

from langgraph.types import StateSnapshot
from pydantic import BaseModel

from app.agents.models import Budget, TerminationReason
from app.core.exceptions import ExternalServiceError
from app.db.models.conversation import ConversationTurn

TERMINAL = {"completed", "stopped", "failed", "abandoned"}


class ResumeReceipt(BaseModel):
    version: Literal[1]
    kind: Literal["clean_boundary"]
    checkpoint_id: str


def checkpoint_id(snapshot: StateSnapshot | None) -> str | None:
    return snapshot.config.get("configurable", {}).get("checkpoint_id") if snapshot else None


def complete_budget(state: dict[str, Any]) -> bool:
    raw = state.get("budget")
    if not isinstance(raw, dict) or not set(Budget.model_fields) <= raw.keys():
        return False
    try:
        budget = Budget.model_validate(raw)
    except ValueError:
        return False
    return (
        0 <= budget.model_calls <= budget.max_model_calls
        and 0 <= budget.tool_calls <= budget.max_tool_calls
        and 0 <= budget.summary_calls <= budget.model_calls
        and 0 <= budget.review_retries <= budget.max_review_retries
        and 0 < budget.max_seconds
        and math.isfinite(budget.max_seconds)
        and budget.elapsed_seconds >= 0
        and math.isfinite(budget.elapsed_seconds)
    )


def terminal_status(state: dict[str, Any]) -> str:
    reason = state.get("termination")
    if reason == TerminationReason.COMPLETED.value:
        return "completed"
    if reason and reason != TerminationReason.EXECUTION_ERROR.value:
        return "stopped"
    return "failed"


def task_view(
    row: ConversationTurn, snapshot: StateSnapshot | None, *, active: bool = False, pausing: bool = False
) -> dict[str, Any]:
    values = dict(snapshot.values) if snapshot and snapshot.values else {}
    if values and (
        values.get("task_id") != str(row.task_id)
        or values.get("conversation_id") not in (None, str(row.conversation_id))
    ):
        raise ExternalServiceError(message="任务检查点与登记不一致。")
    pending = list(snapshot.next) if snapshot else []
    final = bool(values.get("answer") and values.get("termination") and not pending)
    status = row.status
    reason: str | None = None
    can_resume = False
    budget_status = "known" if complete_budget(values) else "unavailable"
    if status == "abandoned":
        reason = "task_abandoned"
    elif active:
        status, reason = "running", "task_running"
    elif final:
        status, reason = terminal_status(values), "task_finished"
    elif not values:
        status, reason = "failed", "checkpoint_missing"
    elif status in TERMINAL:
        reason = "task_finished"
    else:
        status = "interrupted"
        try:
            receipt = ResumeReceipt.model_validate(row.recovery)
        except ValueError:
            receipt = None
        can_resume = bool(
            pending
            and complete_budget(values)
            and receipt is not None
            and receipt.checkpoint_id == checkpoint_id(snapshot)
        )
        if can_resume:
            budget_status = "known"
        else:
            budget_status, reason = "uncertain", "unsafe_checkpoint"
    return {
        **values,
        "task_id": str(row.task_id),
        "conversation_id": str(row.conversation_id),
        "question": row.question,
        "status": status,
        "termination": values.get("termination") or row.termination,
        "answer": values.get("answer") or row.answer,
        "pause_requested": pausing,
        "recovery": {
            "can_resume": can_resume,
            "blocked_reason": reason,
            "budget_status": budget_status,
            "pending_nodes": pending,
        },
    }
