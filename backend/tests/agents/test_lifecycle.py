"""A receipt identifies one saved boundary; it cannot invent missing budget."""

from uuid import uuid4

import pytest
from langgraph.types import StateSnapshot

from app.agents.lifecycle import complete_budget, task_view
from app.agents.models import Budget
from app.db.models.conversation import ConversationTurn


@pytest.mark.parametrize("change", ["missing_usage", "mismatched_id", "missing_receipt", "negative_count"])
def test_invalid_or_stale_receipt_cannot_resume(change: str) -> None:
    task_id, conversation_id = uuid4(), uuid4()
    budget = Budget().model_dump()
    receipt = {"version": 1, "kind": "clean_boundary", "checkpoint_id": "saved"}
    if change == "missing_usage":
        budget.pop("usage")
    elif change == "mismatched_id":
        receipt["checkpoint_id"] = "old"
    elif change == "missing_receipt":
        receipt = {}
    else:
        budget["model_calls"] = -1
    row = ConversationTurn(
        task_id=task_id, conversation_id=conversation_id, question="问题", status="interrupted", recovery=receipt
    )
    snapshot = StateSnapshot(
        values={"task_id": str(task_id), "budget": budget},
        next=("draft",),
        config={"configurable": {"checkpoint_id": "saved"}},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )
    assert task_view(row, snapshot)["recovery"]["can_resume"] is False


def test_missing_budget_does_not_become_zero_consumption() -> None:
    assert complete_budget({"budget": {"max_model_calls": 12}}) is False
