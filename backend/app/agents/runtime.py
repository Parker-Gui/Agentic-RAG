"""Single-process execution ownership; queries and drain signals never wait for a run."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from uuid import UUID

from langgraph.runtime import RunControl

from app.core.exceptions import AlreadyExistsError


@dataclass
class ActiveRun:
    user_id: UUID
    conversation_id: UUID
    task_id: UUID
    control: RunControl = field(default_factory=RunControl)


class RunRegistry:
    """Methods run synchronously on the service's event-loop thread."""

    def __init__(self) -> None:
        self._conversations: dict[UUID, ActiveRun] = {}

    def find(self, conversation_id: UUID) -> ActiveRun | None:
        return self._conversations.get(conversation_id)

    @contextmanager
    def claim(self, user_id: UUID, conversation_id: UUID, task_id: UUID) -> Iterator[ActiveRun]:
        if conversation_id in self._conversations:
            raise AlreadyExistsError(message="对话已有执行中的任务。", details={"reason": "task_running"})
        run = ActiveRun(user_id, conversation_id, task_id)
        self._conversations[conversation_id] = run
        try:
            yield run
        finally:
            if self._conversations.get(conversation_id) is run:
                del self._conversations[conversation_id]


registry = RunRegistry()
