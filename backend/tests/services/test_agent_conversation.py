"""Conversation orchestration uses short transactions and authenticated ownership."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core.exceptions import AlreadyExistsError, NotFoundError
from app.db.models.conversation import Conversation, ConversationTurn
from app.services.agent import AgentService


@pytest.mark.anyio
async def test_begin_followup_loads_sources_and_creates_independent_task() -> None:
    user_id, conversation_id, previous_task, task_id = (uuid4() for _ in range(4))
    row = Conversation(id=conversation_id, user_id=user_id)
    turn = ConversationTurn(
        task_id=previous_task,
        question="客户动态",
        answer="第一条、第二条[E-123456789abc]",
        status="completed",
        evidence=[{"id": "E-123456789abc", "content": "原文", "url": "https://example.org"}],
    )
    db = AsyncMock()

    @asynccontextmanager
    async def context() -> AsyncIterator[AsyncMock]:
        yield db

    with (
        patch("app.services.agent.get_worker_db_context", context),
        patch("app.services.agent.conversation_repo") as repo,
    ):
        repo.get_owned = AsyncMock(return_value=row)
        repo.has_running = AsyncMock(return_value=False)
        repo.list_turns = AsyncMock(return_value=([turn], 1))
        repo.create_turn = AsyncMock()
        result_id, history, evidence = await AgentService()._begin_turn(user_id, "展开第二条", task_id, conversation_id)
        assert result_id == conversation_id
        assert "第一条、第二条" in history
        assert evidence[0]["source_task_id"] == str(previous_task)
        repo.get_owned.assert_awaited_once_with(db, conversation_id=conversation_id, user_id=user_id, lock=True)
        repo.create_turn.assert_awaited_once_with(
            db, conversation_id=conversation_id, task_id=task_id, question="展开第二条"
        )


@pytest.mark.anyio
@pytest.mark.parametrize("owned,running,error", [(False, False, NotFoundError), (True, True, AlreadyExistsError)])
async def test_begin_rejects_foreign_or_running_conversation(
    owned: bool, running: bool, error: type[Exception]
) -> None:
    db = AsyncMock()

    @asynccontextmanager
    async def context() -> AsyncIterator[AsyncMock]:
        yield db

    with (
        patch("app.services.agent.get_worker_db_context", context),
        patch("app.services.agent.conversation_repo") as repo,
    ):
        repo.get_owned = AsyncMock(return_value=Conversation(id=uuid4()) if owned else None)
        repo.has_running = AsyncMock(return_value=running)
        repo.unfinished_turns = AsyncMock(return_value=[])
        repo.create_turn = AsyncMock()
        with pytest.raises(error):
            await AgentService()._begin_turn(uuid4(), "查询客户", uuid4(), uuid4())
        repo.create_turn.assert_not_awaited()
