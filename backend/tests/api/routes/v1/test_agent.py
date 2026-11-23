"""The research API must derive ownership from the authenticated caller."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.core.exceptions import NotFoundError
from tests.helpers import FakeUser, auth_headers_for


@pytest.mark.anyio
async def test_research_task_requires_login(client: AsyncClient) -> None:
    response = await client.post("/api/v1/agent/tasks", json={"question": "分析近期法规变化"})
    assert response.status_code == 401


@pytest.mark.anyio
async def test_research_task_passes_authenticated_user_id(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    task_id = uuid4()
    output = {
        "task_id": str(task_id),
        "question": "分析近期法规变化",
        "answer": "没有证据",
        "termination": "insufficient_evidence",
    }
    with patch("app.services.agent.AgentService.run", new_callable=AsyncMock, return_value=output) as run:
        response = await client.post(
            "/api/v1/agent/tasks",
            json={"question": "分析近期法规变化"},
            headers=auth_headers_for(fake_user),
        )
    assert response.status_code == 201
    assert response.json()["task_id"] == str(task_id)
    run.assert_awaited_once_with(fake_user.id, "分析近期法规变化", conversation_id=None)


@pytest.mark.anyio
async def test_profile_mutation_uses_authenticated_user_id(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    memory_id = uuid4()
    with patch("app.services.agent.AgentService.forget", new_callable=AsyncMock) as forget:
        response = await client.delete(f"/api/v1/agent/profile/{memory_id}", headers=auth_headers_for(fake_user))
    assert response.status_code == 204
    forget.assert_awaited_once_with(fake_user.id, memory_id)


@pytest.mark.anyio
async def test_missing_chat_key_returns_503_before_profile_initialization(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    with patch("app.services.agent.get_profile_memory") as memory:
        response = await client.post(
            "/api/v1/agent/tasks",
            json={"question": "分析近期法规变化"},
            headers=auth_headers_for(fake_user),
        )
    assert response.status_code == 503
    assert response.json()["code"] == "EXTERNAL_SERVICE_ERROR"
    assert "OPENAI_API_KEY" in response.json()["detail"]
    memory.assert_not_called()


@pytest.mark.anyio
async def test_followup_passes_conversation_id(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    conversation_id = uuid4()
    output = {"task_id": str(uuid4()), "conversation_id": str(conversation_id), "question": "展开第二条"}
    with patch("app.services.agent.AgentService.run", new_callable=AsyncMock, return_value=output) as run:
        response = await client.post(
            "/api/v1/agent/tasks",
            json={"question": "展开第二条", "conversation_id": str(conversation_id)},
            headers=auth_headers_for(fake_user),
        )
    assert response.status_code == 201
    assert response.json()["conversation_id"] == str(conversation_id)
    run.assert_awaited_once_with(fake_user.id, "展开第二条", conversation_id=conversation_id)


@pytest.mark.anyio
async def test_conversation_read_requires_login(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/agent/conversations/{uuid4()}")
    assert response.status_code == 401


@pytest.mark.anyio
async def test_conversation_read_uses_authenticated_owner(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    conversation_id = uuid4()
    with patch("app.services.agent.AgentService.get_conversation", new_callable=AsyncMock) as get:
        get.return_value = {"id": conversation_id, "items": [], "total": 0}
        response = await client.get(
            f"/api/v1/agent/conversations/{conversation_id}?limit=10", headers=auth_headers_for(fake_user)
        )
    assert response.status_code == 200
    get.assert_awaited_once_with(fake_user.id, conversation_id, skip=0, limit=10)


@pytest.mark.anyio
async def test_foreign_conversation_returns_not_found(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    with patch("app.services.agent.AgentService.get_conversation", new_callable=AsyncMock, side_effect=NotFoundError):
        response = await client.get(f"/api/v1/agent/conversations/{uuid4()}", headers=auth_headers_for(fake_user))
    assert response.status_code == 404


@pytest.mark.anyio
async def test_conversation_pagination_bounds(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    response = await client.get(f"/api/v1/agent/conversations/{uuid4()}?limit=0", headers=auth_headers_for(fake_user))
    assert response.status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["pause", "abandon"])
async def test_task_control_requires_login(client: AsyncClient, operation: str) -> None:
    assert (await client.post(f"/api/v1/agent/tasks/{uuid4()}/{operation}")).status_code == 401


@pytest.mark.anyio
async def test_task_list_requires_login(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/agent/tasks")).status_code == 401


@pytest.mark.anyio
async def test_pause_acknowledges_pending_drain(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    task_id = uuid4()
    result = {"task_id": str(task_id), "question": "合成问题", "status": "running", "pause_requested": True}
    with patch("app.services.agent.AgentService.pause", new_callable=AsyncMock, return_value=(result, True)) as pause:
        response = await client.post(f"/api/v1/agent/tasks/{task_id}/pause", headers=auth_headers_for(fake_user))
    assert response.status_code == 202 and response.json()["pause_requested"] is True
    pause.assert_awaited_once_with(fake_user.id, task_id)


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["pause", "abandon", "resume"])
async def test_foreign_task_control_is_not_found(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser, operation: str
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    with patch(f"app.services.agent.AgentService.{operation}", new_callable=AsyncMock, side_effect=NotFoundError):
        response = await client.post(f"/api/v1/agent/tasks/{uuid4()}/{operation}", headers=auth_headers_for(fake_user))
    assert response.status_code == 404 and response.json()["code"] == "NOT_FOUND"


@pytest.mark.anyio
async def test_task_list_filters_and_hides_checkpoint_payload(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    item = {
        "task_id": str(uuid4()),
        "question": "合成任务",
        "status": "interrupted",
        "profile": ["PRIVATE"],
        "evidence": [{"content": "PRIVATE"}],
    }
    with patch(
        "app.services.agent.AgentService.list_tasks", new_callable=AsyncMock, return_value={"items": [item], "total": 1}
    ) as listing:
        response = await client.get(
            "/api/v1/agent/tasks?status=interrupted&limit=5", headers=auth_headers_for(fake_user)
        )
    assert response.status_code == 200 and "PRIVATE" not in response.text
    listing.assert_awaited_once_with(fake_user.id, conversation_id=None, status="interrupted", skip=0, limit=5)


@pytest.mark.anyio
@pytest.mark.parametrize("query", ["limit=0", "status=invalid", "skip=-1"])
async def test_task_list_validates_filter(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser, query: str
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    response = await client.get(f"/api/v1/agent/tasks?{query}", headers=auth_headers_for(fake_user))
    assert response.status_code == 422
