"""JWT -> real service -> checkpointed graph -> Mem0 adapter, across new conversations."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.agents import executor
from app.agents import graph as graph_module
from app.agents.memory.profile import ProfileMemory
from app.agents.runtime import RunRegistry
from app.api.deps import get_agent_service
from app.db.session import get_db_session
from app.main import create_app
from app.services import agent as module
from tests.agents.test_personal_memory_loop import COMMAND, PREFERENCE, ProfileSDK, plan_response
from tests.agents.test_research_workflow import _id, fake_search
from tests.agents.test_work_context import RecordingModel
from tests.helpers import FakeUser, auth_headers_for
from tests.services.test_task_lifecycle import Harness


@pytest.mark.anyio
@pytest.mark.parametrize("write_failure", [False, True])
async def test_chat_memory_receipt_and_new_conversation_use_real_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_db: AsyncMock,
    mock_user_repo: Mock,
    fake_user: FakeUser,
    write_failure: bool,
) -> None:
    harness = Harness(tmp_path)
    harness.user = fake_user.id
    harness.model = RecordingModel(responses=[plan_response()])
    sdk = ProfileSDK()
    sdk.fail = write_failure
    memory = ProfileMemory(cast(Any, sdk))
    monkeypatch.setattr(module, "registry", RunRegistry())
    monkeypatch.setattr(module, "open_checkpointer", harness.saver)
    monkeypatch.setattr(module, "get_profile_memory", lambda: memory)
    monkeypatch.setattr(module, "get_llm_provider", lambda: Mock(build_model=lambda **_: harness.model))
    monkeypatch.setattr(graph_module, "get_llm_provider", lambda: Mock(build_model=lambda **_: harness.model))
    for name, method in (("_begin_turn", harness.begin), ("_owned_task", harness.owned), ("_mark", harness.mark)):
        monkeypatch.setattr(harness.service, name, method)
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    app = create_app()
    app.dependency_overrides[get_db_session] = lambda: mock_db
    app.dependency_overrides[get_agent_service] = lambda: harness.service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        headers = auth_headers_for(fake_user)
        first = await client.post("/api/v1/agent/tasks", json={"question": COMMAND}, headers=headers)
        assert first.status_code == 201
        receipt = first.json()["memory_result"]
        assert receipt["status"] == ("uncertain" if write_failure else "saved")
        assert "private-" not in first.text
        saved = await client.get(f"/api/v1/agent/tasks/{first.json()['task_id']}", headers=headers)
        assert saved.json()["memory_result"] == receipt

        harness.model = RecordingModel(
            responses=[
                AIMessage(content='{"goals":["查法规"]}'),
                AIMessage(
                    content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": "second"}]
                ),
                AIMessage(content="已查证"),
                AIMessage(content=f"The rule is effective.[{_id()}]"),
                AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":"fixture"}'),
            ]
        )
        second = await client.post("/api/v1/agent/tasks", json={"question": "查询食品法规"}, headers=headers)
        assert second.status_code == 201 and second.json()["termination"] == "completed"
        assert second.json()["conversation_id"] != first.json()["conversation_id"]
        assert PREFERENCE in str(harness.model.seen[3])
        assert second.json()["memory_result"] is None
        assert sdk.writes == 1
        assert PREFERENCE not in str(second.json()["evidence"])
        assert PREFERENCE not in str(second.json()["claims"])
