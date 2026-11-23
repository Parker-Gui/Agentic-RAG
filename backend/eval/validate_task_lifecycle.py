"""Real DB/process lifecycle validation with synthetic sources and guarded models.

Run write/read as separate processes; crash/crash-read exercises abrupt exit.
Only isolated test users/threads named in the manifest are removed.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langsmith import tracing_context
from pydantic import Field

from app.agents import executor
from app.agents.checkpoints import open_checkpointer
from app.core.config import settings
from app.core.exceptions import AlreadyExistsError, NotFoundError
from app.core.security import create_access_token
from app.db.session import close_db, get_worker_db_context, run_compatible_async
from app.main import create_app
from app.repository import user as user_repo
from app.services.agent import AgentService
from app.services.artifact import ArtifactService
from eval.validate_conversation import ANSWER, demo_search

ROOT = Path(__file__).parents[1] / ".agent_data" / "lifecycle-validation"


class LifecycleModel(BaseChatModel):
    responses: list[AIMessage]
    calls: int = 0
    seen: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "lifecycle-fixture"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "LifecycleModel":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(str(messages))
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def responses(*, remaining: bool = False) -> list[AIMessage]:
    tail = [
        AIMessage(content=ANSWER),
        AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":"fixture"}'),
    ]
    if remaining:
        return tail
    result = [AIMessage(content='{"goals":["客户一","客户二","客户三"]}')]
    for index in range(3):
        result.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_rag_tool", "args": {"query": f"synthetic-{index}"}, "id": f"call-{index}"}
                    ],
                ),
                AIMessage(content="查证结束"),
            ]
        )
    return [*result, *tail]


async def validate(stage: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    crashing = stage.startswith("crash")
    manifest = ROOT / ("crash-manifest.json" if crashing else "manifest.json")
    writing = stage in {"write", "crash"}
    started, release = asyncio.Event(), asyncio.Event()
    tool_calls = 0
    model = LifecycleModel(responses=responses(remaining=not writing and not crashing))

    @tool("search_rag_tool")
    async def search(query: str) -> str:
        """Fixed synthetic source; never calls external tools."""
        nonlocal tool_calls
        tool_calls += 1
        if writing and tool_calls == 3:
            started.set()
            await release.wait()
        return await demo_search.ainvoke({"query": query})

    provider = Mock(build_model=lambda **_: model)
    with (
        tracing_context(enabled=False),
        patch.object(settings, "LANGFUSE_ENABLED", False),
        patch("app.services.agent.get_llm_provider", return_value=provider),
        patch("app.agents.graph.get_llm_provider", return_value=provider),
        patch("app.agents.providers.OpenAIProvider.build_model", side_effect=AssertionError("Live models forbidden")),
        patch("app.agents.providers.OllamaProvider.build_model", side_effect=AssertionError("Live models forbidden")),
        patch("app.services.agent.get_profile_memory", return_value=Mock(recall=AsyncMock(return_value=[]))),
        patch.object(executor, "business_tools", [search]),
        patch.object(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"}),
    ):
        service = AgentService()
        if writing:
            if manifest.exists():
                raise RuntimeError("Complete the paired read/cleanup stage before rerunning write")
            async with get_worker_db_context() as db:
                user = await user_repo.create(
                    db, email=f"task-lifecycle-demo-{uuid4()}@example.invalid", hashed_password="!disabled"
                )
            data = {"user_id": str(user.id)}
            manifest.write_text(json.dumps(data), encoding="utf-8")
            job = asyncio.create_task(service.run(user.id, "比较三家合成客户动态"))
            await asyncio.wait_for(started.wait(), 10)
            listing = await service.list_tasks(user.id)
            task = listing["items"][0]
            data.update(task_id=task["task_id"], conversation_id=task["conversation_id"])
            manifest.write_text(json.dumps(data), encoding="utf-8")
            if crashing:
                # Exit only this validation process, with a known private test user.
                os._exit(0)
            task_id = UUID(task["task_id"])
            async with AsyncClient(
                transport=ASGITransport(app=create_app()), base_url="http://lifecycle-test"
            ) as client:
                response = await client.post(
                    f"/api/v1/agent/tasks/{task_id}/pause",
                    headers={"Authorization": f"Bearer {create_access_token(user.id)}"},
                )
                assert response.status_code == 202 and response.json()["pause_requested"]
            try:
                await service.resume(user.id, task_id)
                raise AssertionError("Concurrent resume unexpectedly executed")
            except AlreadyExistsError:
                pass
            release.set()
            result = await asyncio.wait_for(job, 10)
            assert result["status"] == "interrupted" and result["recovery"]["can_resume"]
            assert result["current_index"] == 3 and result["budget"]["model_calls"] == 7
            assert result["recovery"]["pending_nodes"] == ["draft"] and tool_calls == 3
            report = {"stage": stage, "paused_after_subtask": 3, "model_calls": model.calls, "paid_api_calls": 0}
        else:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            user_id, task_id = UUID(data["user_id"]), UUID(data["task_id"])
            try:
                view = await service.get_task(user_id, task_id)
                try:
                    await service.get_task(uuid4(), task_id)
                    raise AssertionError("Foreign user read checkpoint")
                except NotFoundError:
                    pass
                if crashing:
                    assert view["status"] == "interrupted" and not view["recovery"]["can_resume"]
                    assert view["recovery"]["budget_status"] == "uncertain"
                    try:
                        await service.resume(user_id, task_id)
                        raise AssertionError("Unsafe resume unexpectedly executed")
                    except AlreadyExistsError:
                        pass
                    assert model.calls == 0
                    assert (await service.abandon(user_id, task_id))["status"] == "abandoned"
                    fresh = await service.run(user_id, "放弃后重新研究", conversation_id=UUID(data["conversation_id"]))
                    assert fresh["status"] == "completed"
                    report = {
                        "stage": stage,
                        "hard_exit_rejected": True,
                        "conversation_released": True,
                        "paid_api_calls": 0,
                    }
                else:
                    assert view["recovery"]["can_resume"] and view["budget"]["model_calls"] == 7
                    result = await service.resume(user_id, task_id)
                    assert result["status"] == "completed" and result["answer"] == ANSWER
                    assert model.calls == 2 and tool_calls == 0 and result["budget"]["model_calls"] == 9
                    await service.resume(user_id, task_id)
                    assert model.calls == 2
                    async with AsyncClient(
                        transport=ASGITransport(app=create_app()), base_url="http://lifecycle-test"
                    ) as client:
                        headers = {"Authorization": f"Bearer {create_access_token(user_id)}"}
                        response = await client.get("/api/v1/agent/tasks?status=completed", headers=headers)
                        assert response.status_code == 200 and response.json()["total"] == 1
                        response = await client.post(f"/api/v1/agent/tasks/{task_id}/resume", headers=headers)
                        assert (
                            response.status_code == 200
                            and response.json()["status"] == "completed"
                            and model.calls == 2
                        )
                    exports = await asyncio.gather(*[ArtifactService().create(user_id, task_id) for _ in range(2)])
                    assert sum(created for _, created in exports) == 1 and model.calls == 2
                    report = {
                        "stage": stage,
                        "resumed_model_calls": 2,
                        "repeated_tool_calls": 0,
                        "total_model_calls": 9,
                        "export_extra_calls": 0,
                        "paid_api_calls": 0,
                    }
            finally:
                async with get_worker_db_context() as db:
                    user = await user_repo.get_by_id(db, user_id)
                    if user is None or not user.email.startswith("task-lifecycle-demo-"):
                        raise RuntimeError("Cleanup identity is not an isolated lifecycle test user")
                listing = await service.list_tasks(user_id)
                async with open_checkpointer() as saver:
                    for task in listing["items"]:
                        await saver.adelete_thread(f"{user_id}:{task['task_id']}")
                async with get_worker_db_context() as db:
                    await user_repo.delete(db, db_user=user)
                manifest.unlink()
        async with open_checkpointer() as saver:
            report["checkpointer"] = type(saver).__name__
        (ROOT / f"{stage}-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))  # noqa: T201 -- validation report
    await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["write", "read", "crash", "crash-read"], required=True)
    run_compatible_async(validate(parser.parse_args().stage))


if __name__ == "__main__":
    main()
