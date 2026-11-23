"""Real PostgreSQL/checkpoint validation with scripted models; no paid API calls.

Run --stage write, then --stage read in a new process. Read removes the temporary
database user and its cascaded conversations. Local validation checkpoints remain.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from app.agents import executor
from app.core.exceptions import NotFoundError
from app.core.security import create_access_token
from app.db.session import close_db, get_worker_db_context, run_compatible_async
from app.main import create_app
from app.repository import user as user_repo
from app.schemas.agent import ConversationRead
from app.services.agent import AgentService
from app.services.artifact import ArtifactService

ROOT = Path(__file__).resolve().parents[1] / ".agent_data" / "conversation-validation"
SOURCE = "法规已经生效。适用于企业A。"
SOURCE_ID = "E-" + hashlib.sha256(f"https://example.org/demo\n{SOURCE}".encode()).hexdigest()[:12]
ANSWER = f"1. 法规已经生效。[{SOURCE_ID}]\n2. 适用于企业A。[{SOURCE_ID}]"


@tool("search_rag_tool")
async def demo_search(query: str) -> str:
    """Return a synthetic source for orchestration validation, not real research."""
    del query
    return json.dumps({"results": [{"title": "演示法规", "url": "https://example.org/demo", "content": SOURCE}]})


class DemoModel(BaseChatModel):
    calls: int = 0
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "conversation-demo"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DemoModel":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.seen.append(messages)
        responses = [
            AIMessage(content='{"goals":["核查法规适用范围"]}'),
            AIMessage(
                content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": "demo-call"}]
            ),
            AIMessage(content="已查证"),
            AIMessage(content=ANSWER),
            AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":"演示来源支持"}'),
        ]
        response = responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


async def run(stage: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "state.json"
    model = DemoModel()
    profile = Mock(recall=AsyncMock(return_value=[]))
    with (
        patch("app.services.agent.get_llm_provider", return_value=Mock(build_model=Mock(return_value=model))),
        patch("app.agents.graph.get_llm_provider", return_value=Mock(build_model=Mock(return_value=model))),
        patch("app.agents.providers.OpenAIProvider.build_model", side_effect=AssertionError("Live models forbidden")),
        patch("app.agents.providers.OllamaProvider.build_model", side_effect=AssertionError("Live models forbidden")),
        patch("app.services.agent.get_profile_memory", return_value=profile),
        patch.object(executor, "business_tools", [demo_search]),
        patch.object(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"}),
    ):
        service = AgentService()
        if stage == "write":
            if manifest.exists():
                raise RuntimeError("上次验证尚未清理，请先执行 --stage read")
            async with get_worker_db_context() as db:
                user = await user_repo.create(
                    db, email=f"conversation-demo-{uuid4()}@example.invalid", hashed_password="!disabled"
                )
                user_id = user.id
            # Persist cleanup identity before running the model.
            data = {"user_id": str(user_id)}
            manifest.write_text(json.dumps(data), encoding="utf-8")
            first = await service.run(user_id, "列出法规的两条信息")
            assert first["termination"] == "completed"
            assert first["budget"]["model_calls"] == 5
            data.update(conversation_id=first["conversation_id"], task_id=first["task_id"])
            manifest.write_text(json.dumps(data), encoding="utf-8")
            # Concurrent export uses the real unique constraint, not a mocked repository.
            exports = await asyncio.gather(
                *[ArtifactService().create(user_id, UUID(first["task_id"])) for _ in range(2)]
            )
            assert sum(created for _, created in exports) == 1
            assert exports[0][0]["id"] == exports[1][0]["id"]
            metadata = exports[0][0]
            assert model.calls == 5  # Export never calls the model.
            data.update(
                artifact_id=str(metadata["id"]),
                artifact_sha256=metadata["sha256"],
                artifact_size=metadata["size_bytes"],
            )
            manifest.write_text(json.dumps(data), encoding="utf-8")
            result = {
                "stage": "write",
                "completed": True,
                "model_calls": model.calls,
                "concurrent_export_reused_id": True,
                "export_model_calls": 0,
                "paid_api_calls": 0,
            }
        else:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            user_id = UUID(data["user_id"])
            try:
                conversation_id = UUID(data["conversation_id"])
                before = await service.get_conversation(user_id, conversation_id)
                assert before["total"] == 1
                # Exercise response serialization after the transaction is closed.
                assert ConversationRead.model_validate(before).items[0].answer == ANSWER
                artifact_id = UUID(data["artifact_id"])
                artifacts = ArtifactService()
                metadata = await artifacts.get(user_id, artifact_id)
                payload, filename = await artifacts.download(user_id, artifact_id)
                assert hashlib.sha256(payload).hexdigest() == data["artifact_sha256"] == metadata["sha256"]
                assert len(payload) == data["artifact_size"] == metadata["size_bytes"]
                assert ANSWER in payload.decode("utf-8")
                assert SOURCE_ID in payload.decode("utf-8")
                (ROOT / "brief-demo.md").write_bytes(payload)
                repeat, created = await artifacts.create(user_id, UUID(data["task_id"]))
                assert not created and repeat["id"] == artifact_id
                listed = await artifacts.list_for_task(user_id, UUID(data["task_id"]))
                assert listed["total"] == 1
                try:
                    await artifacts.download(uuid4(), artifact_id)
                except NotFoundError:
                    pass
                else:
                    raise AssertionError("foreign user downloaded artifact")
                # Drive the real HTTP download with real JWT and database authentication.
                async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as client:
                    response = await client.get(
                        metadata["download_url"], headers={"Authorization": f"Bearer {create_access_token(user_id)}"}
                    )
                    assert response.status_code == 200
                    assert response.content == payload
                    assert response.headers["content-type"] == "text/markdown; charset=utf-8"
                    assert response.headers["content-disposition"] == f'attachment; filename="{filename}"'
                assert model.calls == 0  # New-process read/download/reuse requires no model.
                second = await service.run(user_id, "展开第二条", conversation_id=conversation_id)
                assert second["termination"] == "completed"
                assert second["task_id"] != data["task_id"]
                assert second["budget"]["model_calls"] == 5
                for index in (0, 1, 3):
                    rendered = "\n".join(str(message.content) for message in model.seen[index])
                    assert all(line in rendered for line in ANSWER.splitlines())
                assert any(row.get("source_task_id") == data["task_id"] for row in second["evidence"])
                after = await service.get_conversation(user_id, conversation_id)
                assert after["total"] == 2
                try:
                    await service.get_conversation(uuid4(), conversation_id)
                except NotFoundError:
                    pass
                else:
                    raise AssertionError("foreign user accessed conversation")
                resumed = await service.resume(user_id, UUID(second["task_id"]))
                assert resumed["answer"] == second["answer"]
                assert model.calls == 5
                result = {
                    "stage": "read",
                    "turns": 2,
                    "history_in_plan_execute_draft": True,
                    "sources_preserved": True,
                    "foreign_user_denied": True,
                    "resume_reused_answer": True,
                    "independent_budget": True,
                    "paid_api_calls": 0,
                    "artifact_download_persisted": True,
                    "artifact_hash_matches": True,
                    "artifact_foreign_user_denied": True,
                    "authenticated_http_download": True,
                    "export_model_calls": 0,
                }
            finally:
                async with get_worker_db_context() as db:
                    user = await user_repo.get_by_id(db, user_id)
                    if user is not None:
                        await user_repo.delete(db, user)
                manifest.unlink(missing_ok=True)
                await close_db()
        (ROOT / f"{stage}-report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))  # noqa: T201 — CLI validation report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["write", "read"])
    run_compatible_async(run(parser.parse_args().stage))
