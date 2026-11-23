"""Real Mem0/PG/checkpoint validation with synthetic preferences and forbidden live chat.

Creates one isolated test user; removes only its recorded memory IDs and task threads.
The temporary SDK instance uses its own SQLite history, leaving shared history untouched.
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from langchain_core.messages import AIMessage
from langsmith import tracing_context

from app.agents import executor
from app.agents.checkpoints import open_checkpointer
from app.agents.memory.profile import PROFILE_AGENT_ID, ProfileMemory, profile_config
from app.core.config import settings
from app.db.session import close_db, get_worker_db_context, run_compatible_async
from app.repository import user as user_repo
from app.services.agent import AgentService
from eval.validate_conversation import SOURCE_ID, demo_search
from eval.validate_task_lifecycle import LifecycleModel

ROOT = Path(__file__).parents[1] / ".agent_data" / "personal-memory-validation"
PREFERENCE = "以后简报使用英文，先给结论"
COMMAND = "请记住：" + PREFERENCE


def memory_response() -> AIMessage:
    return AIMessage(content=json.dumps({"goals": [], "memory_request": {"text": PREFERENCE, "quote": COMMAND}}))


async def validate() -> None:
    if urlsplit(settings.RAG_OLLAMA_BASE_URL).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("Validation only permits local embedding endpoints")
    ROOT.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    os.environ.setdefault("MEM0_DIR", str(ROOT / "sdk"))
    from mem0 import Memory

    config = profile_config()
    config["history_db_path"] = str(ROOT / f"history-{identifier}.sqlite")
    model = LifecycleModel(responses=[memory_response()])
    service = AgentService()
    user_id: UUID | None = None
    report = {"live_chat_calls": 0, "synthetic_data_only": True}
    with (
        tracing_context(enabled=False),
        patch.object(settings, "LANGFUSE_ENABLED", False),
        patch("mem0.llms.openai.OpenAILLM.generate_response", side_effect=AssertionError("Live Mem0 chat forbidden")),
        patch("mem0.llms.ollama.OllamaLLM.generate_response", side_effect=AssertionError("Live Mem0 chat forbidden")),
        patch("app.agents.providers.OpenAIProvider.build_model", side_effect=AssertionError("Live chat forbidden")),
        patch("app.agents.providers.OllamaProvider.build_model", side_effect=AssertionError("Live chat forbidden")),
        patch("app.services.agent.get_llm_provider", return_value=Mock(build_model=lambda **_: model)),
        patch("app.agents.graph.get_llm_provider", return_value=Mock(build_model=lambda **_: model)),
        patch("app.services.agent.get_profile_memory", side_effect=lambda: profile),
        patch.object(executor, "business_tools", [demo_search]),
        patch.object(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"}),
    ):
        profile = ProfileMemory(Memory.from_config(config))
        try:
            async with get_worker_db_context() as db:
                user = await user_repo.create(
                    db, email=f"personal-memory-demo-{identifier}@example.invalid", hashed_password="!disabled"
                )
                user_id = user.id
            saved = await service.run(user_id, COMMAND)
            assert saved["memory_result"]["status"] == "saved" and model.calls == 1
            report["chat_write_saved"] = True
            profile.close()
            profile = ProfileMemory(Memory.from_config(config))
            read = await profile.recall(str(user_id), "不相关的市场问题")
            assert any(row["memory"] == PREFERENCE for row in read)
            assert await profile.recall(str(uuid4()), "不相关的市场问题") == []
            report["sdk_reopened_scoped_read"] = True
            model = LifecycleModel(
                responses=[
                    AIMessage(content='{"goals":["核查法规"]}'),
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": "research"}],
                    ),
                    AIMessage(content="已查证"),
                    AIMessage(content=f"The rule is effective.[{SOURCE_ID}]"),
                    AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":"fixture"}'),
                ]
            )
            researched = await service.run(user_id, "查询食品法规")
            assert researched["status"] == "completed" and researched["review"]["approved"]
            assert saved["conversation_id"] != researched["conversation_id"]
            assert PREFERENCE in model.seen[3]
            restored = await service.get_task(user_id, UUID(saved["task_id"]))
            assert restored["memory_result"] == saved["memory_result"]
            report["new_conversation_draft_receives_preference"] = True
            report["postgres_checkpoint_receipt_restored"] = True
            model = LifecycleModel(responses=[memory_response()])
            with patch.object(profile._memory.db, "add_history", side_effect=OSError("synthetic history failure")):
                uncertain = await service.run(user_id, COMMAND)
            assert uncertain["memory_result"]["status"] == "uncertain"
            current = await asyncio.to_thread(
                profile._memory.get_all,
                filters={"user_id": str(user_id), "agent_id": PROFILE_AGENT_ID, "source_task_id": uncertain["task_id"]},
                top_k=2,
                show_expired=True,
            )
            assert len(current["results"]) == 1
            report["history_failure_reports_uncertain_after_vector_write"] = True
            report["scripted_model_calls"] = 7
        finally:
            if user_id is not None:
                async with get_worker_db_context() as db:
                    user = await user_repo.get_by_id(db, user_id)
                    if user is None or user.email != f"personal-memory-demo-{identifier}@example.invalid":
                        raise RuntimeError("Cleanup identity mismatch")
                listing = await service.list_tasks(user_id, limit=100)
                task_ids = {task["task_id"] for task in listing["items"]}
                current = await asyncio.to_thread(
                    profile._memory.get_all,
                    filters={"user_id": str(user_id), "agent_id": PROFILE_AGENT_ID},
                    top_k=10,
                    show_expired=True,
                )
                for row in current["results"]:
                    if row.get("metadata", {}).get("source_task_id") not in task_ids:
                        raise RuntimeError("Refusing to remove an unrecorded validation memory")
                    await profile.forget(str(user_id), row["id"])
                async with open_checkpointer() as saver:
                    for task_id in task_ids:
                        await saver.adelete_thread(f"{user_id}:{task_id}")
                async with get_worker_db_context() as db:
                    await user_repo.delete(db, db_user=user)
                report["isolated_records_cleaned"] = True
            profile.close()
            await close_db()
    (ROOT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))  # noqa: T201 -- synthetic validation results only


if __name__ == "__main__":
    run_compatible_async(validate())
