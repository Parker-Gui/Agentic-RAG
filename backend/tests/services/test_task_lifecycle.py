"""Real LangGraph/Deep Agents boundaries with controlled I/O and no paid models."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents import executor
from app.agents import graph as graph_module
from app.agents.runtime import RunRegistry
from app.core.exceptions import AlreadyExistsError, ExternalServiceError, NotFoundError
from app.db.models.conversation import ConversationTurn
from app.services import agent as module
from eval.validate_conversation import ANSWER, SOURCE_ID, DemoModel, demo_search


class Harness:
    def __init__(self, root: Path) -> None:
        self.user = uuid4()
        self.rows: dict[UUID, ConversationTurn] = {}
        self.model: Any = DemoModel()
        self.service = module.AgentService()
        self.root = root

    async def begin(
        self,
        user_id: UUID,
        question: str,
        task_id: UUID,
        conversation_id: UUID | None,
        *,
        new_conversation_id: UUID | None = None,
    ) -> tuple[UUID, str, list[dict[str, Any]]]:
        identifier = conversation_id or new_conversation_id or uuid4()
        self.rows[task_id] = ConversationTurn(
            task_id=task_id, conversation_id=identifier, question=question, status="running", recovery={}
        )
        return identifier, "", []

    async def owned(self, user_id: UUID, task_id: UUID) -> ConversationTurn:
        if user_id != self.user or task_id not in self.rows:
            raise NotFoundError()
        return self.rows[task_id]

    async def mark(
        self, task_id: UUID, status: str, receipt: dict[str, Any], state: dict[str, Any] | None = None
    ) -> ConversationTurn:
        row = self.rows[task_id]
        row.status, row.recovery = status, receipt
        if state:
            row.answer, row.termination = state.get("answer"), state.get("termination")
        return row

    @asynccontextmanager
    async def saver(self) -> AsyncIterator[AsyncSqliteSaver]:
        async with AsyncSqliteSaver.from_conn_string(str(self.root / "checkpoints.sqlite")) as saver:
            yield saver


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    harness = Harness(tmp_path)
    monkeypatch.setattr(module, "registry", RunRegistry())
    monkeypatch.setattr(module, "open_checkpointer", harness.saver)
    monkeypatch.setattr(module, "get_llm_provider", lambda: Mock(build_model=lambda **_: harness.model))
    monkeypatch.setattr(graph_module, "get_llm_provider", lambda: Mock(build_model=lambda **_: harness.model))
    monkeypatch.setattr(
        "app.agents.providers.OpenAIProvider.build_model", Mock(side_effect=AssertionError("Live model forbidden"))
    )
    monkeypatch.setattr(
        "app.agents.providers.OllamaProvider.build_model", Mock(side_effect=AssertionError("Live model forbidden"))
    )
    monkeypatch.setattr(module, "get_profile_memory", lambda: Mock(recall=AsyncMock(return_value=[])))
    monkeypatch.setattr(harness.service, "_begin_turn", harness.begin)
    monkeypatch.setattr(harness.service, "_owned_task", harness.owned)
    monkeypatch.setattr(harness.service, "_mark", harness.mark)
    return harness


async def paused_run(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], int]:
    started, release = asyncio.Event(), asyncio.Event()

    @tool("search_rag_tool")
    async def search(query: str) -> str:
        """Synthetic lookup with a controlled pause point."""
        started.set()
        await release.wait()
        return await demo_search.ainvoke({"query": query})

    monkeypatch.setattr(executor, "business_tools", [search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    job = asyncio.create_task(harness.service.run(harness.user, "查法规并整理"))
    try:
        await asyncio.wait_for(started.wait(), 3)
        task_id = next(iter(harness.rows))
        response, requested = await harness.service.pause(harness.user, task_id)
        assert requested and response["pause_requested"]
        assert not job.done()  # request acknowledgement is not completion
        with pytest.raises(AlreadyExistsError):
            await harness.service.resume(harness.user, task_id)
        with pytest.raises(AlreadyExistsError):
            await harness.service.abandon(harness.user, task_id)
        release.set()
        result = await asyncio.wait_for(job, 3)
        assert result["status"] == "interrupted"
        assert result["recovery"]["can_resume"] is True
        assert result["recovery"]["pending_nodes"] == ["draft"]
        assert result["budget"]["model_calls"] == 3
        assert result["evidence"][0]["id"] == SOURCE_ID
        return result, harness.model.calls
    finally:
        release.set()
        if not job.done():
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)


@pytest.mark.anyio
async def test_pause_finishes_inner_loop_and_resume_skips_saved_work(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, before = await paused_run(harness, monkeypatch)
    task_id = UUID(result["task_id"])
    concurrent = await asyncio.gather(
        harness.service.resume(harness.user, task_id),
        harness.service.resume(harness.user, task_id),
        return_exceptions=True,
    )
    assert sum(isinstance(item, AlreadyExistsError) for item in concurrent) == 1
    finished = next(item for item in concurrent if isinstance(item, dict))
    assert finished["status"] == "completed" and finished["answer"] == ANSWER
    assert harness.model.calls == before + 2  # draft/review only; no repeated tool
    assert finished["budget"]["model_calls"] == 5
    await harness.service.resume(harness.user, task_id)
    assert harness.model.calls == 5
    with pytest.raises(NotFoundError):
        await harness.service.pause(uuid4(), task_id)


@pytest.mark.anyio
async def test_uncertain_crash_refuses_resume_but_can_be_abandoned(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, before = await paused_run(harness, monkeypatch)
    task_id = UUID(result["task_id"])
    harness.rows[task_id].status = "running"
    harness.rows[task_id].recovery = {"kind": "in_flight"}
    view = await harness.service.get_task(harness.user, task_id)
    assert view["status"] == "interrupted"
    assert view["recovery"]["budget_status"] == "uncertain"
    with pytest.raises(AlreadyExistsError):
        await harness.service.resume(harness.user, task_id)
    assert harness.model.calls == before
    assert (await harness.service.abandon(harness.user, task_id))["status"] == "abandoned"
    assert (await harness.service.abandon(harness.user, task_id))["status"] == "abandoned"
    with pytest.raises(AlreadyExistsError):
        await harness.service.resume(harness.user, task_id)


@pytest.mark.anyio
async def test_known_task_without_checkpoint_is_queryable(harness: Harness) -> None:
    task_id, conversation_id = uuid4(), uuid4()
    await harness.begin(harness.user, "尚未生成检查点", task_id, conversation_id)
    result = await harness.service.get_task(harness.user, task_id)
    assert result["status"] == "failed"
    assert result["recovery"]["blocked_reason"] == "checkpoint_missing"
    assert harness.rows[task_id].status == "running"  # GET doesn't mutate metadata


@pytest.mark.anyio
@pytest.mark.parametrize("unsafe", [False, True])
async def test_memory_write_pause_or_interruption_does_not_replay(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, unsafe: bool
) -> None:
    from typing import cast

    from app.agents.memory.profile import ProfileMemory
    from tests.agents.test_personal_memory_loop import COMMAND, ProfileSDK, plan_response
    from tests.agents.test_work_context import RecordingModel

    sdk = ProfileSDK()
    memory = ProfileMemory(cast(Any, sdk))
    harness.model = RecordingModel(responses=[plan_response()])
    monkeypatch.setattr(module, "get_profile_memory", lambda: memory)
    original = graph_module.remember_profile

    async def pause_after(state: Any, factory: Any) -> dict[str, Any]:
        result = await original(state, factory)
        if unsafe:
            raise RuntimeError("simulated loss before checkpoint")
        run = module.registry.find(UUID(state["conversation_id"]))
        assert run is not None
        run.control.request_drain("test_memory_boundary")
        return result

    monkeypatch.setattr(graph_module, "remember_profile", pause_after)
    if unsafe:
        with pytest.raises(RuntimeError):
            await harness.service.run(harness.user, COMMAND)
        task_id = next(iter(harness.rows))
        with pytest.raises(AlreadyExistsError):
            await harness.service.resume(harness.user, task_id)
        await harness.service.abandon(harness.user, task_id)
    else:
        paused = await harness.service.run(harness.user, COMMAND)
        assert paused["recovery"]["can_resume"] is True
        resumed = await harness.service.resume(harness.user, UUID(paused["task_id"]))
        assert resumed["memory_result"]["status"] == "saved"
        assert resumed["status"] == "completed"
    assert sdk.writes == 1 and sdk.rows
    assert harness.model.calls == 1


@pytest.mark.anyio
async def test_resume_invalidates_receipt_before_graph_call(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    result, before = await paused_run(harness, monkeypatch)
    task_id = UUID(result["task_id"])

    async def unavailable(*args: Any, **kwargs: Any) -> ConversationTurn:
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(harness.service, "_mark", unavailable)
    with pytest.raises(RuntimeError):
        await harness.service.resume(harness.user, task_id)
    assert harness.model.calls == before


@pytest.mark.anyio
@pytest.mark.parametrize("stage,remaining_calls", [("draft", 1), ("review", 0)])
async def test_pause_after_draft_or_review_preserves_saved_work(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    remaining_calls: int,
) -> None:
    monkeypatch.setattr(executor, "business_tools", [demo_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    name = "draft_answer" if stage == "draft" else "review_task"
    original = getattr(graph_module, name)

    async def pause_after(state: dict[str, Any], model: Any) -> dict[str, Any]:
        result = await original(state, model)
        run = module.registry.find(UUID(state["conversation_id"]))
        assert run is not None
        run.control.request_drain("test_boundary")
        return result

    monkeypatch.setattr(graph_module, name, pause_after)
    paused = await harness.service.run(harness.user, "核查法规")
    assert paused["status"] == "interrupted"
    before = harness.model.calls
    monkeypatch.setattr(graph_module, name, original)
    finished = await harness.service.resume(harness.user, UUID(paused["task_id"]))
    assert finished["status"] == "completed"
    assert harness.model.calls == before + remaining_calls


@pytest.mark.anyio
async def test_completed_checkpoint_survives_metadata_failure(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "business_tools", [demo_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    original = harness.mark

    async def fail_finish(
        task_id: UUID, status: str, receipt: dict[str, Any], state: dict[str, Any] | None = None
    ) -> ConversationTurn:
        if status == "completed":
            raise RuntimeError("metadata failure")
        return await original(task_id, status, receipt, state)

    monkeypatch.setattr(harness.service, "_mark", fail_finish)
    with pytest.raises(RuntimeError):
        await harness.service.run(harness.user, "核查法规")
    task_id = next(iter(harness.rows))
    assert (await harness.service.get_task(harness.user, task_id))["status"] == "completed"
    monkeypatch.setattr(harness.service, "_mark", original)
    await harness.service.resume(harness.user, task_id)
    assert harness.model.calls == 5


@pytest.mark.anyio
async def test_store_failure_is_not_checkpoint_missing(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    task_id = uuid4()
    await harness.begin(harness.user, "已登记的任务", task_id, uuid4())

    @asynccontextmanager
    async def unavailable() -> AsyncIterator[Any]:
        raise ExternalServiceError(message="store unavailable")
        yield  # pragma: no cover -- defines async context-manager protocol

    monkeypatch.setattr(module, "open_checkpointer", unavailable)
    with pytest.raises(ExternalServiceError):
        await harness.service.get_task(harness.user, task_id)
    assert harness.rows[task_id].status == "running"
