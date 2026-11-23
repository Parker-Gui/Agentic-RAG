"""Bounded task projections reach real tool loops without granting evidence coverage."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import Field

from app.agents import executor
from app.agents.context.selection import select_evidence
from app.agents.context.work import MAX_PROGRESS_ITEMS, MAX_SOURCES, work_prompt, work_tokens
from app.agents.graph import build_graph
from app.agents.models import Budget, Evidence, Subtask
from app.agents.nodes.execute import execute_task
from app.agents.nodes.review import review_task
from app.agents.state import TaskState
from app.core.config import settings
from tests.agents.test_research_workflow import ScriptedModel


class RecordingModel(ScriptedModel):
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(deepcopy(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def source(index: int = 0, **changes: Any) -> Evidence:
    return Evidence(
        **{
            "id": f"E-{index:012x}",
            "subtask_id": f"S-{index + 1}",
            "tool": "search_rag_tool",
            "title": f"客户{index}扩产",
            "content": "扩产项目披露了产能，投资金额尚未确认。",
            "url": f"https://example.invalid/customer/{index}",
            **changes,
        }
    )


def working_state() -> TaskState:
    return {
        "question": "比较客户扩产项目",
        "user_id": "user",
        "task_id": "current-task",
        "current_index": 1,
        "plan": [
            Subtask(id="S-1", goal="查客户扩产", status="covered", evidence_ids=[source().id], attempts=1).model_dump(),
            Subtask(id="S-2", goal="查另一客户扩产").model_dump(),
        ],
        "evidence": [source().model_dump()],
        "budget": Budget(max_review_retries=0).model_dump(),
        "profile": ["PRIVATE_PROFILE"],
        "answer": "PRIVATE_ANSWER",
        "trace": [{"detail": "PRIVATE_TRACE"}],
    }


def payload(text: str) -> dict[str, Any]:
    return json.loads(text[text.index("{") :])


def test_projection_preserves_provenance_and_excludes_non_sources() -> None:
    state = working_state()
    state["evidence"] += [
        source(1, source_task_id="previous-task").model_dump(),
        source(2, title="星系", content="宇宙星云").model_dump(),
        source(3, url=None).model_dump(),
        source(4, url=None, article_id=42).model_dump(),
    ]
    before = deepcopy(state)
    rendered = work_prompt(state, "查另一客户扩产")
    result = payload(rendered)
    by_id = {item["id"]: item for item in result["candidate_sources"]}
    assert set(by_id) == {source().id, source(1).id, source(4).id}
    assert by_id[source(1).id]["source_task_id"] == "previous-task"
    assert "未重新查证" in by_id[source(1).id]["origin"]
    assert by_id[source().id]["url"] == source().url
    assert "未完成最终审查" in result["progress"][1]["status"]
    assert all(secret not in rendered for secret in ("PRIVATE_PROFILE", "PRIVATE_ANSWER", "PRIVATE_TRACE"))
    assert state == before
    assert work_prompt(json.loads(json.dumps(state)), "查另一客户扩产") == rendered


@pytest.mark.parametrize("cap", [0, 1, 80, 200, 600, 1800])
def test_projection_budget_keeps_complete_records_and_original_state(cap: int) -> None:
    state = working_state()
    state["plan"] = [Subtask(id=f"S-{i}", goal="扩产" * 1000, attempts=1).model_dump() for i in range(15)]
    state["current_index"] = 10
    state["evidence"] = [source(i, content="扩产" * 1000, title="扩产" * 1000).model_dump() for i in range(15)]
    before = deepcopy(state)
    text = work_prompt(state, "扩产", max_tokens=cap)
    if cap == 1800:
        assert text and payload(text)["candidate_sources"]
    if text:
        assert work_tokens(text) <= cap
        result = payload(text)
        assert len(result["progress"]) <= MAX_PROGRESS_ITEMS
        assert len(result["candidate_sources"]) <= MAX_SOURCES
        assert result["omitted_progress"] + len(result["progress"]) == 15
        assert result["omitted_sources"] + len(result["candidate_sources"]) == 15
        originals = {item["id"]: item for item in state["evidence"]}
        for item in result["candidate_sources"]:
            assert item["url"] == originals[item["id"]]["url"]
            assert item["excerpt_truncated"] is True
        if result["progress"]:
            assert result["progress"][0]["current"] is True
    assert state == before


def test_oversized_source_is_omitted_without_corrupting_location_or_losing_next_source() -> None:
    state = working_state()
    state["evidence"] = [source(99, url="https://example.invalid/" + "x" * 10000).model_dump(), source().model_dump()]
    result = payload(work_prompt(state, "扩产"))
    assert [item["id"] for item in result["candidate_sources"]] == [source().id]
    assert result["omitted_sources"] == 1


def test_fresh_task_and_unrelated_sources_add_no_context() -> None:
    assert work_prompt({"plan": [Subtask(id="S-1", goal="扩产").model_dump()]}, "扩产") == ""
    assert work_prompt({"evidence": [source(title="星系", content="宇宙星云").model_dump()]}, "扩产") == ""
    # Existing draft selection keeps its prior fallback; only working context filters zero overlap.
    records = [source(title="星系", content="宇宙星云")]
    assert select_evidence("扩产", records) == records
    assert select_evidence("扩产", records, minimum_overlap=1) == []


def test_item_limits_keep_current_and_nearest_steps_even_with_large_token_budget() -> None:
    state = working_state()
    state["plan"] = [Subtask(id=f"S-{i}", goal="扩产", attempts=1).model_dump() for i in range(15)]
    state["current_index"] = 10
    state["evidence"] = [source(i).model_dump() for i in range(15)]
    result = payload(work_prompt(state, "扩产", max_tokens=10000))
    assert [item["id"] for item in result["progress"]] == ["S-10", "S-9", "S-8", "S-7", "S-6", "S-5"]
    assert len(result["candidate_sources"]) == 6
    assert result["omitted_sources"] == 9


@tool("search_rag_tool")
async def customer_search(query: str) -> str:
    """Return a fixed customer source without any external request."""
    return json.dumps(
        {
            "results": [
                {"title": query, "url": f"https://example.invalid/{query}", "content": query + "已披露扩产计划。"}
            ]
        },
        ensure_ascii=False,
    )


def search_call(query: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": query}, "id": query}])


def work_input(messages: list[BaseMessage]) -> str:
    return next(
        str(item.content) for item in messages if isinstance(item, HumanMessage) and "研究问题：" in str(item.content)
    )


@pytest.mark.anyio
async def test_three_subtasks_receive_prior_sources_and_rebuild_after_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "business_tools", [customer_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    path = str(tmp_path / "work-context.sqlite")
    config = {"configurable": {"thread_id": str(uuid4())}}
    initial: TaskState = {"question": "比较客户甲乙扩产", "user_id": str(uuid4()), "task_id": str(uuid4())}
    first = RecordingModel(
        responses=[
            AIMessage(content='{"goals":["客户甲扩产","客户乙扩产","比较客户甲乙扩产"]}'),
            search_call("客户甲扩产"),
            AIMessage(content="取得甲的候选资料"),
        ]
    )
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_graph(first, checkpointer=saver)
        await graph.ainvoke(initial, config=config, interrupt_after=["execute"])
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("execute",)
        before = work_prompt(snapshot.values, snapshot.values["plan"][1]["goal"])
        first_id = snapshot.values["evidence"][0]["id"]
        assert "任务工作参考" not in work_input(first.seen[1])

    second = RecordingModel(responses=[search_call("客户乙扩产"), AIMessage(content="取得乙的候选资料")])
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_graph(second, checkpointer=saver)
        restored = await graph.aget_state(config)
        assert work_prompt(restored.values, restored.values["plan"][1]["goal"]) == before
        await graph.ainvoke(None, config=config, interrupt_after=["execute"])
        snapshot = await graph.aget_state(config)
        sources = snapshot.values["evidence"]
        ids = [item["id"] for item in sources]
        assert before in work_input(second.seen[0])
        assert first_id in work_input(second.seen[0])

    answer = "客户已披露扩产计划。" + "".join(f"[{item}]" for item in ids)
    third = RecordingModel(
        responses=[
            search_call("客户甲扩产"),
            AIMessage(content="比较完成"),
            AIMessage(content=answer),
            AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":"fixture"}'),
        ]
    )
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        result = await build_graph(third, checkpointer=saver).ainvoke(None, config=config)
    assert all(item in work_input(third.seen[0]) for item in ids)
    assert result["termination"] == "completed"
    assert {item["id"] for item in result["evidence"]} == set(ids)
    assert result["budget"]["model_calls"] == first.calls + second.calls + third.calls == 9
    assert result["budget"]["tool_calls"] == 3
    assert result["budget"]["summary_calls"] == 0


@pytest.mark.anyio
async def test_prior_sources_do_not_automatically_cover_new_subtask() -> None:
    state = working_state()
    model = RecordingModel(responses=[AIMessage(content="看到前序资料，但没有查到当前目标的来源")])
    result = await execute_task(state, model)
    assert source().id in work_input(model.seen[0])
    assert result["plan"][1]["status"] == "missing"
    assert result["plan"][1]["evidence_ids"] == []
    state["plan"] = result["plan"]
    state["budget"] = result["budget"]
    state["draft"] = f"客户扩产。[{source().id}]"
    reviewed = await review_task(state, ScriptedModel(responses=[]))
    assert reviewed["termination"] == "insufficient_evidence"
    assert reviewed["review"]["approved"] is False
    assert reviewed["review"]["missing_subtasks"] == ["S-2"]


@pytest.mark.anyio
async def test_tight_input_budget_drops_projection_and_preserves_current_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.agents.nodes import execute

    monkeypatch.setattr(settings, "AGENT_INPUT_TOKENS", 16)
    captured: list[str] = []

    async def capture(model: Any, prompt: str, **kwargs: Any) -> executor.ExecutionResult:
        captured.append(prompt)
        return executor.ExecutionResult(
            messages=[], tool_messages=[], model_calls=0, tool_calls=0, offloaded_tool_results=0, stop_reason=None
        )

    monkeypatch.setattr(execute, "run_executor", capture)
    state = working_state()
    await execute_task(state, ScriptedModel(responses=[]))
    assert state["question"] in captured[0]
    assert state["plan"][1]["goal"] in captured[0]
    assert "任务工作参考" not in captured[0]
