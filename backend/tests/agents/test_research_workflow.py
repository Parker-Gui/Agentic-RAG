"""Exercise a real Deep Agents tool turn and the outer evidence/review path offline."""

import hashlib
import json
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents import executor
from app.agents.evidence import from_tool_messages
from app.agents.graph import build_graph
from app.agents.models import Budget
from app.agents.nodes.review import review_task


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


@tool("search_rag_tool")
async def fake_search(query: str) -> str:
    """Return one source-backed research record."""
    return json.dumps(
        {"results": [{"title": "法规公告", "url": "https://example.org/rule", "content": "法规已经生效。"}]},
        ensure_ascii=False,
    )


@tool("search_rag_tool")
async def long_search(query: str) -> str:
    """Return a source record larger than the tool offload threshold."""
    return json.dumps({"results": [{"title": "长文", "url": "https://example.org/long", "content": "fact " * 12000}]})


def _id() -> str:
    return "E-" + hashlib.sha256("https://example.org/rule\n法规已经生效。".encode()).hexdigest()[:12]


@pytest.mark.anyio
async def test_deep_agent_runs_business_tool_without_subagent_or_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    model = ScriptedModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": "call-1"}]),
            AIMessage(content="已查到资料"),
        ]
    )
    limits = executor.ExecutionLimits(3, 3, float("inf"))
    agent = executor.build_executor(model, limits)
    assert set(agent.nodes["tools"].bound.tools_by_name) == {"search_rag_tool", *executor.FILE_TOOLS}

    result = await executor.run_executor(model, "查法规", max_model_calls=3, max_tool_calls=3, max_seconds=30)
    extracted, errors = from_tool_messages(result.messages, "S-1")
    assert result.tool_calls == 1
    assert [item.id for item in extracted] == [_id()]
    assert errors == []


@pytest.mark.anyio
async def test_outer_graph_produces_reviewed_cited_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    model = ScriptedModel(
        responses=[
            AIMessage(content='{"goals":["核查法规"]}'),
            AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": "call-2"}]),
            AIMessage(content="检索完成"),
            AIMessage(content=f"法规已经生效。[{_id()}]"),
            AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":""}'),
        ]
    )
    graph = build_graph(model)
    result = await graph.ainvoke(
        {
            "question": "法规是否生效？",
            "user_id": str(uuid4()),
            "task_id": str(uuid4()),
            "budget": Budget().model_dump(),
        },
        config={"recursion_limit": 25},
    )
    assert result["termination"] == "completed"
    assert result["answer"] == f"法规已经生效。[{_id()}]"
    assert result["claims"][0]["evidence_ids"] == [_id()]


@pytest.mark.anyio
async def test_review_rejection_adds_verification_subtask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    def call(call_id: str) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": call_id}])
    model = ScriptedModel(
        responses=[
            AIMessage(content='{"goals":["核查法规"]}'),
            call("first"),
            AIMessage(content="完成"),
            AIMessage(content=f"未经支持的结论。[{_id()}]"),
            AIMessage(content='{"approved":false,"unsupported_claims":["未经支持的结论"],"reason":"来源不支持"}'),
            call("second"),
            AIMessage(content="补检完成"),
            AIMessage(content=f"法规已经生效。[{_id()}]"),
            AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":""}'),
        ]
    )
    result = await build_graph(model).ainvoke(
        {"question": "法规是否生效？", "user_id": str(uuid4()), "task_id": str(uuid4())},
        config={"recursion_limit": 30},
    )
    assert result["termination"] == "completed"
    assert len(result["plan"]) == 2
    assert result["budget"]["review_retries"] == 1
    assert result["budget"]["tool_calls"] == 2


@pytest.mark.anyio
async def test_no_evidence_stops_after_review_budget() -> None:
    model = ScriptedModel(
        responses=[
            AIMessage(content='{"goals":["查证事实"]}'),
            AIMessage(content="没有查到资料"),
            AIMessage(content="依然没有资料"),
            AIMessage(content="仍无资料"),
        ]
    )
    result = await build_graph(model).ainvoke(
        {"question": "一个没有资料的问题", "user_id": str(uuid4()), "task_id": str(uuid4())},
        config={"recursion_limit": 25},
    )
    assert result["termination"] == "insufficient_evidence"
    assert result["budget"]["review_retries"] == 2
    assert not result.get("evidence")


def test_tool_error_never_becomes_evidence() -> None:
    messages = [ToolMessage(content='{"results":[],"error":"失效"}', name="search_rag_tool", tool_call_id="x")]
    evidence, errors = from_tool_messages(messages, "S-1")
    assert evidence == []
    assert errors == ["search_rag_tool: 失效"]


@pytest.mark.anyio
async def test_review_rejects_unknown_citation_and_missing_subtask() -> None:
    result = await review_task(
        {
            "draft": "法规已更新。[E-000000000000]",
            "plan": [{"id": "S-1", "goal": "法规"}],
            "evidence": [],
            "budget": Budget(max_review_retries=0).model_dump(),
        },
        ScriptedModel(responses=[]),
    )
    assert result["review"]["approved"] is False
    assert result["termination"] == "insufficient_evidence"


@pytest.mark.anyio
async def test_model_budget_stops_tool_loop_without_recursion_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    model = ScriptedModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": "x"}])
        ]
    )
    result = await executor.run_executor(model, "查法规", max_model_calls=1, max_tool_calls=3, max_seconds=30)
    assert result.stop_reason == "model_budget"
    assert result.tool_calls == 1


@pytest.mark.anyio
async def test_invalid_real_tool_arguments_remain_structured() -> None:
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规", "top_k": 0}, "id": "bad"}],
            ),
            AIMessage(content="参数错误，无法取得证据"),
        ]
    )
    result = await executor.run_executor(model, "查法规", max_model_calls=3, max_tool_calls=3, max_seconds=30)
    messages = [item for item in result.messages if isinstance(item, ToolMessage) and item.name == "search_rag_tool"]
    assert len(messages) == 1
    assert str(messages[0].content).startswith("{"), str(messages[0].content)
    assert json.loads(str(messages[0].content))["results"] == []
    assert "invalid_params" in json.loads(str(messages[0].content))


@pytest.mark.anyio
async def test_offloaded_tool_result_keeps_evidence_outside_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "business_tools", [long_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    model = ScriptedModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "long"}, "id": "long"}]),
            AIMessage(content="已读长文"),
        ]
    )
    result = await executor.run_executor(model, "读长文", max_model_calls=3, max_tool_calls=3, max_seconds=30)
    evidence, errors = from_tool_messages(result.tool_messages, "S-1")
    assert len(evidence) == 1 and evidence[0].content.startswith("fact ")
    assert errors == []
    assert result.offloaded_tool_results == 1


@pytest.mark.anyio
async def test_checkpoint_survives_new_saver_instance(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    model = ScriptedModel(
        responses=[
            AIMessage(content='{"goals":["核查法规"]}'),
            AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": "x"}]),
            AIMessage(content="检索完成"),
            AIMessage(content=f"法规已经生效。[{_id()}]"),
            AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":""}'),
        ]
    )
    path = str(tmp_path / "checkpoint.sqlite")
    config = {"configurable": {"thread_id": str(uuid4())}}
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await build_graph(model, checkpointer=saver).ainvoke(
            {"question": "法规是否生效？", "user_id": str(uuid4()), "task_id": str(uuid4())}, config=config
        )
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        snapshot = await build_graph(model, checkpointer=saver).aget_state(config)
    assert snapshot.values["termination"] == "completed"
    assert snapshot.values["evidence"][0]["id"] == _id()


@pytest.mark.anyio
async def test_interrupted_task_resumes_from_persisted_plan(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    path = str(tmp_path / "resume.sqlite")
    config = {"configurable": {"thread_id": str(uuid4())}}
    initial = {"question": "法规是否生效？", "user_id": str(uuid4()), "task_id": str(uuid4())}
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await build_graph(
            ScriptedModel(responses=[AIMessage(content='{"goals":["核查法规"]}')]), checkpointer=saver
        ).ainvoke(initial, config=config, interrupt_after=["plan"])
    remaining = ScriptedModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": "x"}]),
            AIMessage(content="检索完成"),
            AIMessage(content=f"法规已经生效。[{_id()}]"),
            AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":""}'),
        ]
    )
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_graph(remaining, checkpointer=saver)
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("execute",)
        result = await graph.ainvoke(None, config=config)
    assert result["termination"] == "completed"
