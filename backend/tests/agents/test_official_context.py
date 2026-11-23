"""Exercise the installed official compressor, not a reimplementation of compaction."""

from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents import executor
from app.agents.models import Evidence
from app.agents.nodes.review import review_task
from app.core.config import settings
from tests.agents.test_research_workflow import ScriptedModel, fake_search


@pytest.mark.anyio
async def test_official_summary_replaces_history_and_accounts_extra_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "AGENT_SUMMARY_TRIGGER_TOKENS", 1500)
    monkeypatch.setattr(settings, "AGENT_KEEP_TOKENS", 100)
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {fake_search.name})
    model = ScriptedModel(
        responses=[AIMessage(content="目标：查法规。已确认来源 https://example.org。"), AIMessage(content="继续执行")]
    )
    limits = executor.ExecutionLimits(2, 3, float("inf"))
    agent = executor.build_executor(model, limits, backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True))
    messages = [
        HumanMessage(content="保留法规查询目标。" * 1500),
        AIMessage(content="", tool_calls=[{"id": "old-call", "name": fake_search.name, "args": {"query": "法规"}}]),
        ToolMessage(content="法规结果", tool_call_id="old-call", name=fake_search.name),
        HumanMessage(content="继续查证"),
    ]
    result = await agent.ainvoke({"messages": messages}, output_keys=["messages", "_summarization_event"])
    event = result["_summarization_event"]
    assert limits.summary_calls == 1
    assert limits.model_calls == 2
    assert limits.usage[0]["purpose"] == "summary"
    assert event["cutoff_index"] > 0
    assert event["file_path"]
    assert list(tmp_path.rglob("*.md"))
    assert any(
        isinstance(message, ToolMessage) and message.tool_call_id == "old-call" for message in result["messages"]
    )


@pytest.mark.anyio
async def test_summary_exhaustion_cannot_bypass_shared_model_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AGENT_SUMMARY_TRIGGER_TOKENS", 1000)
    monkeypatch.setattr(settings, "AGENT_KEEP_TOKENS", 100)
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {fake_search.name})
    model = ScriptedModel(responses=[AIMessage(content="摘要")])
    limits = executor.ExecutionLimits(1, 1, float("inf"))
    agent = executor.build_executor(model, limits)
    with pytest.raises(executor.ExecutionStopped):
        await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content="旧记录" * 2000),
                    AIMessage(content="已读"),
                    HumanMessage(content="继续"),
                ]
            }
        )
    assert model.calls == limits.model_calls == 1
    assert limits.summary_calls == 1
    assert limits.stop_reason == "model_budget"


@pytest.mark.anyio
async def test_unreviewed_draft_tail_cannot_be_approved() -> None:
    source = Evidence(
        id="E-123456789abc", subtask_id="S-1", tool="search", title="来源", content="事实", url="https://example.org"
    )
    model = ScriptedModel(responses=[])
    result = await review_task(
        {
            "evidence": [source.model_dump()],
            "plan": [],
            "draft": ("事实[E-123456789abc]\n" * 500),
            "budget": {"max_review_retries": 0},
        },
        model,
    )
    assert result["review"]["approved"] is False
    assert "长度" in result["review"]["reason"]
    assert model.calls == 0


@pytest.mark.anyio
async def test_empty_summary_fails_instead_of_replacing_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AGENT_SUMMARY_TRIGGER_TOKENS", 1000)
    monkeypatch.setattr(settings, "AGENT_KEEP_TOKENS", 100)
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {fake_search.name})
    model = ScriptedModel(responses=[AIMessage(content=" ")])
    limits = executor.ExecutionLimits(3, 1, float("inf"))
    agent = executor.build_executor(model, limits)
    with pytest.raises(ValueError, match="摘要为空"):
        await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content="旧记录" * 2000),
                    AIMessage(content="已读"),
                    HumanMessage(content="继续"),
                ]
            }
        )
    assert model.calls == limits.model_calls == limits.summary_calls == 1


@pytest.mark.anyio
async def test_irreducible_input_stops_without_sending_model_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AGENT_INPUT_TOKENS", 50)
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {fake_search.name})
    model = ScriptedModel(responses=[])
    result = await executor.run_executor(model, "长输入" * 1000, max_model_calls=2, max_tool_calls=1, max_seconds=10)
    assert result.stop_reason == "execution_error"
    assert result.model_calls == model.calls == 0
    assert result.context_events[0]["error"] == "ContextOverflowError"
