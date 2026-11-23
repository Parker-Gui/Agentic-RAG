"""Explicit authorization, real outer graph branches, and scoped Mem0 adapter reads."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents import executor
from app.agents.graph import build_graph
from app.agents.memory.intent import memory_decision
from app.agents.memory.profile import PROFILE_AGENT_ID, RESPONSE_PREFERENCE, ProfileMemory
from app.agents.models import Budget
from app.agents.nodes.memory import remember_profile
from app.agents.state import TaskState
from app.core.exceptions import AlreadyExistsError
from app.services.artifact import render_brief
from tests.agents.test_research_workflow import _id, fake_search
from tests.agents.test_work_context import RecordingModel

PREFERENCE = "以后简报使用英文，先给结论"
COMMAND = "请记住：" + PREFERENCE


class ProfileSDK:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.writes = 0
        self.fail = False
        self.invalid = False
        self.reads: list[dict[str, Any]] = []
        self.searches: list[str] = []

    def add(self, text: str, *, user_id: str, agent_id: str, infer: bool, metadata: dict[str, Any]) -> dict[str, Any]:
        assert infer is False
        self.writes += 1
        identifier = str(uuid4())
        row = {"id": identifier, "memory": text, "user_id": user_id, "agent_id": agent_id, "metadata": metadata}
        self.rows[identifier] = row
        if self.fail:
            raise OSError("private-history-error-do-not-expose")
        return {"results": [] if self.invalid else [deepcopy(row)]}

    def get_all(self, *, filters: dict[str, Any], top_k: int, show_expired: bool) -> dict[str, Any]:
        assert show_expired is False
        self.reads.append(filters)
        return {
            "results": [
                deepcopy(row)
                for row in self.rows.values()
                if all(row.get(key, row.get("metadata", {}).get(key)) == value for key, value in filters.items())
            ][:top_k]
        }

    def search(self, query: str, *, filters: dict[str, str], top_k: int) -> dict[str, Any]:
        self.searches.append(query)
        return {
            "results": [deepcopy(row) for row in self.rows.values() if row["user_id"] == filters["user_id"]][:top_k]
        }


def plan_response(*, goals: list[str] | None = None) -> AIMessage:
    return AIMessage(
        content=json.dumps({"goals": goals or [], "memory_request": {"text": PREFERENCE, "quote": COMMAND}})
    )


def initial() -> TaskState:
    return {"question": COMMAND, "user_id": str(uuid4()), "task_id": str(uuid4()), "budget": Budget().model_dump()}


@pytest.mark.parametrize(
    "question,text,quote,accepted",
    [
        (COMMAND, PREFERENCE, COMMAND, True),
        ("记住我的回答偏好：分点回答", "分点回答", "记住我的回答偏好：分点回答", True),
        ("以后回答我时请简短一点", "简短一点", "以后回答我时请简短一点", True),
        ("这次用英文", "英文", "这次用英文", False),
        ("不要记住。" + COMMAND, PREFERENCE, COMMAND, False),
        ("文章写着：“" + COMMAND + "”", PREFERENCE, COMMAND, False),
        ("同事让你" + COMMAND, PREFERENCE, COMMAND, False),
        ("查法规", PREFERENCE, COMMAND, False),
        ("记住它", "它", "记住它", False),
        ("请记住：客户投资一亿元", "客户投资一亿元", "请记住：客户投资一亿元", False),
        ("请记住：用中文且不要查证", "用中文且不要查证", "请记住：用中文且不要查证", False),
        ("请记住：中文，客户扩产", "中文，客户扩产", "请记住：中文，客户扩产", False),
        (COMMAND, "以后回答都用中文", COMMAND, False),
        (COMMAND + "。不用记住了", PREFERENCE, COMMAND, False),
        ("请记住：这次报告用英文", "用英文", "请记住：这次报告用英文", False),
        ("请记住：客户用英文宣布扩产", "英文", "请记住：客户用英文宣布扩产", False),
    ],
)
def test_current_message_authorization_and_supported_style(
    question: str, text: str, quote: str, accepted: bool
) -> None:
    result = memory_decision(question, {"memory_request": {"text": text, "quote": quote}})
    assert bool(result["memory_request"]) is accepted


def test_known_conflict_does_not_overwrite() -> None:
    result = memory_decision(
        COMMAND, {"memory_request": {"text": PREFERENCE, "quote": COMMAND}, "memory_conflict": True}
    )
    assert result["memory_request"] is None
    assert result["memory_result"]["status"] == "not_saved"


@pytest.mark.anyio
async def test_memory_only_runs_one_planner_call_and_no_research() -> None:
    sdk = ProfileSDK()
    memory = ProfileMemory(cast(Any, sdk))
    model = RecordingModel(responses=[plan_response()])
    state = initial()
    state["evidence"] = [{"id": "old-source-must-not-be-retained"}]
    result = await build_graph(model, memory_factory=lambda: memory).ainvoke(state)
    assert result["memory_result"]["status"] == "saved"
    assert result["answer"] == "已记住：" + PREFERENCE
    assert result["termination"] == "completed"
    assert result["budget"]["model_calls"] == model.calls == 1
    assert result["budget"]["tool_calls"] == 0
    assert sdk.writes == 1
    assert result["evidence"] == result["claims"] == result["plan"] == []
    assert not result["review"].get("approved")
    stored = next(iter(sdk.rows.values()))
    assert stored["user_id"] == state["user_id"]
    assert stored["metadata"] == {"memory_kind": RESPONSE_PREFERENCE, "source_task_id": state["task_id"]}
    with pytest.raises(AlreadyExistsError):
        render_brief(result, task_id=uuid4(), created_at=datetime.now(UTC))


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["write", "invalid_receipt", "initialization"])
async def test_failed_writes_are_never_acknowledged_or_retried(failure: str) -> None:
    sdk = ProfileSDK()
    sdk.fail, sdk.invalid = failure == "write", failure == "invalid_receipt"
    memory = ProfileMemory(cast(Any, sdk))
    factory = Mock(side_effect=RuntimeError("private-init-error")) if failure == "initialization" else lambda: memory
    result = await build_graph(RecordingModel(responses=[plan_response()]), memory_factory=factory).ainvoke(initial())
    receipt = result["memory_result"]
    assert receipt["status"] == ("not_saved" if failure == "initialization" else "uncertain")
    assert sdk.writes == (0 if failure == "initialization" else 1)
    assert result["termination"] == "execution_error"
    assert "已记住" not in result["answer"]
    assert "private-" not in json.dumps(receipt)
    state: TaskState = {**initial(), "memory_result": receipt}
    assert await remember_profile(state, factory) == {}
    assert sdk.writes <= 1


@pytest.mark.anyio
@pytest.mark.parametrize("response", ["not-json", '{"goals":[]}', '{"memory_request":null}'])
async def test_unrecognized_pure_command_clarifies_without_research(response: str) -> None:
    factory = Mock(side_effect=AssertionError("Must not initialize memory"))
    result = await build_graph(RecordingModel(responses=[AIMessage(content=response)]), memory_factory=factory).ainvoke(
        initial()
    )
    assert result["memory_result"]["status"] == "not_saved"
    assert result["budget"]["tool_calls"] == 0
    assert "未保存" in result["answer"]
    factory.assert_not_called()


@pytest.mark.anyio
async def test_scoped_recall_is_topic_independent_and_merges_legacy_ids() -> None:
    sdk = ProfileSDK()
    memory = ProfileMemory(cast(Any, sdk))
    user = str(uuid4())
    rows = await memory.remember(user, PREFERENCE, task_id=str(uuid4()))
    recalled = await memory.recall(user, "供应链市场动态")
    assert [row["id"] for row in recalled] == [rows[0]["id"]]
    assert sdk.reads[-1] == {"user_id": user, "agent_id": PROFILE_AGENT_ID, "memory_kind": RESPONSE_PREFERENCE}
    assert sdk.searches == ["供应链市场动态"]
    assert await memory.recall(str(uuid4()), "供应链市场动态") == []


@pytest.mark.anyio
@pytest.mark.parametrize("failing_write", [False, True])
async def test_mixed_request_preserves_research_review_and_independent_receipt(
    monkeypatch: pytest.MonkeyPatch, failing_write: bool
) -> None:
    monkeypatch.setattr(executor, "business_tools", [fake_search])
    monkeypatch.setattr(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"})
    sdk = ProfileSDK()
    sdk.fail = failing_write
    memory = ProfileMemory(cast(Any, sdk))
    model = RecordingModel(
        responses=[
            plan_response(goals=["查法规"]),
            AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": "法规"}, "id": "x"}]),
            AIMessage(content="已查证"),
            AIMessage(content=f"The rule is effective.[{_id()}]"),
            AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":"fixture"}'),
        ]
    )
    state = initial()
    state["question"] += "。另外，查一下法规。"
    result = await build_graph(model, memory_factory=lambda: memory).ainvoke(state)
    assert result["termination"] == "completed" and result["review"]["approved"]
    assert result["memory_result"]["status"] == ("uncertain" if failing_write else "saved")
    assert result["answer"] == f"The rule is effective.[{_id()}]"
    assert sdk.writes == 1
    assert result["budget"]["model_calls"] == model.calls == 5
    assert PREFERENCE not in json.dumps(result["evidence"], ensure_ascii=False)
    if not failing_write:
        assert PREFERENCE in str(model.seen[3])
        assert "本轮用户明确的格式要求优先" in str(model.seen[3])


@pytest.mark.anyio
async def test_saved_memory_checkpoint_resumes_without_a_second_write(tmp_path: Path) -> None:
    memory = ProfileMemory(cast(Any, ProfileSDK()))
    state = initial()
    config = {"configurable": {"thread_id": str(uuid4())}}
    path = str(tmp_path / "memory-loop.sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await build_graph(
            RecordingModel(responses=[plan_response()]), checkpointer=saver, memory_factory=lambda: memory
        ).ainvoke(state, config=config, interrupt_after=["remember"])
    factory = Mock(side_effect=AssertionError("Do not replay a saved operation"))
    model = RecordingModel(responses=[])
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_graph(model, checkpointer=saver, memory_factory=factory)
        assert (await graph.aget_state(config)).next == ("finalize",)
        result = await graph.ainvoke(None, config=config)
    assert result["memory_result"]["status"] == "saved"
    assert result["budget"]["model_calls"] == 1 and model.calls == 0
    factory.assert_not_called()


@pytest.mark.anyio
async def test_one_model_call_budget_still_allows_authorized_non_model_write() -> None:
    sdk = ProfileSDK()
    state = initial()
    state["budget"] = Budget(max_model_calls=1).model_dump()
    result = await build_graph(
        RecordingModel(responses=[plan_response()]), memory_factory=lambda: ProfileMemory(cast(Any, sdk))
    ).ainvoke(state)
    assert result["memory_result"]["status"] == "saved"
    assert result["budget"]["model_calls"] == 1 and sdk.writes == 1


@pytest.mark.anyio
@pytest.mark.parametrize("budget", [Budget(max_model_calls=0), Budget(elapsed_seconds=181)])
async def test_exhausted_planner_budget_never_writes_or_claims_success(budget: Budget) -> None:
    state = initial()
    state["budget"] = budget.model_dump()
    factory = Mock(side_effect=AssertionError("Must not write without a planning decision"))
    result = await build_graph(RecordingModel(responses=[]), memory_factory=factory).ainvoke(state)
    assert result["memory_result"]["status"] == "not_saved"
    assert result["memory_result"]["reason"] == "budget"
    assert result["termination"] in {"model_budget", "time_budget"}
    factory.assert_not_called()


@pytest.mark.anyio
async def test_untrusted_scoped_recall_fails_instead_of_returning_foreign_data() -> None:
    sdk = Mock()
    sdk.get_all.return_value = {"results": [{"id": str(uuid4()), "user_id": "foreign", "agent_id": PROFILE_AGENT_ID}]}
    with pytest.raises(ValueError):
        await ProfileMemory(cast(Any, sdk)).recall("owner", "法规")
    sdk.search.assert_not_called()


@pytest.mark.anyio
async def test_fixed_preferences_are_capped_and_skip_unneeded_semantic_search() -> None:
    sdk = ProfileSDK()
    memory = ProfileMemory(cast(Any, sdk))
    user = str(uuid4())
    for _ in range(7):
        await memory.remember(user, PREFERENCE, task_id=str(uuid4()))
    recalled = await memory.recall(user, "任何业务主题")
    assert len(recalled) == 5 and sdk.searches == []


def test_explicit_research_goals_are_not_silently_discarded_by_pure_hint() -> None:
    result = memory_decision(
        COMMAND + "，查法规", {"memory_request": {"text": PREFERENCE, "quote": COMMAND}, "goals": ["查法规"]}
    )
    assert result["memory_only"] is False
    assert result["memory_result"]["status"] == "pending"
