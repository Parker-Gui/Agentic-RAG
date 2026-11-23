"""Verify the DeepSeek-compatible wire format without sending paid requests."""

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage

from app.agents.providers import OpenAIProvider
from app.core.config import settings
from app.core.exceptions import ExternalServiceError


@pytest.mark.anyio
async def test_suite_blocks_unmocked_chat_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    model = OpenAIProvider().build_model()
    with pytest.raises(AssertionError, match="Live chat calls forbidden"):
        await model.ainvoke("This must never leave the test process")


@pytest.mark.anyio
async def test_deepseek_request_uses_chat_completions_with_non_thinking_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "deepseek-flash")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIProvider().build_model(http_async_client=client)
        tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a fact",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        response = await model.bind_tools([tool]).ainvoke([HumanMessage(content="test")])
    assert response.content == "ok"
    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["thinking"] == {"type": "disabled"}
    assert body["tools"][0]["function"]["name"] == "lookup"


def test_other_openai_compatible_endpoints_do_not_receive_deepseek_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://example.org/v1")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    model = OpenAIProvider().build_model()
    assert getattr(model, "extra_body", None) is None


@pytest.mark.anyio
async def test_missing_api_key_allows_graph_construction_but_rejects_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents.graph import build_graph

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    graph = build_graph()
    assert "plan" in graph.nodes
    with pytest.raises(ExternalServiceError, match="OPENAI_API_KEY"):
        await graph.ainvoke({"question": "test"})


@pytest.mark.anyio
async def test_deepseek_tool_round_trip_through_controlled_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents.executor import run_executor

    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "deepseek-flash")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        assert body["thinking"] == {"type": "disabled"}
        if len(bodies) == 1:
            names = {tool["function"]["name"] for tool in body["tools"]}
            assert "task" not in names and "execute" not in names
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "search-1",
                        "type": "function",
                        "function": {"name": "search_rag_tool", "arguments": '{"query":"test","top_k":0}'},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            tool_messages = [item for item in body["messages"] if item["role"] == "tool"]
            assert json.loads(tool_messages[-1]["content"])["results"] == []
            message = {"role": "assistant", "content": "参数不合法，未取得证据。"}
            finish = "stop"
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIProvider().build_model(http_async_client=client, max_retries=0)
        result = await run_executor(model, "test", max_model_calls=3, max_tool_calls=2, max_seconds=30)
    assert len(bodies) == 2
    assert result.model_calls == 2 and result.tool_calls == 1
    assert result.messages[-1].content == "参数不合法，未取得证据。"
