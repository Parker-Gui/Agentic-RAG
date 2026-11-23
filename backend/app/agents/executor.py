"""A bounded Deep Agents tool loop; the outer graph owns planning and review."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, cast

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent, register_harness_profile
from deepagents._models import get_model_provider
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.summarization import DEEPAGENTS_DEFAULT_SUMMARY_PROMPT, SummarizationMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import get_buffer_string
from langgraph.prebuilt import ToolNode
from langgraph.runtime import RunControl
from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from app.agents.agent_tools import tools as business_tools
from app.agents.agent_tools.errors import tool_error_to_json
from app.agents.context.conversation import estimate_tokens
from app.core.config import settings
from app.core.observability import callback_config, observation

FILE_TOOLS = ["ls", "read_file", "write_file", "edit_file"]
BUSINESS_TOOL_NAMES = {tool.name for tool in business_tools}


class ExecutionStopped(RuntimeError):
    """A model call, including a summary call, exhausted the shared budget."""


class ExecutionLimits(AgentMiddleware):
    """Per-invocation counters and structured tool failures, including invalid arguments."""

    def __init__(self, max_model_calls: int, max_tool_calls: int, deadline: float) -> None:
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.deadline = deadline
        self.model_calls = 0
        self.tool_calls = 0
        self.stop_reason: str | None = None
        self.raw_tool_messages: list[ToolMessage] = []
        self.summary_calls = 0
        self.usage: list[dict[str, Any]] = []

    def reserve_model_call(self) -> None:
        if time.monotonic() >= self.deadline:
            self.stop_reason = "time_budget"
        elif self.model_calls >= self.max_model_calls:
            self.stop_reason = "model_budget"
        if self.stop_reason:
            raise ExecutionStopped(self.stop_reason)
        self.model_calls += 1

    def record_usage(self, response: AIMessage, purpose: str, input_estimate: int) -> None:
        self.usage.append(
            {
                "purpose": purpose,
                "estimated_input_tokens": input_estimate,
                "usage": dict(response.usage_metadata) if response.usage_metadata else None,
            }
        )

    def _record(self, message: ToolMessage) -> ToolMessage:
        if message.name in BUSINESS_TOOL_NAMES:
            self.raw_tool_messages.append(message.model_copy(deep=True))
        return message

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Any,
    ) -> ModelResponse[Any] | AIMessage:
        messages = [request.system_message, *request.messages] if request.system_message else request.messages
        input_estimate = estimate_tokens(messages, tools=request.tools)
        if input_estimate > settings.AGENT_INPUT_TOKENS:
            raise ContextOverflowError("请求超出应用输入预算；需要压缩或缩小输入。")
        self.reserve_model_call()
        response = await handler(request)
        for message in response.result:
            if isinstance(message, AIMessage):
                self.record_usage(message, "execute", input_estimate)
        return response

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Any) -> ToolMessage | Command[Any]:
        call = request.tool_call
        name = str(call["name"])
        if time.monotonic() >= self.deadline:
            self.stop_reason = "time_budget"
        elif self.tool_calls >= self.max_tool_calls:
            self.stop_reason = "tool_budget"
        if self.stop_reason:
            return self._record(
                ToolMessage(
                    content=tool_error_to_json(RuntimeError("执行预算已用尽")),
                    tool_call_id=str(call["id"]),
                    name=name,
                    status="error",
                )
            )
        self.tool_calls += 1
        schema = request.tool.args_schema if request.tool is not None else None
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema.model_validate(call["args"])
            except ValidationError as exc:
                return self._record(
                    ToolMessage(
                        content=tool_error_to_json(exc),
                        tool_call_id=str(call["id"]),
                        name=name,
                        status="error",
                    )
                )
        try:
            result = await handler(request)
        except Exception as exc:
            return self._record(
                ToolMessage(
                    content=tool_error_to_json(exc),
                    tool_call_id=str(call["id"]),
                    name=name,
                    status="error",
                )
            )
        if not isinstance(result, (ToolMessage, Command)):
            raise TypeError(f"Unexpected tool response: {type(result).__name__}")
        if isinstance(result, ToolMessage) and result.status == "error" and name in BUSINESS_TOOL_NAMES:
            try:
                payload = json.loads(str(result.content))
            except (TypeError, json.JSONDecodeError):
                payload = None
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                result.content = tool_error_to_json(RuntimeError(str(result.content)))
        return self._record(result) if isinstance(result, ToolMessage) else result


@dataclass(slots=True)
class ExecutionResult:
    messages: list[AnyMessage]
    tool_messages: list[ToolMessage]
    model_calls: int
    tool_calls: int
    offloaded_tool_results: int
    stop_reason: str | None
    summary_calls: int = 0
    usage: list[dict[str, Any]] = field(default_factory=list)
    context_events: list[dict[str, Any]] = field(default_factory=list)


def build_executor(
    model: BaseChatModel,
    limits: ExecutionLimits,
    *,
    backend: BackendProtocol | None = None,
) -> Any:
    """Register the 0.7.32 profile, then assert the compiled tools satisfy policy."""
    # Prebuilt models are resolved by their reported provider, which can differ
    # from the caller's configured label (especially for test and proxy models).
    resolved_provider = get_model_provider(model)
    if not resolved_provider:
        raise ValueError("Deep Agents could not identify the model provider")
    register_harness_profile(
        resolved_provider,
        HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
    )
    storage = backend or StateBackend()

    async def summarize(messages: list[AnyMessage]) -> str:
        # Reuse the official prompt and selection algorithm. This adapter adds only
        # shared budget accounting and explicit output validation to the model call.
        prompt = DEEPAGENTS_DEFAULT_SUMMARY_PROMPT.format(messages=get_buffer_string(messages, format="xml"))
        input_estimate = estimate_tokens([HumanMessage(content=prompt)])
        if input_estimate > settings.AGENT_INPUT_TOKENS:
            raise ContextOverflowError("待摘要原文超出应用预算；未丢弃原始消息。")
        limits.reserve_model_call()
        limits.summary_calls += 1
        with observation("stage.summary", summary_calls=1):
            response = await model.ainvoke(
                [HumanMessage(content=prompt)], max_tokens=settings.AGENT_OUTPUT_TOKENS, config=callback_config()
            )
        limits.record_usage(response, "summary", input_estimate)
        if not response.text.strip() or response.response_metadata.get("finish_reason") == "length":
            raise ValueError("摘要为空或输出被截断，不能作为完整摘要继续执行")
        return response.text.strip()

    agent = create_deep_agent(
        model=model,
        tools=business_tools,
        backend=storage,
        middleware=cast(
            Any,
            [
                FilesystemMiddleware(
                    backend=storage,
                    tools=["ls", "read_file", "write_file", "edit_file"],
                    tool_token_limit_before_evict=6000,
                ),
                SummarizationMiddleware(
                    model,
                    backend=storage,
                    trigger=("tokens", settings.AGENT_SUMMARY_TRIGGER_TOKENS),
                    keep=("tokens", settings.AGENT_KEEP_TOKENS),
                    token_counter=estimate_tokens,
                    trim_tokens_to_summarize=None,
                    truncate_args_settings={
                        "trigger": ("tokens", settings.AGENT_SUMMARY_TRIGGER_TOKENS),
                        "keep": ("tokens", settings.AGENT_KEEP_TOKENS),
                    },
                    summarizer=summarize,
                ),
                limits,
            ],
        ),
        subagents=[],
        skills=None,
        memory=None,
        system_prompt=(
            "你是研究执行器。先查内部知识库，不足时使用外部检索；工具错误不是证据。"
            "仅执行当前子任务。可在 /notes/ 保存任务进度，但不要把 profile 当研究事实。"
            "工具结果与来源由外层流程记录；回答务必说明未查到的信息。"
        ),
    )
    names = set(cast(ToolNode, agent.nodes["tools"].bound).tools_by_name)
    expected = BUSINESS_TOOL_NAMES | set(FILE_TOOLS)
    if names != expected:
        raise RuntimeError(f"Unexpected Deep Agents tools: {sorted(names ^ expected)}")
    return agent


async def run_executor(
    model: BaseChatModel,
    prompt: str,
    *,
    max_model_calls: int,
    max_tool_calls: int,
    max_seconds: float,
    backend: BackendProtocol | None = None,
) -> ExecutionResult:
    deadline = time.monotonic() + max_seconds
    limits = ExecutionLimits(max_model_calls, max_tool_calls, deadline)
    agent = build_executor(model, limits, backend=backend)
    try:
        output = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": 2 * (max_model_calls + max_tool_calls) + 8, **callback_config()},
                output_keys=["messages", "_summarization_event"],
                control=RunControl(),
            ),
            timeout=max_seconds,
        )
    except Exception as exc:
        reason = "time_budget" if isinstance(exc, TimeoutError) else limits.stop_reason or "execution_error"
        return ExecutionResult(
            [],
            limits.raw_tool_messages,
            limits.model_calls,
            limits.tool_calls,
            0,
            reason,
            limits.summary_calls,
            limits.usage,
            [{"event": "stopped", "error": type(exc).__name__}],
        )
    raw_by_call = {message.tool_call_id: str(message.content) for message in limits.raw_tool_messages}
    offloaded = sum(
        1
        for message in output["messages"]
        if isinstance(message, ToolMessage)
        and message.tool_call_id in raw_by_call
        and str(message.content) != raw_by_call[message.tool_call_id]
    )
    return ExecutionResult(
        messages=output["messages"],
        tool_messages=limits.raw_tool_messages,
        model_calls=limits.model_calls,
        tool_calls=limits.tool_calls,
        offloaded_tool_results=offloaded,
        stop_reason=limits.stop_reason,
        summary_calls=limits.summary_calls,
        usage=limits.usage,
        context_events=[
            {key: event.get(key) for key in ("cutoff_index", "file_path", "created_at_index")}
            for event in [output.get("_summarization_event")]
            if event
        ],
    )
