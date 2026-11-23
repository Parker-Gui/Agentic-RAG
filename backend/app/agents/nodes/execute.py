"""Run one planned subtask and preserve its source records outside model context."""

import time
from collections.abc import Callable
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.context.conversation import history_prompt
from app.agents.context.work import MAX_WORK_TOKENS, work_prompt, work_tokens
from app.agents.evidence import from_tool_messages
from app.agents.executor import run_executor
from app.agents.models import Budget, Evidence, Subtask, TerminationReason
from app.agents.state import TaskState
from app.core.config import settings


async def execute_task(
    state: TaskState,
    model: BaseChatModel,
    backend_factory: Callable[[str, str], BackendProtocol] | None = None,
) -> dict[str, Any]:
    budget = Budget.model_validate(state.get("budget", {}))
    plan = [Subtask.model_validate(item) for item in state.get("plan", [])]
    index = state.get("current_index", 0)
    if index >= len(plan):
        return {"next_step": "draft"}
    reason = None
    if budget.model_calls >= budget.max_model_calls:
        reason = TerminationReason.MODEL_BUDGET
    elif budget.tool_calls >= budget.max_tool_calls:
        reason = TerminationReason.TOOL_BUDGET
    elif budget.elapsed_seconds >= budget.max_seconds:
        reason = TerminationReason.TIME_BUDGET
    if reason:
        return {"termination": reason.value, "next_step": "finalize"}

    subtask = plan[index]
    subtask.attempts += 1
    profile = "；".join(state.get("profile", [])[:5])
    prompt = (
        f"研究问题：{state['question']}\n当前子任务 [{subtask.id}]：{subtask.goal}\n"
        f"用户偏好（不能当证据）：{profile}\n"
        f"{history_prompt(state)}"
        "请使用工具查证。内部库优先，必要时外网补充。请勿把工具错误当事实。"
    )
    # Leave room for the question/history and existing executor/tool-schema checks.
    work_budget = max(
        0, min(MAX_WORK_TOKENS, settings.AGENT_INPUT_TOKENS // 8, settings.AGENT_INPUT_TOKENS - work_tokens(prompt))
    )
    prompt += work_prompt(state, subtask.goal, max_tokens=work_budget)
    started = time.monotonic()
    trace = list(state.get("trace", []))
    try:
        result = await run_executor(
            model,
            prompt,
            max_model_calls=budget.max_model_calls - budget.model_calls,
            max_tool_calls=budget.max_tool_calls - budget.tool_calls,
            max_seconds=max(0.1, budget.max_seconds - budget.elapsed_seconds),
            backend=backend_factory(state["user_id"], state["task_id"]) if backend_factory else None,
        )
    except Exception as exc:
        subtask.status = "failed"
        plan[index] = subtask
        trace.append({"stage": "execute", "subtask": subtask.id, "error": type(exc).__name__})
        return {
            "plan": [item.model_dump() for item in plan],
            "termination": TerminationReason.EXECUTION_ERROR.value,
            "next_step": "finalize",
            "trace": trace,
        }

    budget.model_calls += result.model_calls
    budget.tool_calls += result.tool_calls
    budget.elapsed_seconds += time.monotonic() - started
    budget.summary_calls += result.summary_calls
    budget.usage.extend(result.usage)
    extracted, errors = from_tool_messages(result.tool_messages, subtask.id)
    existing = {item.id: item for item in (Evidence.model_validate(row) for row in state.get("evidence", []))}
    for item in extracted:
        existing.setdefault(item.id, item)
    subtask.evidence_ids = list(dict.fromkeys([*subtask.evidence_ids, *(item.id for item in extracted)]))
    subtask.status = "covered" if subtask.evidence_ids else "missing"
    plan[index] = subtask
    trace.append(
        {
            "stage": "execute",
            "subtask": subtask.id,
            "model_calls": result.model_calls,
            "tool_calls": result.tool_calls,
            "evidence": len(extracted),
            "errors": len(errors),
            "offloaded_tool_results": result.offloaded_tool_results,
            "summary_calls": result.summary_calls,
            "context_events": result.context_events,
            "stop_reason": result.stop_reason,
        }
    )
    return {
        "plan": [item.model_dump() for item in plan],
        "current_index": index + 1,
        "evidence": [item.model_dump() for item in existing.values()],
        "tool_errors": [*state.get("tool_errors", []), *errors],
        "budget": budget.model_dump(),
        "trace": trace,
        "next_step": "finalize" if result.stop_reason else "execute",
        **({"termination": result.stop_reason} if result.stop_reason else {}),
    }
