"""One project-owned profile write at a checkpoint boundary, outside research tools."""

import asyncio
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from app.agents.memory.intent import memory_decision
from app.agents.memory.profile import ProfileMemory, get_profile_memory
from app.agents.models import Budget, MemoryRequest, MemoryResult, TerminationReason
from app.agents.state import TaskState


async def remember_profile(
    state: TaskState, memory_factory: Callable[[], ProfileMemory] | None = None
) -> dict[str, Any]:
    receipt = MemoryResult.model_validate(state["memory_result"])
    if receipt.status != "pending" or receipt.attempted:
        return {}
    verified = memory_decision(state["question"], {"memory_request": state.get("memory_request")})
    if not verified.get("memory_request"):
        return {"memory_result": verified["memory_result"]}
    request = MemoryRequest.model_validate(verified["memory_request"])
    budget = Budget.model_validate(state.get("budget", {}))
    reason = None
    if budget.elapsed_seconds >= budget.max_seconds:
        reason = TerminationReason.TIME_BUDGET.value
    if reason:
        return {
            "memory_result": MemoryResult(
                status="not_saved", message="任务预算已用尽，未保存长期偏好。", reason="budget"
            ).model_dump(mode="json"),
            "termination": reason,
        }
    started = time.monotonic()
    receipt = MemoryResult(status="not_saved", message="记忆组件不可用，未执行长期偏好写入。", reason="unavailable")
    profile = list(state.get("profile", []))
    try:
        memory = await asyncio.wait_for(
            asyncio.to_thread(memory_factory or get_profile_memory),
            timeout=max(0.1, budget.max_seconds - budget.elapsed_seconds),
        )
        remaining = budget.max_seconds - budget.elapsed_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Initialization exhausted time budget")
        receipt = MemoryResult(
            status="uncertain",
            message="长期记忆保存结果不确定，可能已生效，本次不会自动重试。",
            attempted=True,
            reason="write_uncertain",
        )
        rows = await asyncio.wait_for(
            memory.remember(state["user_id"], request.text, task_id=state["task_id"]),
            timeout=remaining,
        )
        if len(rows) != 1 or rows[0].get("memory") != request.text:
            raise ValueError("Invalid memory write receipt")
        identifier = UUID(str(rows[0]["id"]))
        receipt = MemoryResult(
            status="saved", message=f"已记住：{request.text}", memory_ids=[identifier], attempted=True
        )
        profile = list(dict.fromkeys([request.text, *profile]))[:5]
    except Exception:
        # Cancellation propagates to the lifecycle service; an interrupted thread
        # may still write. Ordinary uncertain outcomes are terminal for this node.
        pass
    budget.elapsed_seconds += time.monotonic() - started
    return {
        "memory_result": receipt.model_dump(mode="json"),
        "profile": profile,
        "budget": budget.model_dump(),
        "trace": [
            *state.get("trace", []),
            {"stage": "memory", "status": receipt.status, "attempted": receipt.attempted},
        ],
    }
