"""Project-owned planning, execution, review and termination around Deep Agents."""

from collections.abc import Callable
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.memory.profile import ProfileMemory
from app.agents.nodes.draft import draft_answer
from app.agents.nodes.execute import execute_task
from app.agents.nodes.finalize import finalize_task
from app.agents.nodes.memory import remember_profile
from app.agents.nodes.plan import plan_task
from app.agents.nodes.review import review_task
from app.agents.providers import get_llm_provider
from app.agents.state import TaskState
from app.core.config import settings
from app.core.observability import budget_snapshot, observation, result_metadata, safe_metadata, set_operation_ids


def _after_execute(state: TaskState) -> str:
    if state.get("next_step") == "finalize":
        return "finalize"
    return "execute" if state.get("current_index", 0) < len(state.get("plan", [])) else "draft"


def _after_review(state: TaskState) -> str:
    return str(state.get("next_step", "finalize"))


def build_graph(
    model: BaseChatModel | None = None,
    *,
    backend_factory: Callable[[str, str], BackendProtocol] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    memory_factory: Callable[[], ProfileMemory] | None = None,
) -> Any:
    llm = model

    def resolve_model() -> BaseChatModel:
        nonlocal llm
        if llm is None:
            llm = get_llm_provider().build_model(max_tokens=settings.AGENT_OUTPUT_TOKENS)
        set_operation_ids(model_name=getattr(llm, "model_name", None) or getattr(llm, "model", None))
        return llm

    async def plan(state: TaskState) -> dict[str, Any]:
        with observation("stage.plan", **safe_metadata(state)) as record:
            before = budget_snapshot(state.get("budget", {}))
            result = await plan_task(state, resolve_model())
            record.update(**result_metadata(result, budget_before=before))
            return result

    async def execute(state: TaskState) -> dict[str, Any]:
        with observation("stage.execute", **safe_metadata(state)) as record:
            before = budget_snapshot(state.get("budget", {}))
            result = await execute_task(state, resolve_model(), backend_factory)
            record.update(**result_metadata(result, budget_before=before))
            return result

    async def remember(state: TaskState) -> dict[str, Any]:
        with observation("stage.memory", **safe_metadata(state)) as record:
            before = budget_snapshot(state.get("budget", {}))
            result = await remember_profile(state, memory_factory)
            record.update(**result_metadata(result, budget_before=before))
            return result

    def after_plan(state: TaskState) -> str:
        receipt = state.get("memory_result") or {}
        if receipt.get("status") == "pending":
            return "remember"
        return after_memory(state)

    def after_memory(state: TaskState) -> str:
        return "finalize" if state.get("memory_only") or state.get("termination") else "execute"

    async def draft(state: TaskState) -> dict[str, Any]:
        with observation("stage.draft", **safe_metadata(state)) as record:
            before = budget_snapshot(state.get("budget", {}))
            result = await draft_answer(state, resolve_model())
            record.update(**result_metadata(result, budget_before=before))
            return result

    async def review(state: TaskState) -> dict[str, Any]:
        with observation("stage.review", **safe_metadata(state)) as record:
            before = budget_snapshot(state.get("budget", {}))
            result = await review_task(state, resolve_model())
            record.update(**result_metadata(result, budget_before=before))
            return result

    async def finalize(state: TaskState) -> dict[str, Any]:
        with observation("stage.finalize", **safe_metadata(state)) as record:
            result = finalize_task(state)
            record.update(**result_metadata(result))
            return result

    workflow = StateGraph(TaskState)  # ty: ignore[invalid-argument-type]
    workflow.add_node("plan", plan)
    workflow.add_node("execute", execute)
    workflow.add_node("remember", remember)
    workflow.add_node("draft", draft)
    workflow.add_node("review", review)
    workflow.add_node("finalize", finalize)
    workflow.add_edge(START, "plan")
    workflow.add_conditional_edges(
        "plan", after_plan, {"remember": "remember", "finalize": "finalize", "execute": "execute"}
    )
    workflow.add_conditional_edges("remember", after_memory, {"finalize": "finalize", "execute": "execute"})
    workflow.add_conditional_edges(
        "execute", _after_execute, {"execute": "execute", "draft": "draft", "finalize": "finalize"}
    )
    workflow.add_conditional_edges(
        "draft", lambda state: str(state.get("next_step", "review")), {"review": "review", "finalize": "finalize"}
    )
    workflow.add_conditional_edges(
        "review", _after_review, {"execute": "execute", "draft": "draft", "finalize": "finalize"}
    )
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer)


graph = build_graph()
