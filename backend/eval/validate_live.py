"""Opt-in live validation; agent and budget stages make billed model calls.

Run from backend/: uv run python -m eval.validate_live --help
Run memory-write then memory-read in separate processes to verify persistence.
memory-failure injects a SQLite history failure into a disposable profile fact.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

from sqlalchemy import text

from app.agents.memory.profile import close_profile_memory, get_profile_memory
from app.db.session import get_worker_db_context, run_compatible_async

REPORTS = Path(__file__).parent / "reports"
MEMORY_STATE = REPORTS / "live_memory_restart_state.json"


def save_report(name: str, value: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"live_{name}_validation.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


async def validate_rag() -> dict[str, Any]:
    from app.services.rag import RagService

    cases = [
        ("EFSA E 962 aspartame acesulfame salt safety assessment", "E 962"),
        ("阿斯巴甜安赛蜜盐 E 962 安全性", "E 962"),
        ("Alternaria toxins genotoxic DNA risk assessment", "Alternaria"),
        ("TFA safe exposure level", "TFA"),
    ]
    checks: list[dict[str, Any]] = []
    for question, title_term in cases:
        started = time.monotonic()
        async with get_worker_db_context() as db:
            hits = await RagService(db).query(question, top_k=3)
        checks.append(
            {
                "question": question,
                "article_ids": [item["article_id"] for item in hits],
                "scores": [round(item["score"], 4) for item in hits],
                "expected_title_term": title_term,
                "hit_at_3": any(title_term.casefold() in item["title"].casefold() for item in hits),
                "seconds": round(time.monotonic() - started, 3),
            }
        )
    async with get_worker_db_context() as db:
        row = (
            await db.execute(
                text("SELECT COUNT(*),MIN(vector_dims(embedding)),MAX(vector_dims(embedding)) FROM article_chunks")
            )
        ).one()
    result = {"chunks": row[0], "dimensions": [row[1], row[2]], "cases": checks}
    save_report("rag", result)
    return result


async def validate_agent() -> dict[str, Any]:
    from app.services.agent import AgentService

    user_id = uuid4()
    question = "请仅根据内部知识库，简要说明 EFSA 对 E 962（阿斯巴甜-安赛蜜盐）的安全性结论并给出处。不需要联网。"
    started = time.monotonic()
    service = AgentService()
    try:
        result = await service.run(user_id, question)
        restored = await service.get_task(user_id, UUID(result["task_id"]))
        result["persisted_answer_matches"] = restored.get("answer") == result.get("answer")
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        save_report("agent", result)
        return {
            key: result.get(key)
            for key in (
                "task_id",
                "termination",
                "answer",
                "budget",
                "review",
                "trace",
                "wall_seconds",
                "persisted_answer_matches",
            )
        }
    finally:
        await asyncio.to_thread(close_profile_memory)


async def validate_budget() -> dict[str, Any]:
    from app.agents.graph import build_graph
    from app.agents.models import Budget

    result = await build_graph().ainvoke(
        {
            "question": "仅根据内部库说明 EFSA 对 E 962 的结论。",
            "user_id": str(uuid4()),
            "task_id": str(uuid4()),
            "budget": Budget(max_model_calls=1, max_tool_calls=1, max_seconds=30).model_dump(),
        }
    )
    report = {"termination": result.get("termination"), "answer": result.get("answer"), "budget": result.get("budget")}
    save_report("budget", report)
    return report


async def validate_checkpoint_read() -> dict[str, Any]:
    """Read a completed live task in a new process without invoking the model."""
    from app.core.exceptions import NotFoundError
    from app.services.agent import AgentService

    previous = json.loads((REPORTS / "live_agent_validation.json").read_text(encoding="utf-8"))
    service = AgentService()
    user_id, task_id = UUID(previous["user_id"]), UUID(previous["task_id"])
    restored = await service.get_task(user_id, task_id)
    resumed = await service.resume(user_id, task_id)
    denied = False
    try:
        await service.get_task(uuid4(), task_id)
    except NotFoundError:
        denied = True
    result = {
        "task_id": str(task_id),
        "answer_matches_after_restart": restored.get("answer") == previous["answer"],
        "completed_resume_returns_same_answer": resumed.get("answer") == previous["answer"],
        "budget_unchanged": resumed.get("budget") == previous["budget"],
        "foreign_user_denied": denied,
    }
    save_report("checkpoint", result)
    return result


async def validate_memory_failure() -> dict[str, Any]:
    from app.core.exceptions import ExternalServiceError

    profile = await asyncio.to_thread(get_profile_memory)
    user_id = str(uuid4())
    added = await profile.remember(user_id, "我长期关注 EFSA 甜味剂法规。")
    memory_id = added[0]["id"]
    try:
        before_history = await profile.history(user_id, memory_id)
        observed: dict[str, Any] = {}
        with patch.object(profile._memory.db, "add_history", side_effect=OSError("injected audit failure")):
            try:
                await profile.update(user_id, memory_id, "我长期关注 EFSA 食品添加剂法规。")
            except ExternalServiceError as exc:
                observed = {"status_code": exc.status_code, "details": exc.details}
        current = await profile._owned(user_id, memory_id)
        after_history = await profile.history(user_id, memory_id)
        result = {
            "injected_failure": "SQLite add_history",
            "api_error": observed,
            "vector_write_already_applied": current["memory"] == "我长期关注 EFSA 食品添加剂法规。",
            "history_unchanged": len(before_history) == len(after_history),
            "cross_store_atomicity": False,
        }
        save_report("memory_failure", result)
        return result
    finally:
        await profile.forget(user_id, memory_id)
        await asyncio.to_thread(close_profile_memory)


async def validate_memory_write() -> dict[str, Any]:
    profile = await asyncio.to_thread(get_profile_memory)
    user_id, other_id = str(uuid4()), str(uuid4())
    try:
        added = await profile.remember(user_id, "我长期关注 EFSA 甜味剂法规，回答请用中文。")
        if not added:
            raise RuntimeError("Mem0 did not store the explicit profile fact")
        state = {"user_id": user_id, "other_id": other_id, "memory_ids": [row["id"] for row in added]}
        REPORTS.mkdir(parents=True, exist_ok=True)
        MEMORY_STATE.write_text(json.dumps(state), encoding="utf-8")
        return {"stored": len(added), "restart_state_saved": True}
    finally:
        await asyncio.to_thread(close_profile_memory)


async def validate_memory_read() -> dict[str, Any]:
    state = json.loads(MEMORY_STATE.read_text(encoding="utf-8"))
    profile = await asyncio.to_thread(get_profile_memory)
    user_id, other_id = state["user_id"], state["other_id"]
    try:
        query = "EFSA 甜味剂法规"
        own = await profile.recall(user_id, query)
        other = await profile.recall(other_id, query)
        denied = False
        try:
            await profile.update(other_id, state["memory_ids"][0], "越权修改")
        except PermissionError:
            denied = True
        await profile.update(user_id, state["memory_ids"][0], "我长期关注 EFSA 食品添加剂法规。")
        updated = await profile.recall(user_id, "EFSA 食品添加剂法规")
        history = await profile.history(user_id, state["memory_ids"][0])
        for memory_id in state["memory_ids"]:
            await profile.forget(user_id, memory_id)
        forgotten = await profile.recall(user_id, query)
        result = {
            "same_user_hits_after_restart": len(own),
            "other_user_hits": len(other),
            "foreign_update_denied": denied,
            "updated_text_visible": any(row["memory"] == "我长期关注 EFSA 食品添加剂法规。" for row in updated),
            "history_rows": len(history),
            "hits_after_forget": len(forgotten),
        }
        save_report("memory", result)
        return result
    finally:
        # Only IDs created by the paired write stage are cleaned up.
        for memory_id in state["memory_ids"]:
            try:
                await profile.forget(user_id, memory_id)
            except PermissionError:
                pass
        await asyncio.to_thread(close_profile_memory)


async def main(stage: str) -> None:
    if stage == "rag":
        result = await validate_rag()
    elif stage == "agent":
        result = await validate_agent()
    elif stage == "budget":
        result = await validate_budget()
    elif stage == "checkpoint-read":
        result = await validate_checkpoint_read()
    elif stage == "memory-failure":
        result = await validate_memory_failure()
    elif stage == "memory-write":
        result = await validate_memory_write()
    else:
        result = await validate_memory_read()
    print(json.dumps(result, ensure_ascii=True, indent=2))  # noqa: T201 - validation CLI output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["rag", "agent", "budget", "checkpoint-read", "memory-write", "memory-read", "memory-failure"],
        required=True,
    )
    run_compatible_async(main(parser.parse_args().stage))
