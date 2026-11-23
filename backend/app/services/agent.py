"""Authenticated research task entry point and profile operations."""

import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

from langgraph.errors import GraphDrained
from langgraph.types import StateSnapshot

from app.agents.checkpoints import open_checkpointer
from app.agents.context.conversation import select_history
from app.agents.graph import build_graph
from app.agents.lifecycle import TERMINAL, checkpoint_id, complete_budget, task_view, terminal_status
from app.agents.memory.notes import notes_backend
from app.agents.memory.profile import get_profile_memory
from app.agents.models import Budget
from app.agents.providers import get_llm_provider
from app.agents.runtime import ActiveRun, registry
from app.core.config import settings
from app.core.exceptions import AlreadyExistsError, ExternalServiceError, NotFoundError
from app.core.observability import callback_config, set_operation_budget, set_operation_ids, trace_operation
from app.db.models.conversation import ConversationTurn
from app.db.session import get_worker_db_context
from app.repository import conversation as conversation_repo

logger = logging.getLogger(__name__)


class AgentService:
    async def _begin_turn(
        self,
        user_id: UUID,
        question: str,
        task_id: UUID,
        conversation_id: UUID | None,
        *,
        new_conversation_id: UUID | None = None,
    ) -> tuple[UUID, str, list[dict[str, Any]]]:
        async with get_worker_db_context() as db:
            if conversation_id is None:
                if new_conversation_id is None:
                    conversation = await conversation_repo.create(db, user_id=user_id)
                else:
                    conversation = await conversation_repo.create(
                        db, user_id=user_id, conversation_id=new_conversation_id
                    )
            else:
                conversation = await conversation_repo.get_owned(
                    db, conversation_id=conversation_id, user_id=user_id, lock=True
                )
                if conversation is None:
                    raise NotFoundError(message="Conversation not found")
                if await conversation_repo.has_running(db, conversation_id=conversation_id):
                    rows = await conversation_repo.unfinished_turns(db, conversation_id=conversation_id)
                    for row in rows:
                        snapshot = await self._snapshot(user_id, row)
                        view = task_view(row, snapshot)
                        await conversation_repo.set_lifecycle(
                            db, task_id=row.task_id, status=view["status"], recovery=row.recovery or {}, state=view
                        )
                    if await conversation_repo.has_running(db, conversation_id=conversation_id):
                        raise AlreadyExistsError(
                            message="对话有未结束任务，请先恢复或放弃。", details={"reason": "unfinished_task"}
                        )
            turns, _ = await conversation_repo.list_turns(
                db, conversation_id=conversation.id, limit=settings.AGENT_HISTORY_TURNS, completed_only=True
            )
            history, evidence = select_history(
                [
                    {
                        "task_id": row.task_id,
                        "question": row.question,
                        "answer": row.answer,
                        "evidence": row.evidence,
                        "status": row.status,
                    }
                    for row in turns
                ]
            )
            await conversation_repo.create_turn(db, conversation_id=conversation.id, task_id=task_id, question=question)
            return conversation.id, history, evidence

    async def _finish_turn(self, task_id: UUID, state: dict[str, Any]) -> None:
        async with get_worker_db_context() as db:
            await conversation_repo.finish_turn(db, task_id=task_id, state=state)

    async def get_conversation(
        self, user_id: UUID, conversation_id: UUID, *, skip: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        async with get_worker_db_context() as db:
            conversation = await conversation_repo.get_owned(db, conversation_id=conversation_id, user_id=user_id)
            if conversation is None:
                raise NotFoundError(message="Conversation not found")
            turns, total = await conversation_repo.list_turns(
                db, conversation_id=conversation_id, skip=skip, limit=limit
            )
            return {"id": conversation.id, "items": turns, "total": total}

    @trace_operation("run")
    async def run(self, user_id: UUID, question: str, *, conversation_id: UUID | None = None) -> dict[str, Any]:
        model = get_llm_provider().build_model(max_tokens=settings.AGENT_OUTPUT_TOKENS)
        task_id = uuid4()
        if conversation_id is not None:
            async with get_worker_db_context() as db:
                if await conversation_repo.get_owned(db, conversation_id=conversation_id, user_id=user_id) is None:
                    raise NotFoundError(message="Conversation not found")
        allocated = conversation_id or uuid4()
        with registry.claim(user_id, allocated, task_id) as run:
            _, history, evidence = await self._begin_turn(
                user_id,
                question,
                task_id,
                conversation_id,
                new_conversation_id=allocated if conversation_id is None else None,
            )
            set_operation_ids(task_id=task_id, conversation_id=allocated, model_name=getattr(model, "model_name", None))
            try:
                memory = await asyncio.to_thread(get_profile_memory)
                memories = await memory.recall(str(user_id), question)
                profile = [str(item["memory"]) for item in memories if item.get("memory")]
                profile_status = "available"
            except asyncio.CancelledError:
                await self._mark(task_id, "interrupted", {"version": 1, "kind": "uncertain"})
                raise
            except Exception as exc:
                logger.warning("Profile recall unavailable: %s", type(exc).__name__)
                profile = []
                profile_status = "unavailable"
            initial = {
                "question": question,
                "user_id": str(user_id),
                "task_id": str(task_id),
                "conversation_id": str(allocated),
                "conversation_history": history,
                "profile": profile,
                "budget": Budget().model_dump(),
                "evidence": evidence,
                "trace": [{"stage": "profile_recall", "status": profile_status}],
            }
            return await self._drive(user_id, run, initial, model=model)

    async def _owned_task(self, user_id: UUID, task_id: UUID) -> ConversationTurn:
        async with get_worker_db_context() as db:
            row = await conversation_repo.get_owned_turn(db, task_id=task_id, user_id=user_id)
        if row is None:
            raise NotFoundError(message="Agent task not found")
        return row

    async def _snapshot(self, user_id: UUID, row: ConversationTurn) -> StateSnapshot:
        async with open_checkpointer() as saver:
            snapshot = await build_graph(checkpointer=saver).aget_state(
                {"configurable": {"thread_id": f"{user_id}:{row.task_id}"}}
            )
        if snapshot.values and snapshot.values.get("user_id") != str(user_id):
            raise ExternalServiceError(message="任务检查点与登记不一致。")
        return snapshot

    async def _mark(
        self, task_id: UUID, status: str, receipt: dict[str, Any], state: dict[str, Any] | None = None
    ) -> ConversationTurn:
        async with get_worker_db_context() as db:
            row = await conversation_repo.set_lifecycle(
                db, task_id=task_id, status=status, recovery=receipt, state=state
            )
        if row is None:
            raise NotFoundError(message="Agent task not found")
        return row

    async def _drive(
        self, user_id: UUID, run: ActiveRun, initial: dict[str, Any] | None, *, model: Any = None
    ) -> dict[str, Any]:
        # Commit BEFORE executing: a prior clean receipt must never survive a new attempt.
        await self._mark(run.task_id, "running", {"version": 1, "kind": "in_flight", "execution_id": str(uuid4())})
        config = {"configurable": {"thread_id": f"{user_id}:{run.task_id}"}, "recursion_limit": 40, **callback_config()}
        try:
            async with open_checkpointer() as saver:
                graph = build_graph(
                    model, backend_factory=notes_backend, checkpointer=saver, memory_factory=get_profile_memory
                )
                drained = False
                try:
                    await graph.ainvoke(initial, config=config, control=run.control, durability="sync")
                except GraphDrained:
                    drained = True
                snapshot = await graph.aget_state(config)
                state = dict(snapshot.values)
                if drained and snapshot.next:
                    if not complete_budget(state) or not checkpoint_id(snapshot):
                        raise ExternalServiceError(message="暂停检查点不完整，禁止自动续跑。")
                    status = "interrupted"
                    receipt = {"version": 1, "kind": "clean_boundary", "checkpoint_id": checkpoint_id(snapshot)}
                else:
                    status, receipt = terminal_status(state), {}
        except (Exception, asyncio.CancelledError):
            try:
                await self._mark(run.task_id, "interrupted", {"version": 1, "kind": "uncertain"})
            except Exception:
                logger.warning("Task interruption metadata unavailable")
            raise
        # Metadata failure here leaves the completed/safe checkpoint intact; never rerun it.
        row = await self._mark(run.task_id, status, receipt, state)
        return task_view(row, snapshot)

    async def get_task(self, user_id: UUID, task_id: UUID) -> dict[str, Any]:
        row = await self._owned_task(user_id, task_id)
        run = registry.find(row.conversation_id)
        active = run is not None and run.task_id == task_id
        return task_view(
            row, await self._snapshot(user_id, row), active=active, pausing=active and run.control.drain_requested
        )

    @trace_operation("resume")
    async def resume(self, user_id: UUID, task_id: UUID) -> dict[str, Any]:
        row = await self._owned_task(user_id, task_id)
        with registry.claim(user_id, row.conversation_id, task_id) as run:
            row = await self._owned_task(user_id, task_id)
            snapshot = await self._snapshot(user_id, row)
            view = task_view(row, snapshot)
            set_operation_ids(conversation_id=row.conversation_id)
            set_operation_budget(view.get("budget", {}))
            if row.status == "abandoned":
                raise AlreadyExistsError(message="任务已放弃。", details={"reason": "task_abandoned"})
            if view["status"] in TERMINAL:
                if view["answer"]:
                    await self._mark(task_id, view["status"], {}, view)
                    return view
                raise AlreadyExistsError(
                    message="任务无法继续，请重新提问。", details={"reason": view["recovery"]["blocked_reason"]}
                )
            if not view["recovery"]["can_resume"]:
                raise AlreadyExistsError(
                    message="任务用量不确定，请先放弃再重新提问。", details={"reason": "unsafe_checkpoint"}
                )
            return await self._drive(user_id, run, None)

    async def pause(self, user_id: UUID, task_id: UUID) -> tuple[dict[str, Any], bool]:
        row = await self._owned_task(user_id, task_id)
        run = registry.find(row.conversation_id)
        if run is not None and run.task_id == task_id and run.user_id == user_id:
            run.control.request_drain("user_pause")
            return {
                "task_id": str(task_id),
                "conversation_id": str(row.conversation_id),
                "question": row.question,
                "status": "running",
                "pause_requested": True,
            }, True
        return await self.get_task(user_id, task_id), False

    async def abandon(self, user_id: UUID, task_id: UUID) -> dict[str, Any]:
        row = await self._owned_task(user_id, task_id)
        with registry.claim(user_id, row.conversation_id, task_id):
            row = await self._owned_task(user_id, task_id)
            snapshot = await self._snapshot(user_id, row)
            view = task_view(row, snapshot)
            if view["status"] in TERMINAL:
                return view
            row = await self._mark(task_id, "abandoned", {"version": 1, "kind": "abandoned"})
            return task_view(row, snapshot)

    async def list_tasks(
        self,
        user_id: UUID,
        *,
        conversation_id: UUID | None = None,
        status: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        async with get_worker_db_context() as db:
            rows, total = await conversation_repo.list_tasks(
                db,
                user_id=user_id,
                conversation_id=conversation_id,
                skip=0 if status else skip,
                limit=None if status else limit,
            )
        items = []
        for row in rows:
            view = await self.get_task(user_id, row.task_id)
            if status is None or view["status"] == status:
                items.append(view)
        if status:
            total, items = len(items), items[skip : skip + limit]
        return {"items": items, "total": total}

    async def remember(self, user_id: UUID, text: str) -> list[dict[str, Any]]:
        memory = await asyncio.to_thread(get_profile_memory)
        return await memory.remember(str(user_id), text)

    async def update_memory(self, user_id: UUID, memory_id: UUID, text: str) -> dict[str, Any]:
        try:
            memory = await asyncio.to_thread(get_profile_memory)
            return await memory.update(str(user_id), str(memory_id), text)
        except PermissionError as exc:
            raise NotFoundError(message="Profile memory not found", details={"memory_id": str(memory_id)}) from exc

    async def forget(self, user_id: UUID, memory_id: UUID) -> None:
        try:
            memory = await asyncio.to_thread(get_profile_memory)
            await memory.forget(str(user_id), str(memory_id))
        except PermissionError as exc:
            raise NotFoundError(message="Profile memory not found", details={"memory_id": str(memory_id)}) from exc

    async def memory_history(self, user_id: UUID, memory_id: UUID) -> list[dict[str, Any]]:
        try:
            memory = await asyncio.to_thread(get_profile_memory)
            return await memory.history(str(user_id), str(memory_id))
        except PermissionError as exc:
            raise NotFoundError(message="Profile memory not found", details={"memory_id": str(memory_id)}) from exc
