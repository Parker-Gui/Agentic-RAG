"""Short-transaction conversation persistence; callers own transaction boundaries."""

from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.conversation import Conversation, ConversationTurn


async def create(db: AsyncSession, *, user_id: UUID, conversation_id: UUID | None = None) -> Conversation:
    row = Conversation(user_id=user_id)
    if conversation_id is not None:
        row.id = conversation_id
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row


async def get_owned(
    db: AsyncSession, *, conversation_id: UUID, user_id: UUID, lock: bool = False
) -> Conversation | None:
    stmt = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    if lock:
        stmt = stmt.with_for_update()
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_turns(
    db: AsyncSession, *, conversation_id: UUID, skip: int = 0, limit: int = 50, completed_only: bool = False
) -> tuple[list[ConversationTurn], int]:
    stmt = select(ConversationTurn).where(ConversationTurn.conversation_id == conversation_id)
    if completed_only:
        stmt = stmt.where(ConversationTurn.status == "completed")
    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = await db.execute(
        stmt.order_by(ConversationTurn.created_at.desc(), ConversationTurn.id.desc()).offset(skip).limit(limit)
    )
    return list(rows.scalars().all()), total


async def has_running(db: AsyncSession, *, conversation_id: UUID) -> bool:
    stmt = (
        select(ConversationTurn.id)
        .where(
            ConversationTurn.conversation_id == conversation_id, ConversationTurn.status.in_(["running", "interrupted"])
        )
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none() is not None


async def create_turn(db: AsyncSession, *, conversation_id: UUID, task_id: UUID, question: str) -> ConversationTurn:
    row = ConversationTurn(conversation_id=conversation_id, task_id=task_id, question=question)
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row


async def get_turn(db: AsyncSession, *, task_id: UUID) -> ConversationTurn | None:
    return (await db.execute(select(ConversationTurn).where(ConversationTurn.task_id == task_id))).scalar_one_or_none()


async def finish_turn(db: AsyncSession, *, task_id: UUID, state: dict[str, Any]) -> ConversationTurn | None:
    row = await get_turn(db, task_id=task_id)
    if row is not None:
        row.answer = state.get("answer")
        row.evidence = state.get("evidence", [])
        row.status = "completed" if state.get("termination") == "completed" else "failed"
        row.termination = state.get("termination")
        await db.flush()
        await db.refresh(row)
    return row


async def get_owned_turn(db: AsyncSession, *, task_id: UUID, user_id: UUID) -> ConversationTurn | None:
    stmt = (
        select(ConversationTurn)
        .join(Conversation)
        .where(ConversationTurn.task_id == task_id, Conversation.user_id == user_id)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_tasks(
    db: AsyncSession, *, user_id: UUID, conversation_id: UUID | None = None, skip: int = 0, limit: int | None = 50
) -> tuple[list[ConversationTurn], int]:
    stmt = select(ConversationTurn).join(Conversation).where(Conversation.user_id == user_id)
    if conversation_id is not None:
        stmt = stmt.where(ConversationTurn.conversation_id == conversation_id)
    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = await db.execute(
        stmt.order_by(ConversationTurn.created_at.desc(), ConversationTurn.id.desc()).offset(skip).limit(limit)
    )
    return list(rows.scalars().all()), total


async def unfinished_turns(db: AsyncSession, *, conversation_id: UUID) -> list[ConversationTurn]:
    rows = await db.execute(
        select(ConversationTurn).where(
            ConversationTurn.conversation_id == conversation_id, ConversationTurn.status.in_(["running", "interrupted"])
        )
    )
    return list(rows.scalars().all())


async def set_lifecycle(
    db: AsyncSession, *, task_id: UUID, status: str, recovery: dict[str, Any], state: dict[str, Any] | None = None
) -> ConversationTurn | None:
    row = await get_turn(db, task_id=task_id)
    if row is not None:
        row.status, row.recovery = status, recovery
        if state is not None:
            row.termination = state.get("termination")
            if state.get("answer"):
                row.answer = state["answer"]
            if "evidence" in state:
                row.evidence = state["evidence"]
        await db.flush()
        await db.refresh(row)
    return row
