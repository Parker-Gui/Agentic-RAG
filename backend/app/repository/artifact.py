"""Artifact access and atomic insert-or-reuse; transaction ownership stays outside."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.artifact import Artifact


async def get_owned(db: AsyncSession, *, artifact_id: UUID, user_id: UUID) -> Artifact | None:
    result = await db.execute(select(Artifact).where(Artifact.id == artifact_id, Artifact.user_id == user_id))
    return result.scalar_one_or_none()


async def get_for_task(db: AsyncSession, *, task_id: UUID, user_id: UUID) -> Artifact | None:
    result = await db.execute(select(Artifact).where(Artifact.task_id == task_id, Artifact.user_id == user_id))
    return result.scalar_one_or_none()


async def create_or_get(
    db: AsyncSession,
    *,
    artifact_id: UUID,
    user_id: UUID,
    task_id: UUID,
    conversation_id: UUID | None,
    title: str,
    filename: str,
    content: str,
    size_bytes: int,
    sha256: str,
    created_at: datetime,
) -> tuple[Artifact | None, bool]:
    stmt = (
        insert(Artifact)
        .values(
            id=artifact_id,
            user_id=user_id,
            task_id=task_id,
            conversation_id=conversation_id,
            title=title,
            format="markdown",
            filename=filename,
            content=content,
            size_bytes=size_bytes,
            sha256=sha256,
            created_at=created_at,
        )
        .on_conflict_do_nothing(index_elements=["task_id"])
        .returning(Artifact.id)
    )
    inserted_id = (await db.execute(stmt)).scalar_one_or_none()
    row = await get_for_task(db, task_id=task_id, user_id=user_id)
    if row is not None:
        await db.flush()
        await db.refresh(row)
    return row, inserted_id is not None
