"""Immutable Markdown snapshots of reviewed research answers."""

import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Artifact(Base, TimestampMixin):
    __tablename__ = "agent_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agent_conversations.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(200))
    format: Mapped[str] = mapped_column(String(20), default="markdown")
    filename: Mapped[str] = mapped_column(String(80))
    content: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))

    def __repr__(self) -> str:
        return f"<Artifact(id={self.id}, task_id={self.task_id})>"
