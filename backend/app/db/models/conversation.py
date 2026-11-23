"""Conversation metadata; detailed execution remains in LangGraph checkpoints."""

import uuid
from typing import Any

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Conversation(Base, TimestampMixin):
    __tablename__ = "agent_conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, user_id={self.user_id})>"


class ConversationTurn(Base, TimestampMixin):
    __tablename__ = "agent_conversation_turns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_conversations.id", ondelete="CASCADE"), index=True
    )
    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    termination: Mapped[str | None] = mapped_column(String(40), nullable=True)
    recovery: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)

    def __repr__(self) -> str:
        return f"<ConversationTurn(task_id={self.task_id}, status={self.status})>"
