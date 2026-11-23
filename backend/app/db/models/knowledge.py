"""Unified sources, immutable input revisions and version-pinned vector chunks."""

from datetime import date
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import SPARSEVEC, Vector
from sqlalchemy import (
    BigInteger,
    Date,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin
from app.db.models.article_chunk import EMBED_DIM, SPARSE_DIM


class KnowledgeDocument(Base, TimestampMixin):
    __tablename__ = "knowledge_documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    source_key: Mapped[str] = mapped_column(String(500), unique=True)
    source_kind: Mapped[str] = mapped_column(String(20))
    article_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("articles.id", ondelete="CASCADE"), unique=True
    )
    active_revision_id: Mapped[UUID | None] = mapped_column(nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["id", "active_revision_id"],
            ["knowledge_document_revisions.document_id", "knowledge_document_revisions.id"],
            name="knowledge_documents_active_revision_fkey",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
        ),
    )

    def __repr__(self) -> str:
        return f"<KnowledgeDocument(id={self.id}, source_key={self.source_key!r})>"


class KnowledgeRevision(Base, TimestampMixin):
    __tablename__ = "knowledge_document_revisions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("knowledge_documents.id", ondelete="CASCADE"), index=True)
    processing_signature: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    input_bytes: Mapped[bytes] = mapped_column(LargeBinary)
    format: Mapped[str] = mapped_column(String(12))
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    site_name: Mapped[str | None] = mapped_column(String(200))
    publish_date: Mapped[date | None] = mapped_column(Date)
    group_type: Mapped[str] = mapped_column(String(50), default="internal")
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    state: Mapped[str] = mapped_column(String(20), default="received")
    failed_stage: Mapped[str | None] = mapped_column(String(20))
    error_code: Mapped[str | None] = mapped_column(String(60))
    error_message: Mapped[str | None] = mapped_column(Text)
    canonical_text: Mapped[str | None] = mapped_column(Text)
    body_hash: Mapped[str | None] = mapped_column(String(64))
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    warnings: Mapped[list[str]] = mapped_column(JSONB, default=list)

    __table_args__ = (
        UniqueConstraint("document_id", "processing_signature", name="knowledge_revisions_document_signature_key"),
        UniqueConstraint("document_id", "id", name="knowledge_revisions_document_id_key"),
    )

    def __repr__(self) -> str:
        return f"<KnowledgeRevision(id={self.id}, state={self.state!r})>"


class KnowledgeChunk(Base, TimestampMixin):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    revision_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_document_revisions.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    context: Mapped[dict[str, Any]] = mapped_column(JSONB)
    embedding_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    sparse_embedding: Mapped[Any] = mapped_column(SPARSEVEC(SPARSE_DIM), nullable=True)

    __table_args__ = (
        UniqueConstraint("revision_id", "chunk_index"),
        Index(
            "knowledge_chunks_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<KnowledgeChunk(id={self.id}, revision_id={self.revision_id}, index={self.chunk_index})>"
