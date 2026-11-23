"""Add unified knowledge sources, revisions and chunks.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import SPARSEVEC, Vector
from sqlalchemy.dialects import postgresql

revision = "b8c9d0e1f2a3"
down_revision = "a7b8c9d0e1f2"
branch_labels = None
depends_on = None


def timestamps() -> list[sa.Column]:
    return [sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True)]


def upgrade() -> None:
    op.add_column("articles", sa.Column("ingestion_payload", postgresql.JSONB(), nullable=True))
    op.create_table("knowledge_documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("source_key", sa.String(500), nullable=False, unique=True),
        sa.Column("source_kind", sa.String(20), nullable=False),
        sa.Column("article_id", sa.BigInteger(), sa.ForeignKey("articles.id", ondelete="CASCADE"), unique=True),
        sa.Column("active_revision_id", sa.Uuid(), nullable=True), *timestamps())
    op.create_table("knowledge_document_revisions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("document_id", sa.Uuid(), sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("processing_signature", sa.String(64), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("input_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("format", sa.String(12), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text()), sa.Column("site_name", sa.String(200)), sa.Column("publish_date", sa.Date()),
        sa.Column("group_type", sa.String(50), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("state", sa.String(20), nullable=False),
        sa.Column("failed_stage", sa.String(20)), sa.Column("error_code", sa.String(60)), sa.Column("error_message", sa.Text()),
        sa.Column("canonical_text", sa.Text()), sa.Column("body_hash", sa.String(64)),
        sa.Column("blocks", postgresql.JSONB(), nullable=False), sa.Column("warnings", postgresql.JSONB(), nullable=False),
        *timestamps(), sa.UniqueConstraint("document_id", "processing_signature", name="knowledge_revisions_document_signature_key"),
        sa.UniqueConstraint("document_id", "id", name="knowledge_revisions_document_id_key"))
    op.create_index("knowledge_document_revisions_document_id_idx", "knowledge_document_revisions", ["document_id"])
    op.create_foreign_key("knowledge_documents_active_revision_fkey", "knowledge_documents", "knowledge_document_revisions",
                          ["id", "active_revision_id"], ["document_id", "id"], deferrable=True, initially="DEFERRED")
    op.create_table("knowledge_chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("revision_id", sa.Uuid(), sa.ForeignKey("knowledge_document_revisions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False), sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start", sa.Integer(), nullable=False), sa.Column("end", sa.Integer(), nullable=False),
        sa.Column("context", postgresql.JSONB(), nullable=False), sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024)), sa.Column("sparse_embedding", SPARSEVEC(250002)), *timestamps(),
        sa.UniqueConstraint("revision_id", "chunk_index"))
    op.create_index("knowledge_chunks_revision_id_idx", "knowledge_chunks", ["revision_id"])
    op.create_index("knowledge_chunks_embedding_idx", "knowledge_chunks", ["embedding"], postgresql_using="hnsw",
                    postgresql_ops={"embedding": "vector_cosine_ops"})


def downgrade() -> None:
    op.drop_table("knowledge_chunks")
    op.drop_constraint("knowledge_documents_active_revision_fkey", "knowledge_documents", type_="foreignkey")
    op.drop_table("knowledge_document_revisions")
    op.drop_table("knowledge_documents")
    op.drop_column("articles", "ingestion_payload")
