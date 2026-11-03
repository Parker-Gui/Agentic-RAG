"""add sparse_embedding to article_chunks (hybrid retrieval)

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-08-27

给 article_chunks 增加 BGE-m3 稀疏向量列 + 稀疏 HNSW 索引，支撑 hybrid 检索。
应用后需重跑索引（python -m app.services.rag）回填已有块的 sparse_embedding。
"""

import sqlalchemy as sa
from pgvector.sqlalchemy import SPARSEVEC

from alembic import op

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None

SPARSE_DIM = 250002  # 与 app.db.models.article_chunk.SPARSE_DIM 一致


def upgrade() -> None:
    op.add_column(
        "article_chunks",
        sa.Column("sparse_embedding", SPARSEVEC(SPARSE_DIM), nullable=True),
    )
    # 稀疏 HNSW（内积算子）。若 pgvector 版本不支持稀疏 HNSW，可去掉本索引，
    # 稀疏检索仍能跑（顺序扫描，慢一些）。
    op.create_index(
        "article_chunks_sparse_embedding_idx",
        "article_chunks",
        ["sparse_embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"sparse_embedding": "sparsevec_ip_ops"},
    )


def downgrade() -> None:
    op.drop_index("article_chunks_sparse_embedding_idx", table_name="article_chunks")
    op.drop_column("article_chunks", "sparse_embedding")
