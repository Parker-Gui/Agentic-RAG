"""Add article_chunks table with pgvector embeddings

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-08-24

Replaces the former external Milvus collection: chunk text and its BGE-m3 dense
vector now live in Postgres next to ``articles``, so a search is one JOIN and a
write shares the article's transaction.

Requires the ``vector`` extension, which ships in the ``pgvector/pgvector``
image (the plain ``postgres`` image does not have it). ``CREATE EXTENSION``
needs superuser or ``rds_superuser``; the compose ``POSTGRES_USER`` is superuser,
managed Postgres usually allows it from the owner role.
"""

import pgvector.sqlalchemy
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None

# Keep in sync with app.db.models.article_chunk.EMBED_DIM -- pgvector columns are
# fixed-width, so a model/column mismatch is a write-time error, not a warning.
EMBED_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "article_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("article_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("group_type", sa.String(length=50), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBED_DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="article_chunks_pkey"),
        sa.ForeignKeyConstraint(
            ["article_id"],
            ["articles.id"],
            name="article_chunks_article_id_fkey",
            ondelete="CASCADE",
        ),
        # Makes re-indexing idempotent: the worker deletes an article's chunks
        # before writing the new ones, and this catches it if it ever forgets.
        sa.UniqueConstraint(
            "article_id",
            "chunk_index",
            name="article_chunks_article_id_chunk_index_key",
        ),
    )
    op.create_index("article_chunks_article_id_idx", "article_chunks", ["article_id"])
    op.create_index("article_chunks_group_type_idx", "article_chunks", ["group_type"])

    # HNSW + cosine, matching ArticleChunk.embedding.cosine_distance() in the repo.
    # The ops class is not optional: an index built with a different distance
    # function is simply ignored by the planner for `<=>` ordering.
    op.create_index(
        "article_chunks_embedding_idx",
        "article_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("article_chunks_embedding_idx", table_name="article_chunks")
    op.drop_index("article_chunks_group_type_idx", table_name="article_chunks")
    op.drop_index("article_chunks_article_id_idx", table_name="article_chunks")
    op.drop_table("article_chunks")
    # The `vector` extension is left installed on purpose -- dropping it would
    # cascade into any other vector column added since.
