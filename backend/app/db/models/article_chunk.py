"""Article chunk model: one embedded slice of an article's body text.

The embedding is a pgvector column in the same database as ``articles``, so a
retrieval is one joined query and a chunk can never outlive its article (the
foreign key cascades). Replaces the previous external Milvus collection.
"""

from typing import Any

from pgvector.sqlalchemy import SPARSEVEC, Vector
from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin

# BGE-m3 dense 向量维度。这是唯一出处（embedding.py 从这里导入）；
# 改这个值必须同时写一个迁移去改列类型，否则写入会被 Postgres 拒绝。
EMBED_DIM = 1024
# BGE-m3 稀疏(lexical)向量维度 = 分词器词表大小(XLM-R)。sparsevec 定长维度，改需迁移。
SPARSE_DIM = 250002


class ArticleChunk(Base, TimestampMixin):
    """One chunk of an article's body text plus its embedding."""

    __tablename__ = "article_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Position of this chunk inside the article, 0-based.
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Denormalised from articles so a search can filter without the join.
    group_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM), nullable=False)

    # Phase 2: 稀疏向量（BGE-m3 lexical_weights），hybrid 检索用。
    # nullable：迁移加列后，旧块在重索引前该列为空；重索引会回填。
    sparse_embedding: Mapped[Any] = mapped_column(SPARSEVEC(SPARSE_DIM), nullable=True)

    __table_args__ = (
        # Re-indexing an article must not be able to duplicate its chunks.
        UniqueConstraint(
            "article_id",
            "chunk_index",
            name="article_chunks_article_id_chunk_index_key",
        ),
        # HNSW over cosine distance — matches the `<=>` operator used in search().
        Index(
            "article_chunks_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # 稀疏向量 HNSW，内积算子，配 search_sparse 的 max_inner_product(<#>)。
        Index(
            "article_chunks_sparse_embedding_idx",
            "sparse_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"sparse_embedding": "sparsevec_ip_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<ArticleChunk(id={self.id}, article_id={self.article_id}, chunk_index={self.chunk_index})>"
