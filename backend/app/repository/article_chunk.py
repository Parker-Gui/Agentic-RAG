"""Article chunk repository: embedding writes and similarity search via pgvector.

Vectors live in Postgres, so retrieval is one JOIN query to ``articles`` (no per-hit
round trip) and writes share the caller's transaction. Repository never commits.
"""

from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.article import Article
from app.db.models.article_chunk import ArticleChunk


async def add(db: AsyncSession, *, rows: list[dict[str, Any]]) -> int:
    """Insert a batch of chunks; returns the number of rows written.

    Each row needs article_id / chunk_index / text / group_type / embedding.
    """
    if not rows:
        return 0
    db.add_all([ArticleChunk(**row) for row in rows])
    await db.flush()
    return len(rows)


async def delete_by_article(db: AsyncSession, *, article_id: int) -> int:
    """Drop every chunk of one article; returns how many were removed.

    Called before a re-index so replaying an article cannot duplicate its chunks.
    """
    result = cast(
        "CursorResult[Any]",
        await db.execute(delete(ArticleChunk).where(ArticleChunk.article_id == article_id)),
    )
    await db.flush()
    return result.rowcount


async def count_by_article(db: AsyncSession, *, article_id: int) -> int:
    """How many chunks an article currently has indexed."""
    result = await db.execute(
        select(func.count()).select_from(ArticleChunk).where(ArticleChunk.article_id == article_id)
    )
    return result.scalar_one()


async def search(
    db: AsyncSession,
    *,
    embedding: list[float],
    top_k: int = 5,
    group_type: str | None = None,
) -> list[dict[str, Any]]:
    """Nearest chunks to ``embedding`` (dense), joined to their article metadata.

    Ordered by cosine distance (``<=>``), matching the HNSW index. ``score`` =
    ``1 - cosine_distance``（pgvector 余弦距离 ∈ [0,2]，故 score ∈ [-1,1]，越大越相关）。
    """
    distance = ArticleChunk.embedding.cosine_distance(embedding)
    stmt = (
        select(
            ArticleChunk.article_id,
            ArticleChunk.chunk_index,
            ArticleChunk.text.label("chunk_text"),
            ArticleChunk.group_type,
            Article.title,
            Article.url,
            Article.site_name,
            Article.publish_date,
            (1 - distance).label("score"),
        )
        .join(Article, Article.id == ArticleChunk.article_id)
        .order_by(distance)
        .limit(top_k)
    )
    if group_type is not None:
        stmt = stmt.where(ArticleChunk.group_type == group_type)

    result = await db.execute(stmt)
    return [
        {
            "article_id": row.article_id,
            "chunk_index": row.chunk_index,
            "chunk_text": row.chunk_text,
            "group_type": row.group_type,
            "title": row.title,
            "url": row.url,
            "site_name": row.site_name,
            "publish_date": str(row.publish_date) if row.publish_date else None,
            "score": float(row.score),
        }
        for row in result.all()
    ]


async def search_sparse(
    db: AsyncSession,
    *,
    sparse: dict[int, float],
    top_k: int = 5,
    group_type: str | None = None,
) -> list[dict[str, Any]]:
    """Phase2：稀疏向量检索，与 search（稠密）对称，返回同结构 dict，供 retrieval.rrf_fuse 融合。

    TODO(impl)：
      - 需先在 ArticleChunk 加 sparse_embedding(sparsevec) 列并迁移（见模型文件）。
      - 用 sparsevec 的内积算子（如 max_inner_product）排序取 top_k，score 用内积值。
      - select / 返回结构与 search 完全一致（article_id/chunk_index/chunk_text/publish_date/...）。
    """
    from pgvector import SparseVector

    from app.db.models.article_chunk import SPARSE_DIM

    query_sv = SparseVector(sparse, SPARSE_DIM) if isinstance(sparse, dict) else sparse
    distance = ArticleChunk.sparse_embedding.max_inner_product(query_sv)  # <#> = 负内积
    stmt = (
        select(
            ArticleChunk.article_id,
            ArticleChunk.chunk_index,
            ArticleChunk.text.label("chunk_text"),
            ArticleChunk.group_type,
            Article.title,
            Article.url,
            Article.site_name,
            Article.publish_date,
            (-distance).label("score"),  # 内积，越大越相关
        )
        .join(Article, Article.id == ArticleChunk.article_id)
        .where(ArticleChunk.sparse_embedding.is_not(None))  # 未回填稀疏向量的旧块跳过
        .order_by(distance)
        .limit(top_k)
    )
    if group_type is not None:
        stmt = stmt.where(ArticleChunk.group_type == group_type)

    result = await db.execute(stmt)
    return [
        {
            "article_id": row.article_id,
            "chunk_index": row.chunk_index,
            "chunk_text": row.chunk_text,
            "group_type": row.group_type,
            "title": row.title,
            "url": row.url,
            "site_name": row.site_name,
            "publish_date": str(row.publish_date) if row.publish_date else None,
            "score": float(row.score),
        }
        for row in result.all()
    ]
