"""嵌入 worker：把未向量化的文章切块、编码、写入 article_chunks。

粘合层——只负责编排 chunking / embedding / repo 三个组件，不含新的业务逻辑。
供 CLI（cron 定时 / 手动触发）调用。

向量与 embedding_done 标志都在 Postgres 里，同一个事务提交：不会再出现
「向量写进去了但文章没标记」或反过来的漂移（Milvus 时代的固有风险）。
"""

from __future__ import annotations

import asyncio
import logging

from pgvector import SparseVector

from app.core.config import settings
from app.db.models.article_chunk import SPARSE_DIM
from app.db.session import get_worker_db_context
from app.repository import article as article_repo
from app.repository import article_chunk as chunk_repo
from app.services.rag.chunking import chunk_text
from app.services.rag.embedding import get_embedder_async

logger = logging.getLogger(__name__)


async def run_indexing(batch_limit: int = settings.RAG_INDEX_BATCH) -> int:
    """处理一批未向量化的文章，返回成功处理的文章数。

    每篇文章：切块 → 编码 → 清旧块 → 写新块 → 标记 embedding_done。
    """
    # 模型加载与编码都是阻塞调用：走线程池，别占事件循环。
    embedder = await get_embedder_async()

    processed = 0
    async with get_worker_db_context() as db:
        articles = await article_repo.list_pending_embedding(db, limit=batch_limit)
        logger.info("待向量化文章: %d 篇", len(articles))

        for article in articles:
            chunks = chunk_text(
                article.body_text or "",
                chunk_size=settings.RAG_CHUNK_SIZE,
                overlap=settings.RAG_CHUNK_OVERLAP,
            )
            if not chunks:
                await article_repo.mark_embedding_done(db, article.id)
                continue
            # 每条 {"dense", "sparse"}
            encoded = await asyncio.to_thread(embedder.encode_with_sparse, chunks)
            # 重跑安全：先清掉这篇文章的旧块，避免 (article_id, chunk_index) 撞唯一约束。
            await chunk_repo.delete_by_article(db, article_id=article.id)
            await chunk_repo.add(
                db,
                rows=[
                    {
                        "article_id": article.id,
                        "chunk_index": i,
                        "text": chunk,
                        "group_type": article.group_type,
                        "embedding": enc["dense"],
                        "sparse_embedding": SparseVector(enc["sparse"], SPARSE_DIM),
                    }
                    for i, (chunk, enc) in enumerate(zip(chunks, encoded, strict=True))
                ],
            )
            await article_repo.mark_embedding_done(db, article.id)
            processed += 1

    logger.info("本次索引完成，处理 %d 篇", processed)
    return processed
