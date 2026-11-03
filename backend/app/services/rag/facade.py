"""RAG 门面：RagService —— 路由/agent 工具唯一可见的入口。

检索：问题 → 编码 → pgvector 检索（→ Phase2: hybrid + rerank + 阈值）。
向量与文章元信息同在 Postgres，一条 JOIN 查询就拿全 title/url/site_name/publish_date。

嵌入是同步阻塞调用，异步里用 asyncio.to_thread 甩线程池；DB 检索原生异步，直接 await。
模型的**加载**同样阻塞（import torch + 读几 GB 权重），所以构造函数不碰模型：
在 query 里用 get_embedder_async() 拿单例，事件循环上一步都不阻塞。
"""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repository import article_chunk as chunk_repo
from app.services.rag.embedding import get_embedder_async


class RagService:
    """RAG 检索用例。路由/agent 工具通过它拿结果，只调用 query。"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def query(self, question: str, top_k: int = 5, group_type: str | None = None) -> list[dict[str, Any]]:
        """检索与问题最相关的文章块，返回带出处的结果列表。

        当前：纯稠密检索（可用）。Phase2 打开开关后改走 _hybrid_retrieve。
        """
        # 1. 编码（加载与编码都阻塞 → 线程池；传函数与参数，不是调用结果）
        embedder = await get_embedder_async()
        vectors = await asyncio.to_thread(embedder.encode, [question])
        query_vec = vectors[0]

        # 2. 检索：Phase2 = hybrid+rerank+阈值；当前 = 稠密 JOIN
        if settings.RAG_HYBRID_ENABLED or settings.RAG_RERANK_ENABLED:
            hits = await self._hybrid_retrieve(question, query_vec, top_k=top_k, group_type=group_type)
        else:
            hits = await chunk_repo.search(self.db, embedding=query_vec, top_k=top_k, group_type=group_type)

        return [
            {
                "article_id": hit["article_id"],
                "title": hit["title"],
                "url": hit["url"],
                "site_name": hit["site_name"],
                "publish_date": hit.get("publish_date"),  # 供工具映射成 Evidence.published_at
                "chunk_text": hit["chunk_text"],
                "score": hit["score"],
            }
            for hit in hits
        ]

    async def _hybrid_retrieve(
        self, question: str, query_vec: list[float], *, top_k: int, group_type: str | None
    ) -> list[dict[str, Any]]:
        """Phase2 检索流水线：hybrid 召回 → rerank → 阈值。

        TODO(impl)：
            from app.services.rag import retrieval, rerank
            # 1. over-fetch：稠密 + 稀疏各召回 RAG_FETCH_K 条
            dense = await chunk_repo.search(self.db, embedding=query_vec, top_k=settings.RAG_FETCH_K, group_type=group_type)
            if settings.RAG_HYBRID_ENABLED:
                sparse_vec = await asyncio.to_thread(self.embedder.encode_with_sparse, [question])  # 取 sparse
                sparse = await chunk_repo.search_sparse(self.db, sparse=sparse_vec[0], top_k=settings.RAG_FETCH_K, group_type=group_type)
                cand = retrieval.rrf_fuse(dense, sparse, top_k=settings.RAG_FETCH_K)
            else:
                cand = dense
            # 2. rerank（已定采用）
            if settings.RAG_RERANK_ENABLED:
                cand = rerank.get_reranker().rerank(question, cand, top_k)
            else:
                cand = cand[:top_k]
            # 3. 阈值过滤（no-answer）
            return retrieval.apply_min_score(cand, settings.RAG_MIN_SCORE)
        """
        from app.services.rag import retrieval

        # 1. 稠密召回：over-fetch RAG_FETCH_K 条，给 rerank 留候选
        cand = await chunk_repo.search(self.db, embedding=query_vec, top_k=settings.RAG_FETCH_K, group_type=group_type)

        # 2. hybrid：稀疏召回 + RRF 融合（TODO 优化：query() 里一次性编码 dense+sparse，避免这里重复前向）
        if settings.RAG_HYBRID_ENABLED:
            embedder = await get_embedder_async()
            encoded = await asyncio.to_thread(embedder.encode_with_sparse, [question])
            sparse = await chunk_repo.search_sparse(
                self.db, sparse=encoded[0]["sparse"], top_k=settings.RAG_FETCH_K, group_type=group_type
            )
            cand = retrieval.rrf_fuse(cand, sparse, top_k=settings.RAG_FETCH_K)

        # 3. rerank：交叉编码器精排（加载 + 打分都阻塞 → 线程池）
        if settings.RAG_RERANK_ENABLED:
            from app.services.rag.rerank import get_reranker_async

            reranker = await get_reranker_async()
            cand = await asyncio.to_thread(reranker.rerank, question, cand, top_k)
        else:
            cand = cand[:top_k]

        # 4. 阈值过滤（no-answer）
        return retrieval.apply_min_score(cand, settings.RAG_MIN_SCORE)
