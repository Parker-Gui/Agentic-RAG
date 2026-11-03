"""交叉编码器重排（rerank）——检索质量的最后一道提纯。

召回阶段（稠密/稀疏/RRF）用的是向量点积，快但粗；rerank 用 cross-encoder 把
(query, chunk_text) 成对喂模型，直接输出校准的相关性分（normalize=True → sigmoid 到 0~1）。
精度高但每个候选一次前向，所以只对召回的 top-N（over-fetch）做。

模型：bge-reranker（settings.RAG_RERANK_MODEL）。懒加载单例，走 ModelScope 下载（国内友好）。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

Hit = dict[str, Any]


@lru_cache(maxsize=1)
def _load_reranker() -> Reranker:
    return Reranker()


# 同 embedding：lru_cache 不加锁，避免并发首调重复加载权重。
_load_lock = threading.Lock()


def get_reranker() -> Reranker:
    """懒加载单例：首次调用时才载入 reranker 模型（阻塞，含 import torch）。

    协程里用 ``get_reranker_async``，别在事件循环上直接调这个。
    """
    with _load_lock:
        return _load_reranker()


async def get_reranker_async() -> Reranker:
    """协程里取 reranker：阻塞的首次加载走线程池。"""
    return await asyncio.to_thread(get_reranker)


class Reranker:
    def __init__(self) -> None:
        from FlagEmbedding import FlagReranker

        from app.core.config import settings
        from app.services.rag.model_store import resolve_model_dir

        model_dir = resolve_model_dir(settings.RAG_RERANK_MODEL)
        self._model = FlagReranker(model_dir, use_fp16=False)  # CPU
        logger.info("reranker 加载完成: %s", model_dir)

    def rerank(self, query: str, candidates: list[Hit], top_k: int) -> list[Hit]:
        """对候选按 (query, chunk_text) 交叉编码打分并重排，返回前 top_k（分写回 score）。"""
        if not candidates:
            return []
        pairs = [(query, c.get("chunk_text") or "") for c in candidates]
        scores = self._model.compute_score(pairs, normalize=True)  # normalize → [0,1]
        if not isinstance(scores, list):  # 单对时 FlagReranker 返回标量
            scores = [scores]
        ranked = sorted(
            ({**c, "score": float(s)} for c, s in zip(candidates, scores, strict=True)),
            key=lambda h: h["score"],
            reverse=True,
        )
        return ranked[:top_k]
