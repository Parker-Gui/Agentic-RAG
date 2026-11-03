"""BGE-m3 文本嵌入封装。

把「文本 → 向量」收敛到一个接口后面，隐藏 FlagEmbedding / BGE-m3 的细节。
模型加载慢（首次下载约 2.3GB 权重），通过 ``get_embedder`` 惰性单例保证全进程只加载一次。

**加载本身是阻塞的**：``import FlagEmbedding`` 会连带 import torch（torch 内部又会
碰 ``os.getcwd()`` / 读文件），再从磁盘读几 GB 权重。所以事件循环里一律走
``get_embedder_async()``——否则 langgraph dev 的 blockbuster 会直接抛
``BlockingError: Blocking call to os.getcwd``。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from functools import lru_cache
from typing import Any

import numpy as np

from app.core.config import settings
from app.db.models.article_chunk import EMBED_DIM

logger = logging.getLogger(__name__)

__all__ = ["EMBED_DIM", "Embedder", "get_embedder", "get_embedder_async"]


class Embedder:
    """BGE-m3 嵌入模型封装。构造即加载模型（慢），故应只实例化一次。"""

    def __init__(self, model_name: str = settings.RAG_EMBED_MODEL) -> None:
        from FlagEmbedding import BGEM3FlagModel

        from app.services.rag.model_store import resolve_model_dir

        model_dir = resolve_model_dir(model_name)
        logger.info("正在加载 BGE-m3 模型: %s", model_dir)
        self._model = BGEM3FlagModel(model_dir, use_fp16=False)
        logger.info("BGE-m3 模型加载完成。")

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """把一批文本编码成 dense 向量；空输入返回 []。"""
        if not texts:
            return []
        output = self._model.encode(texts, batch_size=batch_size, max_length=512)
        dense = np.asarray(output["dense_vecs"])
        return dense.tolist()

    def encode_with_sparse(self, texts: list[str], batch_size: int = 16) -> list[dict[str, Any]]:
        """Phase2：同时取 dense 与 sparse（BGE-m3 lexical_weights），供 hybrid 检索。

        BGE-m3 一次前向即可出三种表示，这里额外保留 sparse。返回每条一个
        {"dense": list[float], "sparse": {token_id(int): weight(float)}}。

        TODO(impl)：
            output = self._model.encode(texts, batch_size=batch_size, max_length=512,
                                        return_dense=True, return_sparse=True)
            dense = np.asarray(output["dense_vecs"]).tolist()
            sparse = output["lexical_weights"]   # list[dict[token_id, weight]]
            return [{"dense": d, "sparse": s} for d, s in zip(dense, sparse, strict=True)]
        """
        if not texts:
            return []
        output = self._model.encode(texts, batch_size=batch_size, max_length=512, return_dense=True, return_sparse=True)
        dense = np.asarray(output["dense_vecs"]).tolist()
        # lexical_weights: list[dict[token_id, weight]]；token id 键统一转 int（SparseVector 要 int 键）
        sparse = [{int(tok): float(w) for tok, w in lw.items()} for lw in output["lexical_weights"]]
        return [{"dense": d, "sparse": s} for d, s in zip(dense, sparse, strict=True)]


@lru_cache(maxsize=1)
def _load_embedder() -> Embedder:
    return Embedder()


# lru_cache 自己不加锁：两个线程同时首调会各加载一份权重（各 2.3GB）。
_load_lock = threading.Lock()


def get_embedder() -> Embedder:
    """惰性单例：首次调用时加载模型（阻塞数十秒），之后复用同一实例。

    同步调用点（CLI、线程池内）用这个；协程里用 ``get_embedder_async``。
    """
    with _load_lock:
        return _load_embedder()


async def get_embedder_async() -> Embedder:
    """协程里取嵌入器：把阻塞的首次加载甩到线程池，不占事件循环。

    已加载时只是一次 lru_cache 命中，``to_thread`` 的开销可以忽略。
    """
    return await asyncio.to_thread(get_embedder)
