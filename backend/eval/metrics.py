"""检索指标（纯函数）。gold=相关文章 id 集合，retrieved=按相关性排序的文章 id 列表。"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence, Set


def recall_at_k[T: Hashable](retrieved: Sequence[T], gold: Set[T], k: int) -> float:
    """前 k 个命中里覆盖了多少 gold。gold 为空返回 0。"""
    if not gold:
        return 0.0
    hit = len(set(retrieved[:k]) & gold)
    return hit / len(gold)


def mrr[T: Hashable](retrieved: Sequence[T], gold: Set[T]) -> float:
    """第一个相关命中的倒数排名（1/rank）；没命中返回 0。"""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k[T: Hashable](retrieved: Sequence[T], gold: Set[T], k: int) -> float:
    """二元相关性的 nDCG@k。DCG = Σ rel_i / log2(i+1)，IDCG 为理想排序。"""
    dcg = sum(1.0 / math.log2(i + 1) for i, doc_id in enumerate(retrieved[:k], start=1) if doc_id in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
