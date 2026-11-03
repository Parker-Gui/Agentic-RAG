"""检索质量评测：对每个问题跑 query_fn，比对 gold 文章，汇总 recall@k/MRR/nDCG。

query_fn 注入不同 RAG 配置（dense-only / +hybrid / +rerank），即可对比出提升。
纯聚合逻辑，query_fn 可 mock，便于单测。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from eval import metrics

# query_fn(question, k) -> list[hit dict]（含 article_id）
QueryFn = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


def _retrieved_article_ids(hits: list[dict[str, Any]]) -> list[int]:
    """命中里的 article_id，去重保序（同一文章多个块只算一次，且保留首次名次）。"""
    seen: set[int] = set()
    ordered: list[int] = []
    for h in hits:
        aid = h["article_id"]
        if aid not in seen:
            seen.add(aid)
            ordered.append(aid)
    return ordered


async def evaluate_retrieval(dataset: list[dict[str, Any]], *, query_fn: QueryFn, k: int = 5) -> dict[str, Any]:
    """对 dataset 每条 {question, expected_article_ids} 跑 query_fn，返回平均指标。"""
    if not dataset:
        return {"recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0, "n": 0, "k": k}

    tot_recall = tot_mrr = tot_ndcg = 0.0
    for row in dataset:
        hits = await query_fn(row["question"], k)
        retrieved = _retrieved_article_ids(hits)
        gold = set(row["expected_article_ids"])
        tot_recall += metrics.recall_at_k(retrieved, gold, k)
        tot_mrr += metrics.mrr(retrieved, gold)
        tot_ndcg += metrics.ndcg_at_k(retrieved, gold, k)

    n = len(dataset)
    return {"recall@k": tot_recall / n, "mrr": tot_mrr / n, "ndcg@k": tot_ndcg / n, "n": n, "k": k}
