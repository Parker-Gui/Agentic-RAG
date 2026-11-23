"""检索结果融合与过滤（纯函数，无 I/O）。

Phase 2 引入 hybrid 检索：稠密（语义）+ 稀疏（专名/词面）两路各自召回，
在这里用 RRF 融合成一个排序，再按最低分阈值裁掉不相关的。放纯函数模块便于单测。

约定：一条命中（Hit）就是 RagService/chunk_repo 产出的 dict，至少含
    {"article_id": int, "chunk_index": int, "chunk_text": str, "score": float, ...}
融合用 (article_id, chunk_index) 作为去重/对齐的稳定键。
"""

from __future__ import annotations

from typing import Any

Hit = dict[str, Any]


def _key(hit: Hit) -> tuple[str | int, int]:
    """一条 chunk 的稳定标识：同一 chunk 在稠密/稀疏两路里要能对齐。"""
    identity = f"revision:{hit['revision_id']}" if hit.get("revision_id") else hit["article_id"]
    return (identity, hit["chunk_index"])


def rrf_fuse(dense_hits: list[Hit], sparse_hits: list[Hit], *, k: int = 60, top_k: int | None = None) -> list[Hit]:
    """Reciprocal Rank Fusion：按排名而非原始分融合两路结果。

    score = Σ_over_lists 1 / (k + rank)   # rank 从 1 开始
    - 只看名次、不看各路分数量纲，天然规避"稠密余弦"和"稀疏内积"不可比的问题。
    - k(默认 60，经验值)越大，越弱化头部名次的优势、结果越平滑。

    TODO(impl)：
      1. 对 dense_hits / sparse_hits 分别按各自出现顺序赋 rank（假设入参已按分排好）。
      2. 以 _key(hit) 聚合两路得分：fused[key] += 1/(k+rank)。
      3. 命中记录取两路中信息更全的一条（通常稠密那条带 chunk_text）。
      4. 按 fused 分降序；top_k 截断（None 表示不截）。
      5. 把 fused 分写回 hit["score"]（覆盖原始分，下游 rerank/阈值统一口径）。
    """
    fused: dict[tuple[str | int, int], float] = {}
    record: dict[tuple[str | int, int], Hit] = {}
    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            key = _key(hit)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            # 保留信息更全的一条：优先带 chunk_text 的（通常来自稠密路，稀疏路可能只有元信息）
            if key not in record or (not record[key].get("chunk_text") and hit.get("chunk_text")):
                record[key] = hit
    ordered = sorted(fused, key=lambda key: fused[key], reverse=True)
    result: list[Hit] = [{**record[key], "score": fused[key]} for key in ordered]
    return result[:top_k] if top_k is not None else result


def apply_min_score(hits: list[Hit], threshold: float) -> list[Hit]:
    """丢弃 score 低于阈值的命中，支持"库内无相关 → 返回空"的 no-answer 语义。

    注意：阈值口径取决于上游给的是什么分——
      - 仅稠密：score = 1 - cosine_distance ∈ [-1,1]
      - RRF 融合后：是 RRF 分（很小的正数，量纲完全不同）
      - rerank 后：交叉编码器分（通常 sigmoid 到 [0,1]）
    所以阈值必须和"当前处于流水线哪一段"配套，别用一个魔法常数跨段套用。
    """
    return [h for h in hits if h.get("score", 0.0) >= threshold]
