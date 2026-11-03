"""Reranker.rerank 逻辑测试（mock 模型，不下载 reranker）。

绕过 __init__（它会加载模型）直接注入假 _model，只验证排序/写分/截断逻辑。
"""

from unittest.mock import MagicMock

from app.services.rag.rerank import Reranker


def _reranker(scores) -> Reranker:
    r = Reranker.__new__(Reranker)  # 跳过 __init__，不加载真模型
    r._model = MagicMock()
    r._model.compute_score.return_value = scores
    return r


def test_rerank_orders_by_score_desc_and_truncates():
    r = _reranker([0.1, 0.9, 0.5])
    cands = [{"chunk_text": "a"}, {"chunk_text": "b"}, {"chunk_text": "c"}]
    out = r.rerank("q", cands, top_k=2)
    assert [h["chunk_text"] for h in out] == ["b", "c"]   # 0.9, 0.5
    assert out[0]["score"] == 0.9                          # 校准分写回


def test_rerank_empty_candidates():
    r = _reranker([])
    assert r.rerank("q", [], 5) == []


def test_rerank_single_pair_scalar_score():
    r = _reranker(0.7)  # 单对时 FlagReranker 返回标量，rerank 要能兜住
    out = r.rerank("q", [{"chunk_text": "x"}], top_k=5)
    assert out[0]["score"] == 0.7
