"""retrieval.rrf_fuse / apply_min_score 纯函数测试（无 I/O，普通同步测试）。"""

from app.services.rag.retrieval import apply_min_score, rrf_fuse


def _h(aid: int, ci: int, **kw) -> dict:
    return {"article_id": aid, "chunk_index": ci, **kw}


def test_rrf_fuse_orders_by_reciprocal_rank():
    # A: dense#1 + sparse#2；C: sparse#1；B: dense#2 → A > C > B
    dense = [_h(1, 0, chunk_text="A"), _h(2, 0, chunk_text="B")]
    sparse = [_h(3, 0, chunk_text="C"), _h(1, 0)]
    out = rrf_fuse(dense, sparse, k=60)
    assert [(h["article_id"], h["chunk_index"]) for h in out] == [(1, 0), (3, 0), (2, 0)]


def test_rrf_fuse_keeps_record_with_chunk_text():
    # 同一 chunk 两路都命中，保留带正文的那条（来自 dense 路），而非稀疏路的元信息条
    out = rrf_fuse([_h(1, 0, chunk_text="full")], [_h(1, 0)])
    assert out[0]["chunk_text"] == "full"


def test_rrf_fuse_writes_fused_score():
    out = rrf_fuse([_h(1, 0)], [_h(1, 0)], k=60)
    assert abs(out[0]["score"] - (1 / 61 + 1 / 61)) < 1e-9


def test_rrf_fuse_top_k_truncates():
    dense = [_h(1, 0), _h(2, 0), _h(3, 0)]
    assert len(rrf_fuse(dense, [], top_k=2)) == 2


def test_apply_min_score_filters_below_threshold():
    hits = [_h(1, 0, score=0.5), _h(2, 0, score=0.1)]
    assert [h["article_id"] for h in apply_min_score(hits, 0.3)] == [1]
