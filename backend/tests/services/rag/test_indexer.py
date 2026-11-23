"""Indexing stores dense-only output as NULL sparse data, retaining the FlagEmbedding branch."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.rag import indexer


@pytest.mark.anyio
@pytest.mark.parametrize("supports_sparse", [True, False])
async def test_indexer_selects_supported_encoding_path(monkeypatch: pytest.MonkeyPatch, supports_sparse: bool) -> None:
    monkeypatch.setattr(indexer.settings, "RAG_INDEX_SOURCE", "legacy")
    model = SimpleNamespace(
        supports_sparse=supports_sparse,
        encode=Mock(return_value=[[1.0] * 1024]),
        encode_with_sparse=Mock(return_value=[{"dense": [1.0] * 1024, "sparse": {3: 0.5}}]),
    )
    db = AsyncMock()

    @asynccontextmanager
    async def context():
        yield db

    monkeypatch.setattr(indexer, "get_worker_db_context", context)
    monkeypatch.setattr(indexer, "get_embedder_async", AsyncMock(return_value=model))
    monkeypatch.setattr(indexer, "chunk_text", lambda *args, **kwargs: ["text chunk"])
    monkeypatch.setattr(
        indexer.article_repo,
        "list_pending_embedding",
        AsyncMock(return_value=[SimpleNamespace(id=1, body_text="text", group_type="regulatory")]),
    )
    mark = AsyncMock()
    add = AsyncMock(return_value=1)
    monkeypatch.setattr(indexer.article_repo, "mark_embedding_done", mark)
    monkeypatch.setattr(indexer.chunk_repo, "delete_by_article", AsyncMock())
    monkeypatch.setattr(indexer.chunk_repo, "add", add)
    assert await indexer.run_indexing() == 1
    assert add.await_args is not None
    row = add.await_args.kwargs["rows"][0]
    assert (row["sparse_embedding"] is not None) == supports_sparse
    if supports_sparse:
        model.encode.assert_not_called()
    else:
        model.encode_with_sparse.assert_not_called()
    mark.assert_awaited_once_with(db, 1)
