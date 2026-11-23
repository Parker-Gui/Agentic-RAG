"""Compare retained legacy chunks and unified documents on the frozen 22 seed cases.

Local embeddings only. No sampling, label edits, generation or automatic tuning.
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sqlalchemy import func, select

from app.core.config import settings
from app.db.models.article import Article
from app.db.models.article_chunk import ArticleChunk
from app.db.models.knowledge import KnowledgeChunk, KnowledgeDocument, KnowledgeRevision
from app.db.session import get_worker_db_context, run_compatible_async
from app.services.rag import RagService, facade
from app.services.rag.embedding import get_embedder_async
from eval.retrieval_eval import _retrieved_article_ids, evaluate_retrieval
from eval.validate_documents import local_embeddings_only

ROOT = Path(".agent_data/multiformat-v1").resolve()
FROZEN = Path(".agent_data/retrieval-baseline-20261123").resolve()


async def compare() -> dict[str, Any]:
    corpus = json.loads((FROZEN / "corpus.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (FROZEN / "dataset.seed.jsonl").read_text(encoding="utf-8").splitlines()]
    async with get_worker_db_context() as db:
        articles = list((await db.scalars(select(Article).order_by(Article.id))).all())
        expected = {source["article_id"]: source for source in corpus["sources"]}
        assert len(articles) == len(expected) == 20
        for article in articles:
            assert hashlib.sha256((article.body_text or "").encode()).hexdigest() == expected[article.id]["body_sha256"]
            assert article.title == expected[article.id]["title"] and article.url == expected[article.id]["url"]
        assert await db.scalar(select(func.count()).select_from(ArticleChunk)) == 212
        documents = list((await db.scalars(select(KnowledgeDocument))).all())
        # Synthetic integration fixtures must not enter this same-corpus comparison.
        assert len(documents) == 20 and all(document.article_id in expected for document in documents)
        assert all(document.active_revision_id is not None for document in documents)
        new_chunks = await db.scalar(
            select(func.count())
            .select_from(KnowledgeChunk)
            .join(KnowledgeRevision)
            .join(KnowledgeDocument, KnowledgeDocument.active_revision_id == KnowledgeRevision.id)
        )
    model = await get_embedder_async()

    class CachedEmbedder:
        supports_sparse = False

        def __init__(self) -> None:
            self.cache: dict[str, list[float]] = {}

        def encode(self, texts: list[str]) -> list[list[float]]:
            for question in texts:
                if question not in self.cache:
                    self.cache[question] = model.encode([question])[0]
            return [self.cache[question] for question in texts]

    cached = CachedEmbedder()

    async def load_cached() -> CachedEmbedder:
        return cached

    results: dict[str, Any] = {}
    per_query: dict[str, Any] = {}
    original_source = settings.RAG_INDEX_SOURCE
    try:
        with patch.object(facade, "get_embedder_async", load_cached):
            for source in ["legacy", "documents"]:
                settings.RAG_INDEX_SOURCE = source
                source_scores: dict[str, Any] = {}
                hits_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

                async def query(
                    question: str, k: int, *, cache: dict[tuple[str, int], list[dict[str, Any]]] = hits_cache
                ) -> list[dict[str, Any]]:
                    key = (question, k)
                    if key not in cache:
                        async with get_worker_db_context() as db:
                            cache[key] = await RagService(db).query(question, top_k=k)
                    return cache[key]

                for k in [5, 10]:
                    scores = await evaluate_retrieval(cases, query_fn=query, k=k)
                    distinct_count = 0
                    for row in cases:
                        distinct_count += len(_retrieved_article_ids(await query(row["question"], k)))
                    scores["mean_distinct_articles"] = distinct_count / len(cases)
                    scores["by_kind"] = {
                        kind: await evaluate_retrieval(
                            [row for row in cases if row["kind"] == kind], query_fn=query, k=k
                        )
                        for kind in ["single", "compare"]
                    }
                    source_scores[str(k)] = scores
                results[source] = source_scores
                per_query[source] = [
                    {
                        "case_id": row["case_id"],
                        "question": row["question"],
                        "expected_article_ids": row["expected_article_ids"],
                        "top5": await query(row["question"], 5),
                        "top10": await query(row["question"], 10),
                    }
                    for row in cases
                ]
                sys.stdout.write(
                    json.dumps({"index_source": source, "scores": source_scores}, ensure_ascii=False) + "\n"
                )
                sys.stdout.flush()
    finally:
        settings.RAG_INDEX_SOURCE = original_source
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "corpus_version": corpus["version"],
        "articles": 20,
        "legacy_chunks": 212,
        "document_chunks": new_chunks,
        "cases": len(cases),
        "scores": results,
        "per_query": per_query,
        "label_status": "existing source-checked seeds; not an independent blind test",
        "scope": "combined structure-char-v1 and title-prefix implementation; not a single-variable chunk-size experiment",
    }


def main() -> None:
    if settings.RAG_EMBED_PROVIDER != "ollama" or settings.RAG_HYBRID_ENABLED or settings.RAG_RERANK_ENABLED:
        raise SystemExit("Require local Ollama, dense-only, no reranker")
    events: list[dict[str, Any]] = []
    with local_embeddings_only(events):
        report = run_compatible_async(compare())
    report.update({"http": events, "chat_calls": 0})
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "retrieval-comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
