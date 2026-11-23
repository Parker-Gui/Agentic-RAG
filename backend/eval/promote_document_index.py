"""One-time atomic promotion of validated news revisions; never copies test files.

Run after schema migration. Keeps articles and article_chunks byte-for-byte.
Refuses a nonempty destination document index rather than merging silently.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Table, create_engine, func, select, update
from sqlalchemy.engine import make_url

from app.core.config import settings
from app.db.models.article import Article
from app.db.models.article_chunk import ArticleChunk
from app.db.models.knowledge import KnowledgeChunk, KnowledgeDocument, KnowledgeRevision
from app.services.rag.ingestion import processing_config

VALIDATION_DB = "intelligence_multiformat_v1_validation"
ROOT = Path(".agent_data/multiformat-v1").resolve()


def promote() -> dict[str, Any]:
    if settings.POSTGRES_DB == VALIDATION_DB:
        raise SystemExit("Destination must be the project DB, not validation DB")
    before = json.loads((ROOT / "before.json").read_text(encoding="utf-8"))
    backup = ROOT / before["backup"]
    if hashlib.sha256(backup.read_bytes()).hexdigest() != before["backup_sha256"]:
        raise SystemExit("Pre-migration backup integrity check failed")
    validation = json.loads((ROOT / "integration.json").read_text(encoding="utf-8"))
    assert validation["checks"]["legacy_article_mapping"] == 20
    url = make_url(settings.DATABASE_URL_SYNC)
    source = create_engine(url.set(database=VALIDATION_DB))
    destination = create_engine(url)
    try:
        with source.connect() as origin:
            documents = [
                dict(row)
                for row in origin.execute(
                    select(KnowledgeDocument.__table__).where(KnowledgeDocument.article_id.is_not(None))
                ).mappings()
            ]
            revisions = [
                dict(row)
                for row in origin.execute(
                    select(KnowledgeRevision.__table__).where(
                        KnowledgeRevision.id.in_([d["active_revision_id"] for d in documents])
                    )
                ).mappings()
            ]
            chunks = [
                dict(row)
                for row in origin.execute(
                    select(KnowledgeChunk.__table__).where(KnowledgeChunk.revision_id.in_([r["id"] for r in revisions]))
                ).mappings()
            ]
        assert len(documents) == len(revisions) == before["articles"] == 20
        assert all(rev["state"] == "ready" and rev["config"] == processing_config() for rev in revisions)
        revisions_by_doc = {r["document_id"]: r for r in revisions}
        for revision in revisions:
            assert hashlib.sha256(revision["input_bytes"]).hexdigest() == revision["input_hash"]
            assert hashlib.sha256(revision["canonical_text"].encode()).hexdigest() == revision["body_hash"]
            revision_chunks = [c for c in chunks if c["revision_id"] == revision["id"]]
            assert revision_chunks and all(c["embedding"] is not None for c in revision_chunks)
            for chunk in revision_chunks:
                assert chunk["text"] == revision["canonical_text"][chunk["start"] : chunk["end"]]
                chunk["context"].setdefault("context_ranges", [])
        with destination.begin() as connection:
            assert connection.scalar(select(func.count()).select_from(KnowledgeDocument)) == 0
            articles = {row.id: row for row in connection.execute(select(Article.__table__))}
            assert len(articles) == 20
            assert connection.scalar(select(func.count()).select_from(ArticleChunk)) == before["chunks"] == 212
            for document in documents:
                article = articles[document["article_id"]]
                revision = revisions_by_doc[document["id"]]
                assert revision["format"] == "text" and revision["canonical_text"] == (article.body_text or "").strip()
                assert revision["title"] == article.title and revision["url"] == article.url
                assert revision["publish_date"] == article.publish_date and revision["group_type"] == article.group_type
            connection.execute(
                cast(Table, KnowledgeDocument.__table__).insert(),
                [{**d, "active_revision_id": None} for d in documents],
            )
            connection.execute(cast(Table, KnowledgeRevision.__table__).insert(), revisions)
            # Allocate destination sequence values; source chunk row IDs are not external identities.
            connection.execute(
                cast(Table, KnowledgeChunk.__table__).insert(),
                [{k: v for k, v in c.items() if k != "id"} for c in chunks],
            )
            for document in documents:
                connection.execute(
                    update(KnowledgeDocument)
                    .where(KnowledgeDocument.id == document["id"])
                    .values(active_revision_id=document["active_revision_id"])
                )
        return {
            "news_documents": len(documents),
            "active_chunks": len(chunks),
            "legacy_chunks_retained": 212,
            "synthetic_files_copied": 0,
            "activation": "one transaction",
            "index_source_changed": False,
        }
    finally:
        source.dispose()
        destination.dispose()


def main() -> None:
    report = promote()
    (ROOT / "promotion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
