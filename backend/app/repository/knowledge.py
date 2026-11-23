"""Unified knowledge persistence. Flush here; transaction ownership stays outside."""

from typing import Any
from uuid import UUID, uuid4

from pgvector import SparseVector
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models.article_chunk import SPARSE_DIM
from app.db.models.knowledge import KnowledgeChunk, KnowledgeDocument, KnowledgeRevision


async def get_document(db: AsyncSession, *, document_id: UUID) -> KnowledgeDocument | None:
    return await db.get(KnowledgeDocument, document_id)


async def get_by_article(db: AsyncSession, *, article_id: int) -> KnowledgeDocument | None:
    return await db.scalar(select(KnowledgeDocument).where(KnowledgeDocument.article_id == article_id))


async def receive(
    db: AsyncSession, *, source_key: str, source_kind: str, article_id: int | None, revision: dict[str, Any]
) -> KnowledgeRevision:
    await db.execute(
        pg_insert(KnowledgeDocument)
        .values(
            id=uuid4(),
            source_key=source_key,
            source_kind=source_kind,
            article_id=article_id,
        )
        .on_conflict_do_nothing(index_elements=[KnowledgeDocument.source_key])
    )
    document = await db.scalar(select(KnowledgeDocument).where(KnowledgeDocument.source_key == source_key))
    assert document is not None
    await db.execute(
        pg_insert(KnowledgeRevision)
        .values(
            id=uuid4(),
            document_id=document.id,
            state="received",
            blocks=[],
            warnings=[],
            **revision,
        )
        .on_conflict_do_nothing(index_elements=[KnowledgeRevision.document_id, KnowledgeRevision.processing_signature])
    )
    row = await db.scalar(
        select(KnowledgeRevision).where(
            KnowledgeRevision.document_id == document.id,
            KnowledgeRevision.processing_signature == revision["processing_signature"],
        )
    )
    assert row is not None
    await db.flush()
    await db.refresh(row)
    return row


async def get_revision(db: AsyncSession, *, revision_id: UUID) -> KnowledgeRevision | None:
    return await db.get(KnowledgeRevision, revision_id)


async def latest_revision(db: AsyncSession, *, document_id: UUID) -> KnowledgeRevision | None:
    return await db.scalar(
        select(KnowledgeRevision)
        .where(KnowledgeRevision.document_id == document_id)
        .order_by(KnowledgeRevision.created_at.desc(), KnowledgeRevision.id.desc())
        .limit(1)
    )


async def update_revision(db: AsyncSession, *, revision_id: UUID, values: dict[str, Any]) -> KnowledgeRevision:
    row = await db.get(KnowledgeRevision, revision_id)
    assert row is not None
    for key, value in values.items():
        setattr(row, key, value)
    await db.flush()
    await db.refresh(row)
    return row


async def replace_chunks(db: AsyncSession, *, revision_id: UUID, chunks: list[dict[str, Any]]) -> None:
    await db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.revision_id == revision_id))
    for chunk in chunks:
        context = {
            k: chunk[k]
            for k in ("kind", "heading_path", "locator", "parent_start", "parent_end", "split_reason", "context_ranges")
        }
        db.add(
            KnowledgeChunk(
                revision_id=revision_id,
                chunk_index=chunk["index"],
                text=chunk["text"],
                start=chunk["start"],
                end=chunk["end"],
                embedding_text=chunk["embedding_text"],
                context=context,
            )
        )
    await db.flush()


async def list_chunks(db: AsyncSession, *, revision_id: UUID) -> list[KnowledgeChunk]:
    result = await db.scalars(
        select(KnowledgeChunk).where(KnowledgeChunk.revision_id == revision_id).order_by(KnowledgeChunk.chunk_index)
    )
    return list(result.all())


async def activate(db: AsyncSession, *, revision_id: UUID, encoded: list[dict[str, Any]]) -> KnowledgeRevision:
    row = await db.get(KnowledgeRevision, revision_id)
    assert row is not None
    chunks = await list_chunks(db, revision_id=revision_id)
    if not chunks or len(encoded) != len(chunks):
        raise ValueError("Cannot activate an incomplete vector set")
    for chunk, vector in zip(chunks, encoded, strict=True):
        chunk.embedding = vector["dense"]
        chunk.sparse_embedding = (
            SparseVector(vector["sparse"], SPARSE_DIM) if vector.get("sparse") is not None else None
        )
    document = await db.get(KnowledgeDocument, row.document_id)
    assert document is not None
    document.active_revision_id = row.id
    row.state, row.failed_stage, row.error_code, row.error_message = "ready", None, None, None
    await db.flush()
    await db.refresh(row)
    return row


async def list_article_documents(db: AsyncSession) -> list[KnowledgeDocument]:
    result = await db.scalars(select(KnowledgeDocument).where(KnowledgeDocument.article_id.is_not(None)))
    return list(result.all())


async def search(
    db: AsyncSession,
    *,
    embedding: list[float],
    top_k: int = 5,
    group_type: str | None = None,
    sparse: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    if sparse is None:
        distance = KnowledgeChunk.embedding.cosine_distance(embedding)
        score = 1 - distance
    else:
        distance = KnowledgeChunk.sparse_embedding.max_inner_product(SparseVector(sparse, SPARSE_DIM))
        score = -distance
    latest = aliased(KnowledgeRevision)
    update_state = (
        select(latest.state)
        .where(latest.document_id == KnowledgeDocument.id)
        .order_by(latest.created_at.desc(), latest.id.desc())
        .limit(1)
        .correlate(KnowledgeDocument)
        .scalar_subquery()
    )
    stmt = (
        select(
            KnowledgeChunk.chunk_index,
            KnowledgeChunk.text.label("chunk_text"),
            KnowledgeChunk.start,
            KnowledgeChunk.end,
            KnowledgeChunk.context,
            KnowledgeRevision.document_id,
            KnowledgeRevision.id.label("revision_id"),
            KnowledgeRevision.title,
            KnowledgeRevision.url,
            KnowledgeRevision.site_name,
            KnowledgeRevision.publish_date,
            KnowledgeRevision.format,
            KnowledgeRevision.body_hash,
            KnowledgeDocument.article_id,
            score.label("score"),
            update_state.label("update_state"),
        )
        .select_from(KnowledgeChunk)
        .join(KnowledgeRevision, KnowledgeRevision.id == KnowledgeChunk.revision_id)
        .join(
            KnowledgeDocument,
            (KnowledgeDocument.id == KnowledgeRevision.document_id)
            & (KnowledgeDocument.active_revision_id == KnowledgeRevision.id),
        )
        .where(KnowledgeRevision.state == "ready", KnowledgeChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    if sparse is not None:
        stmt = stmt.where(KnowledgeChunk.sparse_embedding.is_not(None))
    if group_type is not None:
        stmt = stmt.where(KnowledgeRevision.group_type == group_type)
    records = (await db.execute(stmt)).mappings().all()
    return [
        {
            **{key: value for key, value in record.items() if key != "context"},
            "document_id": str(record["document_id"]),
            "revision_id": str(record["revision_id"]),
            "score": float(record["score"]),
            **record["context"],
        }
        for record in records
    ]
