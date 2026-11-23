"""Serial, resumable imports with short durable stages and atomic activation."""

import asyncio
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.db.models.article_chunk import EMBED_DIM
from app.db.models.knowledge import KnowledgeRevision
from app.db.session import get_worker_db_context, knowledge_source_lock
from app.repository import article as article_repo
from app.repository import knowledge as knowledge_repo
from app.services.rag.chunking import CHUNKING_VERSION, chunk_document
from app.services.rag.document_types import PARSER_VERSION, ContentBlock, DocumentError, DocumentFormat, ParsedDocument
from app.services.rag.embedding import EncodedChunk, get_embedder_async
from app.services.rag.parsers import MAX_FILE_BYTES, parse_isolated


def processing_config() -> dict[str, Any]:
    return {
        "parser": PARSER_VERSION,
        "chunker": CHUNKING_VERSION,
        "chunk_size": settings.RAG_CHUNK_SIZE,
        "overlap": settings.RAG_CHUNK_OVERLAP,
        "budget": "characters",
        "embed_provider": settings.RAG_EMBED_PROVIDER,
        "embed_model": settings.RAG_OLLAMA_EMBED_MODEL
        if settings.RAG_EMBED_PROVIDER == "ollama"
        else settings.RAG_EMBED_MODEL,
        "embed_endpoint": settings.RAG_OLLAMA_BASE_URL if settings.RAG_EMBED_PROVIDER == "ollama" else None,
    }


def revision_status(row: KnowledgeRevision, *, active_revision_id: UUID | None) -> dict[str, Any]:
    return {
        "document_id": str(row.document_id),
        "revision_id": str(row.id),
        "state": row.state,
        "active_revision_id": str(active_revision_id) if active_revision_id else None,
        "active": row.id == active_revision_id,
        "failed_stage": row.failed_stage,
        "error_code": row.error_code,
        "error": row.error_message,
        "warnings": row.warnings,
        "format": row.format,
        "title": row.title,
        "input_hash": row.input_hash,
        "body_hash": row.body_hash,
        "config": row.config,
    }


class KnowledgeService:
    """Public ingestion/read facade; raw paths are accepted only by the local CLI."""

    async def import_file(self, path: Path, *, source_key: str, group_type: str = "internal") -> dict[str, Any]:
        suffix = path.suffix.lower().lstrip(".")
        format = "html" if suffix in {"html", "htm"} else suffix
        if format not in {"html", "md", "docx", "pdf"}:
            raise ValidationError(message="Supported formats: HTML, MD, DOCX, text PDF", code="DOCUMENT_UNSUPPORTED")

        # Read at most limit+1 bytes rather than loading an arbitrarily large input.
        def read() -> bytes:
            with path.open("rb") as stream:
                return stream.read(MAX_FILE_BYTES + 1)

        data = await asyncio.to_thread(read)
        return await self.import_bytes(
            data,
            format=cast(DocumentFormat, format),
            title=path.name,
            source_key=f"file:{source_key}",
            group_type=group_type,
        )

    async def import_bytes(
        self,
        data: bytes,
        *,
        format: DocumentFormat,
        title: str,
        source_key: str,
        group_type: str = "internal",
        article_id: int | None = None,
        url: str | None = None,
        site_name: str | None = None,
        publish_date: date | None = None,
    ) -> dict[str, Any]:
        if not source_key.strip() or len(source_key) > 500:
            raise ValidationError(message="source_key must contain 1..500 characters")
        if len(data) > MAX_FILE_BYTES:
            raise ValidationError(message="Input exceeds 20 MiB", code="DOCUMENT_SIZE_LIMIT")
        config = processing_config()
        input_hash = hashlib.sha256(data).hexdigest()
        metadata = {
            "format": format,
            "title": title,
            "group_type": group_type,
            "url": url,
            "site_name": site_name,
            "publish_date": str(publish_date) if publish_date else None,
        }
        signature = hashlib.sha256(
            json.dumps({"input": input_hash, "config": config, **metadata}, sort_keys=True).encode()
        ).hexdigest()
        async with knowledge_source_lock(source_key):
            async with get_worker_db_context() as db:
                row = await knowledge_repo.receive(
                    db,
                    source_key=source_key,
                    source_kind="article" if article_id else "file",
                    article_id=article_id,
                    revision={
                        "processing_signature": signature,
                        "input_hash": input_hash,
                        "input_bytes": data,
                        "format": format,
                        "title": title,
                        "url": url,
                        "site_name": site_name,
                        "publish_date": publish_date,
                        "group_type": group_type,
                        "config": config,
                    },
                )
                revision_id = row.id
                reused = row.state == "ready"
            result = await self._run(revision_id)
            result["reused"] = reused
            return result

    async def status(self, document_id: UUID, *, revision_id: UUID | None = None) -> dict[str, Any]:
        async with get_worker_db_context() as db:
            document = await knowledge_repo.get_document(db, document_id=document_id)
            if document is None:
                raise NotFoundError(message="Document not found")
            row = (
                await knowledge_repo.get_revision(db, revision_id=revision_id)
                if revision_id
                else await knowledge_repo.latest_revision(db, document_id=document_id)
            )
            if row is None or row.document_id != document_id:
                raise NotFoundError(message="Document revision not found")
            return revision_status(row, active_revision_id=document.active_revision_id)

    async def resume(self, revision_id: UUID) -> dict[str, Any]:
        async with get_worker_db_context() as db:
            row = await knowledge_repo.get_revision(db, revision_id=revision_id)
            if row is None:
                raise NotFoundError(message="Revision not found")
            document = await knowledge_repo.get_document(db, document_id=row.document_id)
            assert document is not None
            source_key = document.source_key
        async with knowledge_source_lock(source_key):
            return await self._run(revision_id)

    async def _update(self, revision_id: UUID, **values: Any) -> None:
        async with get_worker_db_context() as db:
            await knowledge_repo.update_revision(db, revision_id=revision_id, values=values)

    async def _run(self, revision_id: UUID) -> dict[str, Any]:
        async with get_worker_db_context() as db:
            row = await knowledge_repo.get_revision(db, revision_id=revision_id)
            assert row is not None
        document_id = row.document_id
        if row.state == "ready":
            # Reimporting an old successful input is idempotent, not an implicit rollback.
            return await self.status(document_id, revision_id=revision_id)
        if row.config != processing_config():
            raise ValidationError(
                message="Processing configuration changed; reimport creates a new revision",
                code="DOCUMENT_CONFIG_CHANGED",
            )
        async with get_worker_db_context() as db:
            document = await knowledge_repo.get_document(db, document_id=document_id)
            active = (
                await knowledge_repo.get_revision(db, revision_id=document.active_revision_id)
                if document and document.active_revision_id
                else None
            )
        if active is not None and active.id != row.id and active.created_at > row.created_at:
            raise ValidationError(
                message="This revision was superseded; import current input instead of resuming an older revision",
                code="DOCUMENT_SUPERSEDED",
            )
        stage = "parsing"
        try:
            if row.canonical_text is None:
                await self._update(revision_id, state="parsing", error_code=None, error_message=None, failed_stage=None)
                parsed = await asyncio.to_thread(
                    parse_isolated, row.input_bytes, cast(DocumentFormat, row.format), title=row.title
                )
                await self._update(
                    revision_id,
                    state="parsed",
                    canonical_text=parsed.text,
                    body_hash=hashlib.sha256(parsed.text.encode()).hexdigest(),
                    blocks=[b.model_dump(mode="json") for b in parsed.blocks],
                    warnings=parsed.warnings,
                )
            else:
                if hashlib.sha256(row.canonical_text.encode()).hexdigest() != row.body_hash:
                    raise DocumentError("body_hash_mismatch", "Saved parser output failed its integrity check")
                parsed = ParsedDocument(
                    title=row.title,
                    format=cast(DocumentFormat, row.format),
                    text=row.canonical_text,
                    blocks=[ContentBlock.model_validate(b) for b in row.blocks],
                    warnings=row.warnings,
                )
            stage = "chunking"
            async with get_worker_db_context() as db:
                chunks = await knowledge_repo.list_chunks(db, revision_id=revision_id)
            if not chunks:
                await self._update(revision_id, state="chunking")
                pieces = chunk_document(parsed, chunk_size=row.config["chunk_size"], overlap=row.config["overlap"])
                if not pieces:
                    raise DocumentError("empty_chunks", "No indexable chunks", stage=stage)
                async with get_worker_db_context() as db:
                    await knowledge_repo.replace_chunks(
                        db, revision_id=revision_id, chunks=[piece.model_dump(mode="json") for piece in pieces]
                    )
                    await knowledge_repo.update_revision(db, revision_id=revision_id, values={"state": "chunked"})
                    chunks = await knowledge_repo.list_chunks(db, revision_id=revision_id)
            stage = "embedding"
            await self._update(revision_id, state="embedding")
            model = await get_embedder_async()
            texts = [chunk.embedding_text for chunk in chunks]
            encoded: list[EncodedChunk]
            if model.supports_sparse:
                encoded = await asyncio.to_thread(model.encode_with_sparse, texts)
            else:
                vectors = await asyncio.to_thread(model.encode, texts)
                encoded = [{"dense": vector, "sparse": None} for vector in vectors]
            if len(encoded) != len(chunks) or any(
                len(v["dense"]) != EMBED_DIM or not all(math.isfinite(x) for x in v["dense"]) for v in encoded
            ):
                raise ValueError("Embedding output is incomplete or invalid")
            async with get_worker_db_context() as db:
                await knowledge_repo.activate(db, revision_id=revision_id, encoded=cast(list[dict[str, Any]], encoded))
        except Exception as exc:
            code = exc.code if isinstance(exc, DocumentError) else f"{stage}_failed"
            state = code if code in {"needs_ocr", "unsupported"} else "failed"
            # Persist a safe summary; original exception details may contain file paths or endpoint data.
            message = str(exc) if isinstance(exc, DocumentError) else f"{stage} failed ({type(exc).__name__})"
            await self._update(revision_id, state=state, failed_stage=stage, error_code=code, error_message=message)
        return await self.status(document_id, revision_id=revision_id)

    async def read(
        self,
        document_id: UUID,
        *,
        revision_id: UUID | None = None,
        start: int = 0,
        max_chars: int = 2000,
        expected_hash: str | None = None,
    ) -> dict[str, Any]:
        if start < 0 or not 1 <= max_chars <= 20000:
            raise ValidationError(message="Invalid document reading range")
        async with get_worker_db_context() as db:
            document = await knowledge_repo.get_document(db, document_id=document_id)
            if document is None:
                raise NotFoundError(message="Document not found")
            chosen = revision_id or document.active_revision_id
            row = await knowledge_repo.get_revision(db, revision_id=chosen) if chosen else None
            if row is None or row.document_id != document_id or row.state != "ready":
                raise NotFoundError(message="Readable document revision not found")
            if expected_hash is not None and expected_hash != row.body_hash:
                raise ValidationError(
                    message="Document version does not match source offsets", code="DOCUMENT_HASH_MISMATCH"
                )
            body = row.canonical_text or ""
            if start > len(body):
                raise ValidationError(message="Reading offset exceeds document length")
            end = min(len(body), start + max_chars)
            latest = await knowledge_repo.latest_revision(db, document_id=document_id)
            ranges = [
                {"start": b["start"], "end": b["end"], **b["locator"], "heading_path": b["heading_path"]}
                for b in row.blocks
                if b["start"] < end and b["end"] > start
            ]
            return {
                "document_id": str(document_id),
                "revision_id": str(row.id),
                "article_id": document.article_id,
                "title": row.title,
                "url": row.url,
                "published_at": str(row.publish_date) if row.publish_date else None,
                "format": row.format,
                "content": body[start:end],
                "start": start,
                "end": end,
                "truncated": end < len(body),
                "next_start": end if end < len(body) else None,
                "locator": {"ranges": ranges},
                "body_hash": row.body_hash,
                "source_kind": "internal_kb",
                "active": row.id == document.active_revision_id,
                "update_state": latest.state if latest else None,
                "warnings": row.warnings,
            }

    async def sync_articles(self, *, batch_limit: int = 100) -> int:
        async with get_worker_db_context() as db:
            articles = await article_repo.list_for_knowledge(db)
        processed = 0
        for article in articles:
            payload = article.ingestion_payload or {}
            data = str(payload.get("input") or article.body_text or "").encode()
            result = await self.import_bytes(
                data,
                format=cast(DocumentFormat, payload.get("format", "text")),
                source_key=f"article:{article.id}",
                title=article.title,
                article_id=article.id,
                url=article.url,
                site_name=article.site_name,
                publish_date=article.publish_date,
                group_type=article.group_type,
            )
            if result["state"] != "ready" or not result["active"]:
                raise DocumentError(
                    result["error_code"] or "inactive_revision", result["error"] or "Article revision is not active"
                )
            if not article.embedding_done:
                async with get_worker_db_context() as db:
                    await article_repo.mark_embedding_done(db, article.id)
            if not result["reused"]:
                processed += 1
            if processed >= batch_limit:
                break
        return processed
