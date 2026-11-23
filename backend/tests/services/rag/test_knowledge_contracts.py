"""File source identity must survive retrieval, reading and reviewed export."""

import json
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage

from app.agents.agent_tools.rag_tool import _hit_to_record
from app.agents.evidence import from_tool_messages
from app.core.exceptions import NotFoundError, ValidationError
from app.schemas.rag import RagSource
from app.services.artifact import render_brief
from app.services.rag import ingestion
from app.services.rag.document_types import DocumentError
from app.services.rag.embedding import Embedder
from app.services.rag.retrieval import rrf_fuse
from eval.retrieval_eval import _retrieved_article_ids


def test_file_evidence_is_versioned_and_exports_without_a_fake_url() -> None:
    doc_id, revision_id = str(uuid4()), str(uuid4())
    payload = {
        "results": [
            {
                "document_id": doc_id,
                "revision_id": revision_id,
                "title": "Policy.docx",
                "format": "docx",
                "content": "x" * 2300,
                "start": 40,
                "end": 2340,
                "locator": {"block": 3},
                "body_hash": "body",
            }
        ]
    }
    evidence, errors = from_tool_messages(
        [ToolMessage(content=json.dumps(payload), tool_call_id="read", name="read_article_tool")], "S1"
    )
    assert not errors and len(evidence) == 1
    item = evidence[0]
    assert item.document_id == doc_id and item.url is None and item.article_id is None
    assert item.end == 2040 and item.truncated
    changed = json.loads(json.dumps(payload))
    changed["results"][0]["revision_id"] = str(uuid4())
    other, _ = from_tool_messages([ToolMessage(content=json.dumps(changed), tool_call_id="read")], "S1")
    assert item.id != other[0].id
    brief = render_brief(
        {
            "question": "Policy?",
            "answer": f"Fact [{item.id}]",
            "termination": "completed",
            "review": {"approved": True},
            "evidence": [item.model_dump()],
        },
        task_id=uuid4(),
        created_at=datetime.now(UTC),
    )
    assert doc_id in brief and revision_id in brief and "Policy.docx" in brief
    assert "article_id=None" not in brief


def test_file_rrf_keys_do_not_collide_on_null_article_id() -> None:
    a = {"article_id": None, "revision_id": str(uuid4()), "chunk_index": 0, "chunk_text": "A"}
    b = {"article_id": None, "revision_id": str(uuid4()), "chunk_index": 0, "chunk_text": "B"}
    assert len(rrf_fuse([a, b], [b, a])) == 2


def test_document_id_evaluation_does_not_merge_files_into_one_null_article() -> None:
    assert _retrieved_article_ids(
        [
            {"document_id": "A", "article_id": None},
            {"document_id": "B", "article_id": None},
            {"article_id": 3, "document_id": "C"},
        ]
    ) == ["document:A", "document:B", 3]


def test_real_database_date_serializes_in_tool_output() -> None:
    record = _hit_to_record({"publish_date": date(2026, 11, 23), "chunk_text": "Fact"})
    assert json.loads(json.dumps(record))["published_at"] == "2026-11-23"


def test_query_schema_accepts_document_source_without_news_identity() -> None:
    result = RagSource(document_id=str(uuid4()), revision_id=str(uuid4()), chunk_text="Fact", score=0.8)
    assert result.article_id is None


@pytest.mark.anyio
async def test_read_refuses_another_document_revision_and_wrong_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    document_id, revision_id = uuid4(), uuid4()
    document = SimpleNamespace(active_revision_id=revision_id, article_id=None)
    revision = SimpleNamespace(id=revision_id, document_id=uuid4(), state="ready", body_hash="correct")

    @asynccontextmanager
    async def db_context():
        yield AsyncMock()

    monkeypatch.setattr(ingestion, "get_worker_db_context", db_context)
    monkeypatch.setattr(ingestion.knowledge_repo, "get_document", AsyncMock(return_value=document))
    monkeypatch.setattr(ingestion.knowledge_repo, "get_revision", AsyncMock(return_value=revision))
    with pytest.raises(NotFoundError):
        await ingestion.KnowledgeService().read(document_id, revision_id=revision_id)
    revision.document_id = document_id
    with pytest.raises(ValidationError, match="version"):
        await ingestion.KnowledgeService().read(document_id, expected_hash="wrong")


def test_flagembedding_rejects_actual_token_overflow_before_model_call() -> None:
    embedder = object.__new__(Embedder)
    model_call = Mock()
    embedder._model = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [1] * 513), encode=model_call)
    with pytest.raises(ValueError, match="silent truncation"):
        embedder.encode(["short string with unexpectedly many model tokens"])
    model_call.assert_not_called()


@pytest.mark.anyio
async def test_resume_refuses_to_activate_a_superseded_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    old_id, active_id, document_id = uuid4(), uuid4(), uuid4()
    old = SimpleNamespace(
        id=old_id,
        document_id=document_id,
        state="failed",
        config=ingestion.processing_config(),
        created_at=datetime.now(UTC) - timedelta(days=1),
    )
    active = SimpleNamespace(id=active_id, created_at=datetime.now(UTC))

    @asynccontextmanager
    async def db_context():
        yield AsyncMock()

    monkeypatch.setattr(ingestion, "get_worker_db_context", db_context)
    monkeypatch.setattr(ingestion.knowledge_repo, "get_revision", AsyncMock(side_effect=[old, active]))
    monkeypatch.setattr(
        ingestion.knowledge_repo, "get_document", AsyncMock(return_value=SimpleNamespace(active_revision_id=active_id))
    )
    with pytest.raises(ValidationError) as exc:
        await ingestion.KnowledgeService()._run(old_id)
    assert exc.value.code == "DOCUMENT_SUPERSEDED"


@pytest.mark.anyio
async def test_article_sync_counts_new_work_and_rejects_inactive_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    article = SimpleNamespace(
        id=1,
        ingestion_payload=None,
        body_text="Body",
        title="Title",
        url="https://example.org",
        site_name="Example",
        publish_date=None,
        group_type="competitor",
        embedding_done=True,
    )

    @asynccontextmanager
    async def db_context():
        yield AsyncMock()

    monkeypatch.setattr(ingestion, "get_worker_db_context", db_context)
    monkeypatch.setattr(ingestion.article_repo, "list_for_knowledge", AsyncMock(return_value=[article]))
    service = ingestion.KnowledgeService()
    service.import_bytes = AsyncMock(return_value={"state": "ready", "active": True, "reused": False})
    assert await service.sync_articles() == 1
    service.import_bytes.return_value = {"state": "ready", "active": True, "reused": True}
    assert await service.sync_articles() == 0
    service.import_bytes.return_value = {
        "state": "ready",
        "active": False,
        "reused": True,
        "error_code": None,
        "error": None,
    }
    with pytest.raises(DocumentError):
        await service.sync_articles()
