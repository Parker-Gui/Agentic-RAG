"""Explicit local integration checks; only runs in the dedicated restored test DB.

Usage: POSTGRES_DB=intelligence_multiformat_v1_validation uv run python -m eval.validate_documents
No chat calls. Synthetic documents remain in this isolated DB for inspection.
"""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pymupdf
from docx import Document
from langchain_core.messages import ToolMessage
from sqlalchemy import delete, func, select

from app.agents.agent_tools.rag_tool import read_article_tool, search_rag_tool
from app.agents.evidence import from_tool_messages
from app.core.config import settings
from app.core.exceptions import AlreadyExistsError, ValidationError
from app.db.models.article import Article
from app.db.models.knowledge import KnowledgeChunk
from app.db.session import get_worker_db_context, knowledge_source_lock, run_compatible_async
from app.repository import article as article_repo
from app.repository import knowledge as knowledge_repo
from app.services.artifact import render_brief
from app.services.crawler.common.models import Candidate
from app.services.crawler.delivery.persistence import save_hits_to_db
from app.services.crawler.fetch.detail import parse_article
from app.services.crawler.orchestration.pipeline import _article_hit
from app.services.rag import KnowledgeService, RagService, ingestion

ROOT = Path(".agent_data/multiformat-v1").resolve()
RUN_KEY = os.environ.get("DOCUMENT_VALIDATION_RUN") or uuid4().hex


@contextmanager
def local_embeddings_only(events: list[dict[str, Any]]):
    original = httpx.Client.send

    def guarded(client: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        if request.url.host not in {"127.0.0.1", "localhost", "::1"} or request.url.path not in {
            "/api/embed",
            "/api/tags",
        }:
            raise AssertionError("Only local embeddings and model metadata are authorized")
        response = original(client, request, **kwargs)
        events.append({"path": request.url.path, "status": response.status_code})
        return response

    with patch.object(httpx.Client, "send", guarded):
        yield


def fixtures() -> dict[str, Path]:
    folder = ROOT / "fixtures"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "travel.html").write_text(
        "<article><h1>Travel Policy HTML</h1><p>HTML allowance is 610 yuan.</p></article>", encoding="utf-8"
    )
    (folder / "travel.md").write_text("# Travel Policy Markdown\n\nMarkdown allowance is 620 yuan.\n", encoding="utf-8")
    word = Document()
    word.add_heading("Travel Policy Word", 1)
    word.add_paragraph("Word allowance is 630 yuan.")
    table = word.add_table(rows=2, cols=2)
    for cell, value in zip(
        [c for row in table.rows for c in row.cells], ["City", "Limit", "Shanghai", "630"], strict=True
    ):
        cell.text = value
    word.save(str(folder / "travel.docx"))
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 50), "PDF travel allowance is 640 yuan.")
        pdf.save(folder / "travel.pdf")
    return {suffix: folder / f"travel.{suffix}" for suffix in ["html", "md", "docx", "pdf"]}


async def crash_child() -> None:
    original = ingestion.KnowledgeService._update

    async def save_then_crash(self: KnowledgeService, revision_id: UUID, **values: Any) -> None:
        await original(self, revision_id, **values)
        if values.get("state") == "parsed":
            (ROOT / "crash-receipt.json").write_text(json.dumps({"revision_id": str(revision_id)}), encoding="utf-8")
            os._exit(73)

    with patch.object(ingestion.KnowledgeService, "_update", save_then_crash):
        await KnowledgeService().import_bytes(
            b"# Crash recovery\n\nThis saved parser output must survive process exit.",
            format="md",
            title="Crash recovery",
            source_key=f"file:validation/{RUN_KEY}/crash",
        )


async def validate() -> dict[str, Any]:
    service = KnowledgeService()
    checks: dict[str, Any] = {}
    fixture_paths = fixtures()
    imported: dict[str, dict[str, Any]] = {}
    for suffix, path in fixture_paths.items():
        row = await service.import_file(path, source_key=f"validation/{RUN_KEY}/{suffix}")
        assert row["state"] == "ready", row
        imported[suffix] = row
        read = await service.read(UUID(row["document_id"]), revision_id=UUID(row["revision_id"]))
        assert "allowance" in read["content"] and read["locator"]["ranges"]
        payload = json.loads(
            await read_article_tool.ainvoke({"document_id": row["document_id"], "revision_id": row["revision_id"]})
        )
        evidence, errors = from_tool_messages(
            [ToolMessage(content=json.dumps(payload), tool_call_id="read", name="read_article_tool")], "S1"
        )
        assert len(evidence) == 1 and not errors and evidence[0].document_id == row["document_id"]
        brief = render_brief(
            {
                "question": "Travel?",
                "answer": f"Policy [{evidence[0].id}]",
                "termination": "completed",
                "review": {"approved": True},
                "evidence": [evidence[0].model_dump()],
            },
            task_id=uuid4(),
            created_at=datetime.now(UTC),
        )
        assert row["document_id"] in brief
        sys.stdout.write(json.dumps({"format": suffix, "state": "ready", "evidence": True}) + "\n")
        sys.stdout.flush()
    checks["formats_read_evidence_brief"] = list(imported)
    duplicate = await service.import_file(fixture_paths["md"], source_key=f"validation/{RUN_KEY}/md")
    assert duplicate["revision_id"] == imported["md"]["revision_id"]
    checks["idempotent_reimport"] = True

    class FailedEmbedder:
        supports_sparse = False

        def encode(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("intentional local embedding failure")

    async def failure() -> FailedEmbedder:
        return FailedEmbedder()

    with patch.object(ingestion, "get_embedder_async", failure):
        failed = await service.import_bytes(
            b"# New policy\n\nUpdated markdown allowance is 720 yuan.",
            format="md",
            title="travel.md",
            source_key=f"file:validation/{RUN_KEY}/md",
        )
    assert failed["state"] == "failed" and failed["failed_stage"] == "embedding"
    assert failed["active_revision_id"] == duplicate["revision_id"]
    old_read = await service.read(UUID(duplicate["document_id"]))
    assert "620" in old_read["content"]
    with patch.object(ingestion, "parse_isolated", side_effect=AssertionError("Resume must reuse parsed output")):
        recovered = await service.resume(UUID(failed["revision_id"]))
    assert recovered["state"] == "ready" and recovered["active"]
    assert "720" in (await service.read(UUID(recovered["document_id"])))["content"]
    assert (
        "620"
        in (await service.read(UUID(duplicate["document_id"]), revision_id=UUID(duplicate["revision_id"])))["content"]
    )
    checks["failed_embedding_keeps_old_revision_then_resumes"] = True
    try:
        await service.read(UUID(recovered["document_id"]), expected_hash="not-this-version")
        raise AssertionError("Hash mismatch accepted")
    except ValidationError:
        checks["hash_mismatch_rejected"] = True

    async with knowledge_source_lock(f"file:validation/{RUN_KEY}/md"):
        try:
            await service.import_file(fixture_paths["md"], source_key=f"validation/{RUN_KEY}/md")
            raise AssertionError("Concurrent writer accepted")
        except AlreadyExistsError:
            checks["concurrent_writer_rejected"] = True

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 8, 8), False)
        pixmap.clear_with(255)
        page.insert_image(page.rect, pixmap=pixmap)
        image_pdf = pdf.tobytes()
    scan = await service.import_bytes(
        image_pdf, format="pdf", title="Scan", source_key=f"file:validation/{RUN_KEY}/scan"
    )
    assert scan["state"] == "needs_ocr" and scan["active_revision_id"] is None
    checks["scan_not_activated"] = True

    child = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "eval.validate_documents", "--crash-child"],
        capture_output=True,
        env=dict(os.environ),
    )
    stderr = child.stderr.decode(errors="replace") if isinstance(child.stderr, bytes) else child.stderr
    assert child.returncode == 73, stderr
    receipt = json.loads((ROOT / "crash-receipt.json").read_text(encoding="utf-8"))
    with patch.object(ingestion, "parse_isolated", side_effect=AssertionError("Saved parser stage must be reused")):
        crash_recovered = await service.resume(UUID(receipt["revision_id"]))
    assert crash_recovered["state"] == "ready"
    checks["hard_process_exit_and_resume"] = True

    await service.sync_articles()
    async with get_worker_db_context() as db:
        articles = await article_repo.list_for_knowledge(db)
        documents = await knowledge_repo.list_article_documents(db)
        mapped = {row.article_id: row for row in documents}
        assert len(articles) == 20 and set(mapped) == {a.id for a in articles}
        for article in articles:
            active_id = mapped[article.id].active_revision_id
            body = article.body_text
            assert active_id is not None and body is not None
            snapshot = await knowledge_repo.get_revision(db, revision_id=active_id)
            assert snapshot and snapshot.state == "ready" and snapshot.canonical_text == body.strip()
        checks["legacy_article_mapping"] = len(mapped)
        checks["new_chunk_count"] = await db.scalar(select(func.count()).select_from(KnowledgeChunk))
        hits = await RagService(db).query("Updated markdown travel allowance 720 yuan", top_k=10)
        assert any(hit["document_id"] == recovered["document_id"] for hit in hits)
        assert all(hit["revision_id"] != duplicate["revision_id"] for hit in hits)
    tool_payload = json.loads(await search_rag_tool.ainvoke({"query": "Word travel allowance 630 yuan", "top_k": 10}))
    assert any(hit.get("document_id") == imported["docx"]["document_id"] for hit in tool_payload["results"])
    checks["retrieval_and_tool_use_active_versions_only"] = True
    return checks


async def validate_crawler_delivery() -> dict[str, Any]:
    """Verify the existing crawler's real PG upsert, not a fabricated document insert."""
    url = f"https://example.invalid/multiformat-check/{RUN_KEY}"
    candidate = Candidate("Crawler integration", url)
    html = "<article><h2>Long body</h2><p>" + ("Full source sentence. " * 450) + "TAIL-FACT</p></article>"
    parsed = parse_article(html)
    assert len(parsed.body_text) > 8000 and parsed.body_text.endswith("TAIL-FACT")
    hit = _article_hit({"name": "validation", "group": "competitor"}, candidate, parsed, None)
    article_id: int | None = None
    try:
        await save_hits_to_db([hit])
        async with get_worker_db_context() as db:
            row = await article_repo.get_by_url(db, url)
            assert row is not None
            article_id = row.id
            assert row.body_text == parsed.body_text and row.ingestion_payload == parsed.ingestion_payload
            await article_repo.mark_embedding_done(db, row.id)
        await save_hits_to_db([hit])
        async with get_worker_db_context() as db:
            row = await article_repo.get_by_url(db, url)
            assert row is not None and row.embedding_done
        short = parse_article("<article><h2>New section</h2><p>A <b>new</b> policy fact.</p></article>")
        await save_hits_to_db([_article_hit({"name": "validation", "group": "competitor"}, candidate, short, None)])
        async with get_worker_db_context() as db:
            row = await article_repo.get_by_url(db, url)
            assert row is not None and not row.embedding_done and row.ingestion_payload
        payload = row.ingestion_payload
        result = await KnowledgeService().import_bytes(
            payload["input"].encode(),
            format="html",
            title=row.title,
            source_key=f"article:{row.id}",
            article_id=row.id,
            url=row.url,
            site_name=row.site_name,
            group_type=row.group_type,
        )
        assert result["state"] == "ready" and result["active"]
        read = await KnowledgeService().read(UUID(result["document_id"]))
        assert read["content"] == short.body_text and read["article_id"] == article_id
        return {
            "long_body_tail_saved": True,
            "structured_payload_saved": True,
            "identical_upsert_keeps_done": True,
            "changed_upsert_marks_pending": True,
            "crawler_to_document_read": True,
            "test_article_cleaned": True,
        }
    finally:
        if article_id is not None:
            async with get_worker_db_context() as db:
                await db.execute(delete(Article).where(Article.id == article_id, Article.url == url))
        async with get_worker_db_context() as db:
            assert await db.scalar(select(func.count()).select_from(Article)) == 20


def main() -> None:
    if settings.POSTGRES_DB != "intelligence_multiformat_v1_validation":
        raise SystemExit("This validation only runs in intelligence_multiformat_v1_validation")
    if settings.RAG_EMBED_PROVIDER != "ollama" or settings.RAG_INDEX_SOURCE != "documents":
        raise SystemExit("Require local Ollama and the documents index")
    ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["DOCUMENT_VALIDATION_RUN"] = RUN_KEY
    if "--crash-child" in sys.argv:
        run_compatible_async(crash_child())
        return
    events: list[dict[str, Any]] = []
    with local_embeddings_only(events):
        checks = run_compatible_async(validate_crawler_delivery() if "--crawler-only" in sys.argv else validate())
    report = {
        "checks": checks,
        "http": events,
        "chat_calls": 0,
        "scope": "isolated PostgreSQL, synthetic imports and restored news",
    }
    report_file = "crawler-integration.json" if "--crawler-only" in sys.argv else "integration.json"
    (ROOT / report_file).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
