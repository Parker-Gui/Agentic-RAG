"""Promote only successful, source-bearing tool records to research evidence."""

import hashlib
import json
import re
from collections.abc import Sequence

from langchain_core.messages import AnyMessage, ToolMessage

from app.agents.models import Claim, Evidence

_CITATION = re.compile(r"\[(E-[0-9a-f]{12})\]")


def from_tool_messages(messages: Sequence[AnyMessage], subtask_id: str) -> tuple[list[Evidence], list[str]]:
    evidence: list[Evidence] = []
    errors: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(str(message.content))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            continue
        if payload.get("error"):
            errors.append(f"{message.name}: {payload['error']}")
        for record in payload["results"]:
            if not isinstance(record, dict):
                continue
            raw_content = str(record.get("content") or "")
            content = raw_content.strip()
            source_start = record.get("start")
            if isinstance(source_start, int):
                source_start += len(raw_content) - len(raw_content.lstrip())
            url = record.get("url")
            article_id = record.get("article_id") or payload.get("article_id")
            document_id, revision_id = record.get("document_id"), record.get("revision_id")
            if not content or (not url and article_id is None and not document_id):
                continue
            source = (
                f"document:{document_id}:{revision_id}:{record.get('start')}" if document_id else str(url or article_id)
            )
            digest = hashlib.sha256(f"{source}\n{content}".encode()).hexdigest()[:12]
            evidence.append(
                Evidence(
                    id=f"E-{digest}",
                    subtask_id=subtask_id,
                    tool=message.name or "unknown",
                    title=str(record.get("title") or source),
                    content=content[:2000],
                    url=str(url) if url else None,
                    article_id=int(article_id) if article_id is not None else None,
                    document_id=str(document_id) if document_id else None,
                    revision_id=str(revision_id) if revision_id else None,
                    format=record.get("format"),
                    body_hash=record.get("body_hash"),
                    start=source_start,
                    end=(source_start + len(content[:2000])) if isinstance(source_start, int) else None,
                    locator=record.get("locator") or {},
                    truncated=bool(record.get("truncated")) or len(content) > 2000,
                    update_state=record.get("update_state"),
                    published_at=str(record["published_at"]) if record.get("published_at") else None,
                    source_kind=str(record["source_kind"]) if record.get("source_kind") else None,
                )
            )
    return evidence, errors


def source_location(source: dict) -> str:
    """Format a real source identity without inventing a file URL or server path."""
    if source.get("document_id"):
        locator = source.get("locator") or {}
        detail = (
            json.dumps(locator, ensure_ascii=False) if locator else f"chars={source.get('start')}:{source.get('end')}"
        )
        origin = f"{source['url']}; " if source.get("url") else ""
        update = (
            f"; latest_update={source['update_state']}" if source.get("update_state") not in {None, "ready"} else ""
        )
        return f"{origin}document_id={source['document_id']}; revision={source.get('revision_id')}; {detail}{update}"
    return str(source.get("url") or f"article_id={source.get('article_id')}")


def citations(text: str) -> list[str]:
    return _CITATION.findall(text)


def claims_from_draft(draft: str) -> list[Claim]:
    return [
        Claim(text=line.strip(), evidence_ids=citations(line))
        for line in draft.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
