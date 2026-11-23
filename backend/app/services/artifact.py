"""Export reviewed answers without another model call or a second content store."""

import hashlib
import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from app.agents.evidence import citations, source_location
from app.core.config import settings
from app.core.exceptions import AlreadyExistsError, NotFoundError
from app.core.observability import set_operation_ids, trace_operation
from app.db.models.artifact import Artifact
from app.db.session import get_worker_db_context
from app.repository import artifact as artifact_repo
from app.services.agent import AgentService

_ANY_CITATION = re.compile(r"\[(E-[^\]]+)\]")


def render_brief(state: dict[str, Any], *, task_id: UUID, created_at: datetime) -> str:
    """Copy the approved answer verbatim; add source metadata, not generated facts."""
    answer = state.get("answer", "")
    review = state.get("review") or {}
    if state.get("termination") != "completed" or review.get("approved") is not True or not answer.strip():
        raise AlreadyExistsError(message="简报仅支持已完成且审查通过的任务。", code="ARTIFACT_NOT_READY")
    sources = {row["id"]: row for row in state.get("evidence", [])}
    cited_ids = list(dict.fromkeys(citations(answer)))
    if not cited_ids or any(source_id not in sources for source_id in _ANY_CITATION.findall(answer)):
        raise AlreadyExistsError(message="答案引用缺失或不能对应已保存证据。", code="ARTIFACT_SOURCE_INVALID")
    lines = ["# 情报简报", "", f"- 生成时间：{created_at.isoformat()}", f"- 任务：{task_id}"]
    if state.get("conversation_id"):
        lines.append(f"- 对话：{state['conversation_id']}")
    lines.extend(["", "## 研究问题", "", state["question"], "", "## 正文", "", answer, "", "## 引用来源", ""])
    for source_id in cited_ids:
        source = sources[source_id]
        title = " ".join(str(source.get("title") or source_id).splitlines())
        origin = source_location(source)
        if not source.get("url") and source.get("article_id") is None and not source.get("document_id"):
            raise AlreadyExistsError(message="所引证据缺少来源位置。", code="ARTIFACT_SOURCE_INVALID")
        lines.append(f"- [{source_id}] {title}；来源：{origin}；发布日期：{source.get('published_at') or '未知'}")
    lines.extend(["", "## 范围与未解决事项", ""])
    pending = [str(item["goal"]) for item in state.get("plan", []) if item.get("status") != "covered"]
    if pending:
        lines.extend(f"- 任务记录待核查：{goal}" for goal in pending)
    else:
        lines.append("任务未记录未解决项，不代表所有信息均已覆盖。")
    lines.append("\n本简报是已审查任务结果的导出快照，事实与适用范围请结合所列原始来源核实。")
    return "\n".join(lines) + "\n"


def artifact_metadata(row: Artifact) -> dict[str, Any]:
    return {
        "id": row.id,
        "task_id": row.task_id,
        "conversation_id": row.conversation_id,
        "title": row.title,
        "format": row.format,
        "filename": row.filename,
        "size_bytes": row.size_bytes,
        "sha256": row.sha256,
        "created_at": row.created_at,
        "download_url": f"{settings.API_V1_STR}/agent/artifacts/{row.id}/download",
    }


class ArtifactService:
    @trace_operation("export")
    async def create(self, user_id: UUID, task_id: UUID) -> tuple[dict[str, Any], bool]:
        async with get_worker_db_context() as db:
            existing = await artifact_repo.get_for_task(db, task_id=task_id, user_id=user_id)
            if existing is not None:
                set_operation_ids(conversation_id=existing.conversation_id)
                return artifact_metadata(existing), False
        state = await AgentService().get_task(user_id, task_id)
        set_operation_ids(conversation_id=state.get("conversation_id"))
        artifact_id, created_at = uuid4(), datetime.now(UTC)
        content = render_brief(state, task_id=task_id, created_at=created_at)
        encoded = content.encode("utf-8")
        async with get_worker_db_context() as db:
            row, created = await artifact_repo.create_or_get(
                db,
                artifact_id=artifact_id,
                user_id=user_id,
                task_id=task_id,
                conversation_id=UUID(state["conversation_id"]) if state.get("conversation_id") else None,
                title="情报简报",
                filename=f"brief-{artifact_id}.md",
                content=content,
                size_bytes=len(encoded),
                sha256=hashlib.sha256(encoded).hexdigest(),
                created_at=created_at,
            )
            if row is None:
                raise NotFoundError(message="Task artifact not found")
            return artifact_metadata(row), created

    async def list_for_task(self, user_id: UUID, task_id: UUID) -> dict[str, Any]:
        await AgentService().get_task(user_id, task_id)
        async with get_worker_db_context() as db:
            row = await artifact_repo.get_for_task(db, task_id=task_id, user_id=user_id)
            items = [artifact_metadata(row)] if row is not None else []
            return {"items": items, "total": len(items)}

    async def _owned(self, user_id: UUID, artifact_id: UUID) -> Artifact:
        async with get_worker_db_context() as db:
            row = await artifact_repo.get_owned(db, artifact_id=artifact_id, user_id=user_id)
            if row is None:
                raise NotFoundError(message="Artifact not found")
            return row

    async def get(self, user_id: UUID, artifact_id: UUID) -> dict[str, Any]:
        return artifact_metadata(await self._owned(user_id, artifact_id))

    async def download(self, user_id: UUID, artifact_id: UUID) -> tuple[bytes, str]:
        row = await self._owned(user_id, artifact_id)
        return row.content.encode("utf-8"), row.filename
