"""Artifact export validates reviewed sources, ownership and immutable reuse."""

import hashlib
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.exceptions import AlreadyExistsError, NotFoundError
from app.services.artifact import ArtifactService, artifact_metadata, render_brief
from tests.helpers import FakeArtifact, reviewed_task_state


@pytest.fixture
def artifact_repo() -> Iterator[MagicMock]:
    @asynccontextmanager
    async def context() -> AsyncIterator[AsyncMock]:
        yield AsyncMock()

    with (
        patch("app.services.artifact.get_worker_db_context", context),
        patch("app.services.artifact.artifact_repo") as repo,
    ):
        yield repo


def test_brief_preserves_answer_and_exports_only_cited_sources() -> None:
    state = reviewed_task_state()
    brief = render_brief(state, task_id=uuid4(), created_at=datetime.now(UTC))
    assert state["answer"] in brief
    assert "客户公告" in brief and "https://example.org/news" in brief
    assert "2026-11-19" in brief
    assert "无关材料" not in brief
    assert "个人偏好不应导出" not in brief
    assert "private/checkpoints.sqlite" not in brief
    assert "不代表所有信息均已覆盖" in brief


@pytest.mark.parametrize(
    "change",
    [
        {"termination": "execution_error"},
        {"review": {"approved": False}},
        {"answer": " "},
        {"termination": "model_budget"},
    ],
)
def test_unfinished_or_unreviewed_task_cannot_be_exported(change: dict[str, Any]) -> None:
    state = reviewed_task_state() | change
    with pytest.raises(AlreadyExistsError) as caught:
        render_brief(state, task_id=uuid4(), created_at=datetime.now(UTC))
    assert caught.value.code == "ARTIFACT_NOT_READY"


@pytest.mark.parametrize("answer", ["没有引用", "未知引用[E-000000000000]", "合法[E-123456789abc]和异常[E-bad]"])
def test_missing_or_invalid_source_rejected(answer: str) -> None:
    state = reviewed_task_state() | {"answer": answer}
    with pytest.raises(AlreadyExistsError) as caught:
        render_brief(state, task_id=uuid4(), created_at=datetime.now(UTC))
    assert caught.value.code == "ARTIFACT_SOURCE_INVALID"


@pytest.mark.anyio
async def test_create_saves_utf8_hash_and_uses_task_ownership(artifact_repo: MagicMock) -> None:
    user_id, task_id = uuid4(), uuid4()
    row = FakeArtifact()
    artifact_repo.get_for_task = AsyncMock(return_value=None)
    artifact_repo.create_or_get = AsyncMock(return_value=(row, True))
    with patch(
        "app.services.artifact.AgentService.get_task", new_callable=AsyncMock, return_value=reviewed_task_state()
    ) as get:
        metadata, created = await ArtifactService().create(user_id, task_id)
    get.assert_awaited_once_with(user_id, task_id)
    assert created and metadata["id"] == row.id
    call = artifact_repo.create_or_get.await_args
    assert call is not None
    payload = call.kwargs["content"].encode("utf-8")
    assert call.kwargs["size_bytes"] == len(payload)
    assert call.kwargs["sha256"] == hashlib.sha256(payload).hexdigest()
    assert call.kwargs["user_id"] == user_id


@pytest.mark.anyio
async def test_repeated_creation_reuses_saved_snapshot(artifact_repo: MagicMock) -> None:
    row = FakeArtifact()
    artifact_repo.get_for_task = AsyncMock(return_value=row)
    artifact_repo.create_or_get = AsyncMock()
    with patch("app.services.artifact.AgentService.get_task", new_callable=AsyncMock) as get:
        metadata, created = await ArtifactService().create(row.user_id, row.task_id)
    assert not created and metadata == artifact_metadata(row)
    artifact_repo.create_or_get.assert_not_awaited()
    get.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["get", "download"])
async def test_foreign_artifact_not_found(artifact_repo: MagicMock, method: str) -> None:
    artifact_repo.get_owned = AsyncMock(return_value=None)
    with pytest.raises(NotFoundError):
        await getattr(ArtifactService(), method)(uuid4(), uuid4())


@pytest.mark.anyio
async def test_foreign_task_cannot_create_or_list_artifacts(artifact_repo: MagicMock) -> None:
    artifact_repo.get_for_task = AsyncMock(return_value=None)
    artifact_repo.create_or_get = AsyncMock()
    with patch("app.services.artifact.AgentService.get_task", new_callable=AsyncMock, side_effect=NotFoundError):
        with pytest.raises(NotFoundError):
            await ArtifactService().create(uuid4(), uuid4())
        with pytest.raises(NotFoundError):
            await ArtifactService().list_for_task(uuid4(), uuid4())
    artifact_repo.create_or_get.assert_not_awaited()


def test_source_without_location_is_rejected() -> None:
    state = reviewed_task_state()
    state["evidence"][0].pop("url")
    with pytest.raises(AlreadyExistsError) as caught:
        render_brief(state, task_id=uuid4(), created_at=datetime.now(UTC))
    assert caught.value.code == "ARTIFACT_SOURCE_INVALID"


@pytest.mark.anyio
async def test_conflict_without_owned_snapshot_returns_not_found(artifact_repo: MagicMock) -> None:
    artifact_repo.get_for_task = AsyncMock(return_value=None)
    artifact_repo.create_or_get = AsyncMock(return_value=(None, False))
    with patch(
        "app.services.artifact.AgentService.get_task", new_callable=AsyncMock, return_value=reviewed_task_state()
    ):
        with pytest.raises(NotFoundError):
            await ArtifactService().create(uuid4(), uuid4())
