"""Real routing, auth and download responses over stubbed artifact services."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.core.exceptions import AlreadyExistsError, NotFoundError
from app.services.artifact import artifact_metadata
from tests.helpers import FakeArtifact, FakeUser, auth_headers_for


@pytest.mark.anyio
@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/tasks/{id}/artifacts"),
        ("GET", "/tasks/{id}/artifacts"),
        ("GET", "/artifacts/{id}"),
        ("GET", "/artifacts/{id}/download"),
    ],
)
async def test_artifact_endpoints_require_auth(client: AsyncClient, method: str, path: str) -> None:
    response = await client.request(method, "/api/v1/agent" + path.format(id=uuid4()))
    assert response.status_code == 401


@pytest.mark.anyio
@pytest.mark.parametrize("created,expected_status", [(True, 201), (False, 200)])
async def test_create_returns_metadata_and_reuse_status(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser, created: bool, expected_status: int
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    row = FakeArtifact()
    with patch(
        "app.services.artifact.ArtifactService.create",
        new_callable=AsyncMock,
        return_value=(artifact_metadata(row), created),
    ) as create:
        response = await client.post(
            f"/api/v1/agent/tasks/{row.task_id}/artifacts", headers=auth_headers_for(fake_user)
        )
    assert response.status_code == expected_status
    assert response.json()["id"] == str(row.id)
    assert "content" not in response.json()
    create.assert_awaited_once_with(fake_user.id, row.task_id)


@pytest.mark.anyio
async def test_download_delivers_exact_markdown_attachment(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    row = FakeArtifact()
    with patch(
        "app.services.artifact.ArtifactService.download",
        new_callable=AsyncMock,
        return_value=(row.content.encode("utf-8"), row.filename),
    ) as download:
        response = await client.get(f"/api/v1/agent/artifacts/{row.id}/download", headers=auth_headers_for(fake_user))
    assert response.status_code == 200
    assert response.content == row.content.encode("utf-8")
    assert response.headers["content-type"] == "text/markdown; charset=utf-8"
    assert response.headers["content-disposition"] == f'attachment; filename="{row.filename}"'
    download.assert_awaited_once_with(fake_user.id, row.id)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "method,path,service_method",
    [
        ("POST", "/tasks/{id}/artifacts", "create"),
        ("GET", "/tasks/{id}/artifacts", "list_for_task"),
        ("GET", "/artifacts/{id}", "get"),
        ("GET", "/artifacts/{id}/download", "download"),
    ],
)
async def test_foreign_resources_return_404(
    client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser, method: str, path: str, service_method: str
) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    with patch(
        f"app.services.artifact.ArtifactService.{service_method}", new_callable=AsyncMock, side_effect=NotFoundError
    ):
        response = await client.request(
            method, "/api/v1/agent" + path.format(id=uuid4()), headers=auth_headers_for(fake_user)
        )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_unreviewed_task_returns_409(client: AsyncClient, mock_user_repo: AsyncMock, fake_user: FakeUser) -> None:
    mock_user_repo.get_by_id = AsyncMock(return_value=fake_user)
    with patch(
        "app.services.artifact.ArtifactService.create",
        new_callable=AsyncMock,
        side_effect=AlreadyExistsError(code="ARTIFACT_NOT_READY"),
    ):
        response = await client.post(f"/api/v1/agent/tasks/{uuid4()}/artifacts", headers=auth_headers_for(fake_user))
    assert response.status_code == 409
    assert response.json()["code"] == "ARTIFACT_NOT_READY"
