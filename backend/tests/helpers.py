"""Shared test doubles and helpers.

Lives at the ``tests/`` root because every layer needs the same fakes: the
service-layer tests build them directly, the route tests hand them back from
patched repositories. Fixtures that wrap these doubles live in the nearest
``conftest.py`` instead.
"""

import hashlib
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID, uuid4

from app.core.security import create_access_token
from app.db.models.user import UserRole


class FakeArtifact:
    """Persisted artifact double shared by service and authenticated route tests."""

    def __init__(self, *, content: str = "# 情报简报\n\n正文。\n") -> None:
        self.id = uuid4()
        self.user_id = uuid4()
        self.task_id = uuid4()
        self.conversation_id = None
        self.title = "情报简报"
        self.format = "markdown"
        self.filename = f"brief-{self.id}.md"
        self.content = content
        self.size_bytes = len(content.encode("utf-8"))
        self.sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.created_at = datetime.now(UTC)


def reviewed_task_state() -> dict[str, Any]:
    """Synthetic reviewed task; unused sources and private context must not leak."""
    return {
        "question": "客户近期有什么动态？",
        "answer": "客户宣布扩产。[E-123456789abc]",
        "termination": "completed",
        "review": {"approved": True},
        "evidence": [
            {
                "id": "E-123456789abc",
                "title": "客户公告",
                "url": "https://example.org/news",
                "published_at": "2026-11-19",
            },
            {"id": "E-abcdefabcdef", "title": "无关材料", "url": "https://example.org/unused"},
        ],
        "profile": ["个人偏好不应导出"],
        "trace": [{"internal_path": "private/checkpoints.sqlite"}],
        "plan": [{"goal": "查询客户新闻", "status": "covered"}],
    }


class FakeUser:
    """Stand-in for ``app.db.models.user.User`` returned by a mocked repository."""

    def __init__(
        self,
        *,
        id: UUID | None = None,
        email: str = "user@example.com",
        full_name: str | None = "Test User",
        is_active: bool = True,
        role: UserRole = UserRole.USER,
        is_app_admin: bool = False,
        hashed_password: str | None = "$argon2id$fakehash",
    ) -> None:
        self.id = id or uuid4()
        self.email = email
        self.full_name = full_name
        self.is_active = is_active
        self.role = role.value
        self.is_app_admin = is_app_admin
        self.hashed_password = hashed_password
        self.avatar_url = None
        self.onboarding_completed_at = None
        self.oauth_provider = None
        self.created_at = datetime.now(UTC)
        self.updated_at = None

    def has_role(self, required_role: UserRole) -> bool:
        if self.role == UserRole.ADMIN.value:
            return True
        return self.role == required_role.value


class FakeArticle:
    """Stand-in for ``app.db.models.article.Article`` returned by a mocked repository."""

    def __init__(
        self,
        *,
        id: int = 1,
        site_name: str = "Example News",
        title: str = "Example headline",
        url: str = "https://example.com/a",
        publish_date: date = date(2026, 8, 20),
        body_text: str | None = "Full body text",
        summary: str | None = "A short summary",
        group_type: str = "competitor",
        is_external: bool = False,
        embedding_done: bool = False,
    ) -> None:
        self.id = id
        self.site_name = site_name
        self.title = title
        self.url = url
        self.publish_date = publish_date
        self.body_text = body_text
        self.summary = summary
        self.group_type = group_type
        self.is_external = is_external
        self.embedding_done = embedding_done
        self.created_at = datetime.now(UTC)
        self.updated_at = None


def auth_headers_for(user: FakeUser) -> dict[str, str]:
    """Build a Bearer header carrying a real access token for ``user``."""
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def by_id(*users: FakeUser) -> Callable[..., FakeUser | None]:
    """Build a ``get_by_id`` side effect resolving several users by id."""
    lookup = {user.id: user for user in users}
    return lambda db, user_id: lookup.get(user_id)
