"""Global fixtures — everything here is available to every test package.

Layer-specific wiring lives in the nearest ``conftest.py`` instead:
``tests/api/conftest.py`` (ASGI client, patched repositories) and
``tests/services/conftest.py`` (service instances).
"""

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.db.models.user import UserRole
from tests.helpers import FakeArticle, FakeUser


@pytest.fixture(autouse=True)
def forbid_live_chat_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """An incomplete model double must fail locally, never spend a configured API key."""

    def protect(model_type: type[Any]) -> None:
        original_sync, original_async = model_type._generate, model_type._agenerate

        def blocked(model: Any, *args: Any, **kwargs: Any) -> Any:
            client = getattr(model, "http_client", None)
            if not isinstance(getattr(client, "_transport", None), httpx.MockTransport):
                raise AssertionError("Live chat calls forbidden; use a scripted model or explicit MockTransport")
            return original_sync(model, *args, **kwargs)

        async def async_blocked(model: Any, *args: Any, **kwargs: Any) -> Any:
            client = getattr(model, "http_async_client", None)
            if not isinstance(getattr(client, "_transport", None), httpx.MockTransport):
                raise AssertionError("Live chat calls forbidden; use a scripted model or explicit MockTransport")
            return await original_async(model, *args, **kwargs)

        monkeypatch.setattr(model_type, "_generate", blocked)
        monkeypatch.setattr(model_type, "_agenerate", async_blocked)

    for model_type in (ChatOpenAI, ChatOllama):
        protect(model_type)


@pytest.fixture
def anyio_backend() -> str:
    """Pin anyio to a single backend so async tests don't run twice (asyncio + trio)."""
    return "asyncio"


@pytest.fixture
def mock_db() -> AsyncMock:
    """Stand-in DB session.

    Nothing under test touches it directly -- each test patches the repository
    functions it needs, so the session only has to exist and be awaitable.
    """
    return AsyncMock()


@pytest.fixture
def fake_user() -> FakeUser:
    return FakeUser()


@pytest.fixture
def fake_admin() -> FakeUser:
    return FakeUser(email="admin@example.com", role=UserRole.ADMIN)


@pytest.fixture
def fake_article() -> FakeArticle:
    return FakeArticle()
