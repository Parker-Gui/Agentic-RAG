"""Ownership and search-filter checks for the Mem0 adapter."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from app.agents.memory.profile import PROFILE_AGENT_ID, ProfileMemory, profile_config
from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.db.models.article_chunk import EMBED_DIM


class FakeMemory:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.last_filters: dict[str, str] = {}

    def add(self, text: str, *, user_id: str, agent_id: str, infer: bool) -> dict[str, Any]:
        assert infer is False
        row = {"id": str(len(self.rows) + 1), "memory": text, "user_id": user_id, "agent_id": agent_id}
        self.rows[row["id"]] = row
        return {"results": [row]}

    def search(self, query: str, *, filters: dict[str, str], top_k: int) -> dict[str, Any]:
        self.last_filters = filters
        return {"results": [row for row in self.rows.values() if row["user_id"] == filters["user_id"]][:top_k]}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        return self.rows.get(memory_id)

    def get_all(self, *, filters: dict[str, str], top_k: int, show_expired: bool) -> dict[str, Any]:
        return {"results": []}

    def update(self, memory_id: str, *, text: str) -> dict[str, Any]:
        self.rows[memory_id]["memory"] = text
        return self.rows[memory_id]

    def delete(self, memory_id: str) -> None:
        del self.rows[memory_id]


@pytest.mark.anyio
async def test_profile_is_filtered_and_foreign_id_cannot_be_changed() -> None:
    memory = FakeMemory()
    profile = ProfileMemory(cast(Any, memory))
    await profile.remember("user-a", "长期关注法规")
    recalled = await profile.recall("user-a", "法规")
    assert len(recalled) == 1
    assert memory.last_filters == {"user_id": "user-a", "agent_id": PROFILE_AGENT_ID}
    with pytest.raises(PermissionError):
        await profile.update("user-b", "1", "新偏好")
    with pytest.raises(PermissionError):
        await profile.forget("user-b", "1")
    assert memory.rows["1"]["memory"] == "长期关注法规"
    updated = await profile.update("user-a", "1", "长期关注食品法规")
    assert updated["memory"] == "长期关注食品法规"
    await profile.forget("user-a", "1")
    assert memory.rows == {}


def test_profile_pgvector_uses_distinct_1024_dimensional_table() -> None:
    config = profile_config()["vector_store"]["config"]
    assert config["collection_name"] == "agent_profile_memories"
    assert config["embedding_model_dims"] == EMBED_DIM == 1024


def test_profile_uses_deepseek_chat_and_local_ollama_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "deepseek-flash")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://chat-only.example")
    monkeypatch.setattr(settings, "RAG_OLLAMA_EMBED_MODEL", "bge-m3")
    monkeypatch.setattr(settings, "RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    config = profile_config()
    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["config"]["model"] == "deepseek-flash"
    assert config["llm"]["config"]["openai_base_url"] == "https://api.deepseek.com"
    assert config["embedder"]["provider"] == "ollama"
    assert config["embedder"]["config"]["model"] == "bge-m3"
    assert config["embedder"]["config"]["ollama_base_url"] == "http://127.0.0.1:11434"


def test_close_releases_shared_pool_once_and_is_idempotent() -> None:
    pool = SimpleNamespace(close=Mock())
    client = SimpleNamespace(close=Mock())
    memory = SimpleNamespace(
        close=Mock(),
        vector_store=SimpleNamespace(connection_pool=pool),
        _telemetry_vector_store=SimpleNamespace(connection_pool=pool),
        llm=SimpleNamespace(client=client),
    )
    profile = ProfileMemory(cast(Any, memory))
    profile.close()
    profile.close()
    pool.close.assert_called_once()
    client.close.assert_called_once()
    memory.close.assert_called_once()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["remember", "update", "forget"])
async def test_failed_write_reports_uncertain_outcome_without_retry(operation: str) -> None:
    failure = OSError("history storage unavailable")
    failing_write = Mock(side_effect=failure)
    memory = SimpleNamespace(
        add=failing_write,
        update=failing_write,
        delete=failing_write,
        get=lambda memory_id: {"id": memory_id, "user_id": "user-a", "agent_id": PROFILE_AGENT_ID},
    )
    profile = ProfileMemory(cast(Any, memory))
    with pytest.raises(ExternalServiceError) as captured:
        if operation == "remember":
            await profile.remember("user-a", "长期关注法规")
        elif operation == "update":
            await profile.update("user-a", "1", "长期关注食品法规")
        else:
            await profile.forget("user-a", "1")

    assert captured.value.status_code == 503
    assert captured.value.details == {
        "operation": operation,
        "memory_id": None if operation == "remember" else "1",
        "may_have_applied": True,
    }
    assert captured.value.__cause__ is failure
    failing_write.assert_called_once()
