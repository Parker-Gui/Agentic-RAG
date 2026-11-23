"""Thin Mem0 OSS adapter for explicit, user-owned profile facts."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from app.agents.memory.notes import DATA_ROOT
from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.db.models.article_chunk import EMBED_DIM

if TYPE_CHECKING:
    from mem0 import Memory

logger = logging.getLogger(__name__)
PROFILE_AGENT_ID = "intelligence-profile"
PROFILE_TABLE = "agent_profile_memories"
RESPONSE_PREFERENCE = "response_preference"


def profile_config() -> dict[str, Any]:
    """Separate 1024-dimensional PGVector table and SQLite audit history."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if settings.LLM_PROVIDER == "openai":
        llm = {
            "provider": "openai",
            "config": {
                "model": settings.OPENAI_MODEL,
                "api_key": settings.OPENAI_API_KEY,
                "openai_base_url": settings.OPENAI_BASE_URL,
            },
        }
    else:
        llm = {
            "provider": "ollama",
            "config": {
                "model": settings.OLLAMA_MODEL,
                "ollama_base_url": settings.OLLAMA_BASE_URL or "http://127.0.0.1:11434",
            },
        }
    return {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "dbname": settings.POSTGRES_DB,
                "collection_name": PROFILE_TABLE,
                "embedding_model_dims": EMBED_DIM,
                "user": settings.POSTGRES_USER,
                "password": settings.POSTGRES_PASSWORD,
                "host": settings.POSTGRES_SERVER,
                "port": settings.POSTGRES_PORT,
                "hnsw": True,
                "minconn": 1,
                "maxconn": 5,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": settings.RAG_OLLAMA_EMBED_MODEL,
                "embedding_dims": EMBED_DIM,
                "ollama_base_url": settings.RAG_OLLAMA_BASE_URL,
            },
        },
        "llm": llm,
        "history_db_path": str(DATA_ROOT / "profile_history.sqlite"),
    }


class ProfileMemory:
    def __init__(self, memory: Memory) -> None:
        self._memory = memory
        self._closed = False

    def close(self) -> None:
        """Mem0 2.2.1 closes SQLite only; explicitly release its PGVector pools too."""
        if self._closed:
            return
        self._closed = True
        resources: list[Any] = [self._memory]
        for name in ("vector_store", "_telemetry_vector_store", "_entity_store"):
            store = getattr(self._memory, name, None)
            pool = getattr(store, "connection_pool", None)
            if pool is not None:
                resources.append(pool)
        for component in (getattr(self._memory, "llm", None), getattr(self._memory, "embedding_model", None)):
            client = getattr(component, "client", None)
            if client is not None:
                resources.append(
                    client if callable(getattr(client, "close", None)) else getattr(client, "_client", None)
                )
            resources.append(getattr(getattr(component, "config", None), "http_client", None))
        seen: set[int] = set()
        for resource in resources:
            close = getattr(resource, "close", None)
            if callable(close) and id(resource) not in seen:
                seen.add(id(resource))
                try:
                    close()
                except Exception as exc:
                    logger.warning("Profile resource shutdown failed: %s", type(exc).__name__)

    async def recall(self, user_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        response = await asyncio.to_thread(
            self._memory.get_all,
            filters={"user_id": user_id, "agent_id": PROFILE_AGENT_ID, "memory_kind": RESPONSE_PREFERENCE},
            top_k=limit,
            show_expired=False,
        )
        styles = response["results"]
        if not isinstance(styles, list) or any(
            not isinstance(row, dict)
            or row.get("user_id") != user_id
            or row.get("agent_id") != PROFILE_AGENT_ID
            or row.get("metadata", {}).get("memory_kind") != RESPONSE_PREFERENCE
            for row in styles
        ):
            raise ValueError("Invalid scoped preference response")
        by_id = {row["id"]: row for row in styles if row.get("id") and row.get("memory")}
        if len(by_id) >= limit:
            return list(by_id.values())[:limit]
        response = await asyncio.to_thread(
            self._memory.search,
            query,
            filters={"user_id": user_id, "agent_id": PROFILE_AGENT_ID},
            top_k=limit,
        )
        for row in response.get("results", []):
            if row.get("user_id") == user_id and row.get("agent_id") == PROFILE_AGENT_ID and row.get("id"):
                by_id.setdefault(row["id"], row)
        return list(by_id.values())[:limit]

    async def remember(self, user_id: str, text: str, *, task_id: str | None = None) -> list[dict[str, Any]]:
        fact = text.strip()
        if not fact or len(fact) > 500:
            raise ValueError("Profile fact must contain 1–500 characters")
        metadata = (
            {"metadata": {"memory_kind": RESPONSE_PREFERENCE, "source_task_id": str(UUID(task_id))}}
            if task_id is not None
            else {}
        )
        response = await self._mutate(
            "remember",
            self._memory.add,
            fact,
            user_id=user_id,
            agent_id=PROFILE_AGENT_ID,
            infer=False,
            **metadata,
        )
        return response.get("results", [])

    async def _mutate(
        self, operation: str, call: Callable[..., Any], *args: Any, memory_id: str | None = None, **kwargs: Any
    ) -> Any:
        try:
            return await asyncio.to_thread(call, *args, **kwargs)
        except Exception as exc:
            # PGVector and SQLite history are separate transactions. Never claim
            # rollback or retry a write automatically after an uncertain failure.
            logger.warning("Profile %s failed: %s", operation, type(exc).__name__)
            raise ExternalServiceError(
                message="记忆操作失败，部分内容可能已写入；请查询当前记录后再操作。",
                details={"operation": operation, "memory_id": memory_id, "may_have_applied": True},
            ) from exc

    async def _owned(self, user_id: str, memory_id: str) -> dict[str, Any]:
        row = await asyncio.to_thread(self._memory.get, memory_id)
        if not row or row.get("user_id") != user_id or row.get("agent_id") != PROFILE_AGENT_ID:
            raise PermissionError("Profile memory not found for this user")
        return row

    async def update(self, user_id: str, memory_id: str, text: str) -> dict[str, Any]:
        fact = text.strip()
        if not fact or len(fact) > 500:
            raise ValueError("Profile fact must contain 1–500 characters")
        await self._owned(user_id, memory_id)
        return await self._mutate("update", self._memory.update, memory_id, text=fact, memory_id=memory_id)

    async def forget(self, user_id: str, memory_id: str) -> None:
        await self._owned(user_id, memory_id)
        await self._mutate("forget", self._memory.delete, memory_id, memory_id=memory_id)

    async def history(self, user_id: str, memory_id: str) -> list[dict[str, Any]]:
        await self._owned(user_id, memory_id)
        return await asyncio.to_thread(self._memory.history, memory_id)


_memory: ProfileMemory | None = None
_lock = threading.Lock()


def get_profile_memory() -> ProfileMemory:
    global _memory
    with _lock:
        if _memory is None:
            # Mem0 reads these variables at import time and otherwise writes to
            # the OS user's home. Keep its local metadata with the task data.
            os.environ.setdefault("MEM0_DIR", str(DATA_ROOT / "mem0"))
            os.environ.setdefault("MEM0_TELEMETRY", "false")
            from mem0 import Memory

            _memory = ProfileMemory(Memory.from_config(profile_config()))
        return _memory


def close_profile_memory() -> None:
    global _memory
    with _lock:
        if _memory is not None:
            _memory.close()
            _memory = None
