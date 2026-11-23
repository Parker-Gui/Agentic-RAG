"""Isolated authenticated application runtime; scenario orchestration stays outside the agent."""

import asyncio
import json
import os
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langsmith import tracing_context

from app.agents import executor
from app.agents.agent_tools import read_article_tool, search_rag_tool
from app.agents.checkpoints import open_checkpointer
from app.agents.evidence import from_tool_messages
from app.agents.memory.profile import PROFILE_AGENT_ID, ProfileMemory, profile_config
from app.agents.models import Budget
from app.core.config import settings
from app.core.security import create_access_token
from app.db.session import get_worker_db_context
from app.main import create_app
from app.repository import user as user_repo
from app.services.agent import AgentService
from eval.e2e_data import E2ECase, save_json
from eval.validate_task_lifecycle import LifecycleModel


class EmptyProfile:
    async def recall(self, user_id: str, query: str) -> list[Any]:
        return []


class AppRuntime:
    def __init__(
        self,
        root: Path,
        *,
        live: bool = False,
        model_http: httpx.AsyncClient | None = None,
        request_count: Callable[[], int] | None = None,
    ) -> None:
        self.root, self.live, self.model_http = root, live, model_http
        self.request_count = request_count or (lambda: 0)
        self.stack = ExitStack()
        self.users: list[UUID] = []
        self.suffix = uuid4().hex
        self.profile: ProfileMemory | None = None
        self.model: LifecycleModel | None = None
        self.external_attempts: list[str] = []
        self.steps: list[dict[str, Any]] = []
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.search_calls = 0
        self.kind = ""
        self.client: httpx.AsyncClient | None = None
        self.email_by_id: dict[UUID, str] = {}

    def model_call_count(self) -> int:
        return self.request_count() if self.live else (self.model.calls if self.model is not None else 0)

    async def __aenter__(self) -> "AppRuntime":
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MEM0_TELEMETRY", "false")
            os.environ.setdefault("MEM0_DIR", str(self.root / "mem0"))
            self.stack.enter_context(tracing_context(enabled=False))
            self.stack.enter_context(patch.object(settings, "LANGFUSE_ENABLED", False))
            self.stack.enter_context(
                patch(
                    "mem0.llms.openai.OpenAILLM.generate_response",
                    side_effect=AssertionError("Mem0 live chat forbidden"),
                )
            )
            self.stack.enter_context(
                patch(
                    "mem0.llms.ollama.OllamaLLM.generate_response",
                    side_effect=AssertionError("Mem0 live chat forbidden"),
                )
            )
            if self.live:
                from app.agents.providers import get_llm_provider

                if settings.LLM_PROVIDER != "openai" or self.model_http is None:
                    raise ValueError("This evaluation supports only the configured OpenAI-compatible chat transport")
                if urlsplit(settings.RAG_OLLAMA_BASE_URL).hostname not in {"localhost", "127.0.0.1", "::1"}:
                    raise ValueError("Only local embedding endpoints are allowed")
                provider = get_llm_provider()

                def build_model(**kwargs: Any) -> Any:
                    kwargs.update(max_tokens=1024, max_retries=0, streaming=False, http_async_client=self.model_http)
                    return provider.build_model(**kwargs)

                self.stack.enter_context(patch.object(settings, "AGENT_OUTPUT_TOKENS", 1024))
                os.environ.setdefault("MEM0_TELEMETRY", "false")
                os.environ.setdefault("MEM0_DIR", str(self.root / "mem0"))
                from mem0 import Memory

                config = profile_config()
                config["history_db_path"] = str(self.root / "history.sqlite")
                self.profile = ProfileMemory(Memory.from_config(config))
                profile = self.profile
            else:
                for name in (
                    "app.agents.providers.OpenAIProvider.build_model",
                    "app.agents.providers.OllamaProvider.build_model",
                    "langchain_openai.ChatOpenAI._generate",
                    "langchain_openai.ChatOpenAI._agenerate",
                    "langchain_ollama.ChatOllama._generate",
                    "langchain_ollama.ChatOllama._agenerate",
                ):
                    self.stack.enter_context(
                        patch(name, side_effect=AssertionError("Live chat forbidden in fixture mode"))
                    )

                def build_model(**kwargs: Any) -> Any:
                    del kwargs
                    if self.model is None:
                        raise ValueError("Fixture not configured")
                    return self.model

                profile = EmptyProfile()
            for name in ("app.services.agent.get_llm_provider", "app.agents.graph.get_llm_provider"):
                self.stack.enter_context(patch(name, return_value=Mock(build_model=build_model)))
            self.stack.enter_context(patch("app.services.agent.get_profile_memory", return_value=profile))

            @tool("tavily_search")
            async def denied_search(query: str) -> str:
                """External network is disabled for this local-corpus evaluation."""
                del query
                self.external_attempts.append("tavily_search")
                return json.dumps({"results": [], "error": "external_network_disabled_for_evaluation"})

            @tool("tavily_extract")
            async def denied_extract(urls: list[str]) -> str:
                """External extraction is disabled for this evaluation."""
                del urls
                self.external_attempts.append("tavily_extract")
                return json.dumps({"results": [], "error": "external_network_disabled_for_evaluation"})

            if self.live:
                tools = [search_rag_tool, read_article_tool, denied_search, denied_extract]
            else:

                @tool("search_rag_tool")
                async def fixture_search(query: str) -> str:
                    """Only reads this case's fixed synthetic source."""
                    self.search_calls += 1
                    if self.kind == "recovery" and self.search_calls == 2:
                        self.started.set()
                        await self.release.wait()
                    if self.kind == "empty":
                        return '{"results":[]}'
                    if self.kind == "tool_error":
                        return '{"results":[],"error":"synthetic tool failure"}'
                    index = 1 if self.kind == "recovery" and "乙" in query else 0
                    return json.dumps({"results": [self.sources[index]]}, ensure_ascii=False)

                @tool("read_article_tool")
                async def fixture_read(article_id: int) -> str:
                    """Read only fixed fixture data, never the business database."""
                    return json.dumps({"results": [row for row in self.sources if row["article_id"] == article_id]})

                tools = [fixture_search, fixture_read, denied_search, denied_extract]
            self.stack.enter_context(patch.object(executor, "business_tools", tools))
            self.stack.enter_context(patch.object(executor, "BUSINESS_TOOL_NAMES", {item.name for item in tools}))
            created: list[UUID] = []
            async with get_worker_db_context() as db:
                for role in ("owner", "other"):
                    email = f"e2e-demo-{self.suffix}-{role}@example.invalid"
                    user = await user_repo.create(db, email=email, hashed_password="!disabled")
                    created.append(user.id)
                    self.email_by_id[user.id] = email
            self.users = created
            self.client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(), raise_app_exceptions=False), base_url="http://e2e.local"
            )
            return self
        except Exception:
            await self.__aexit__(None, None, None)
            raise

    def configure(self, case: E2ECase) -> None:
        self.kind = case.metadata.kind
        if self.live:
            return
        self.sources = [
            {
                "article_id": i + 1,
                "title": f"合成客户{name}公告",
                "url": f"https://example.invalid/fixture/{i}",
                "content": f"合成客户{name}披露扩产计划。",
            }
            for i, name in enumerate(("甲", "乙"))
        ]
        evidence, _ = from_tool_messages(
            [
                ToolMessage(
                    content=json.dumps({"results": self.sources}), name="search_rag_tool", tool_call_id="fixture"
                )
            ],
            "S-1",
        )

        def search(name: str) -> AIMessage:
            return AIMessage(content="", tool_calls=[{"name": "search_rag_tool", "args": {"query": name}, "id": name}])

        goals = ["核查甲", "核查乙"] if self.kind in {"missing_goal", "recovery"} else ["核查公告"]
        responses = [AIMessage(content=json.dumps({"goals": goals}))]
        if self.kind in {"empty", "tool_error"}:
            responses += [search("甲"), *[AIMessage(content="证据不足，未取得可用资料。") for _ in range(3)]]
        elif self.kind != "budget":
            for goal in goals:
                responses += [search(goal), AIMessage(content="已执行检索")]
            selected = evidence if self.kind == "recovery" else evidence[:1]
            answer = "\n".join(f"{row.content}[{row.id}]" for row in selected)
            responses += [
                AIMessage(content=answer),
                AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":"synthetic"}'),
            ]
        self.model = LifecycleModel(responses=responses)
        if self.kind == "budget":
            self.stack.enter_context(patch("app.services.agent.Budget", side_effect=lambda: Budget(max_model_calls=1)))

    async def api(self, method: str, path: str, *, payload: Any = None, other: bool = False) -> httpx.Response:
        if self.client is None:
            raise RuntimeError("Runtime not initialized")
        step: dict[str, Any] = {"method": method, "path": path, "input": payload, "state": "request_pending"}
        self.steps.append(step)
        save_json(self.root / "steps.json", self.steps)
        token = create_access_token(self.users[1 if other else 0])
        response = await self.client.request(method, path, json=payload, headers={"Authorization": f"Bearer {token}"})
        try:
            body = response.json()
        except ValueError:
            body = response.text
        step.update(state="received", http_status=response.status_code, output=body)
        save_json(self.root / "steps.json", self.steps)
        return response

    async def __aexit__(self, *args: Any) -> None:
        self.release.set()
        try:
            if self.client is not None:
                await self.client.aclose()
            service = AgentService()
            for user_id in self.users:
                async with get_worker_db_context() as db:
                    user = await user_repo.get_by_id(db, user_id)
                    if user is None or user.email != self.email_by_id[user_id]:
                        raise RuntimeError("Refusing cleanup: identity mismatch")
                listing = await service.list_tasks(user_id, limit=100)
                tasks = {row["task_id"] for row in listing["items"]}
                if self.profile is not None:
                    rows = await asyncio.to_thread(
                        self.profile._memory.get_all,
                        filters={"user_id": str(user_id), "agent_id": PROFILE_AGENT_ID},
                        top_k=100,
                        show_expired=True,
                    )
                    for row in rows["results"]:
                        if row.get("metadata", {}).get("source_task_id") not in tasks:
                            raise RuntimeError("Refusing cleanup: unrecorded memory")
                        await self.profile.forget(str(user_id), row["id"])
                async with open_checkpointer() as saver:
                    for task_id in tasks:
                        await saver.adelete_thread(f"{user_id}:{task_id}")
                async with get_worker_db_context() as db:
                    await user_repo.delete(db, db_user=user)
            save_json(self.root / "cleanup.json", {"isolated_users_removed": len(self.users)})
        finally:
            if self.profile is not None:
                self.profile.close()
            self.stack.close()
