"""Run the same source-annotated tasks through legacy and new graphs.

Requires a working chat model, populated Postgres articles and any external search key
the chosen questions need. The report records observed values only.
"""

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import psycopg
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.config import settings
from eval.agent_metrics import aggregate, score_case
from eval.paths import FIXTURES

CASES = FIXTURES / "agent_cases.jsonl"
REPORT = Path(__file__).parent / "reports" / "agent_comparison.json"


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


async def preflight() -> None:
    """Refuse to publish comparison numbers from an unavailable model or corpus."""
    if settings.LLM_PROVIDER == "ollama":
        base_url = settings.OLLAMA_BASE_URL or "http://127.0.0.1:11434"
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
        installed = {str(item.get("name", "")).split(":")[0] for item in response.json().get("models", [])}
        if settings.OLLAMA_MODEL.split(":")[0] not in installed:
            raise RuntimeError(f"Configured Ollama chat model {settings.OLLAMA_MODEL!r} is not installed")
    elif not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for the configured chat provider")
    try:
        async with await psycopg.AsyncConnection.connect(settings.DATABASE_DSN, connect_timeout=3) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                extension = await cursor.fetchone()
                if not extension or not extension[0]:
                    raise RuntimeError("PostgreSQL pgvector extension is not installed")
                await cursor.execute("SELECT COUNT(*) FROM article_chunks")
                articles = await cursor.fetchone()
                if not articles or articles[0] == 0:
                    raise RuntimeError("article_chunks contains no indexed corpus")
    except psycopg.Error as exc:
        raise RuntimeError("PostgreSQL corpus is unavailable") from exc


def _legacy_evidence(messages: list[Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload, _ = json.JSONDecoder().raw_decode(str(message.content).lstrip())
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            continue
        evidence.extend(item for item in payload["results"] if isinstance(item, dict))
    return evidence


async def run_legacy(question: str) -> dict[str, Any]:
    from app.agents.legacy_graph import graph

    start = time.monotonic()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": f"legacy-comparison-{uuid4()}"}},
    )
    messages = result["messages"]
    answer = next((str(item.content) for item in reversed(messages) if isinstance(item, AIMessage)), "")
    return {
        "version": "legacy",
        "answer": answer,
        "evidence": _legacy_evidence(messages),
        "tool_calls": sum(isinstance(item, ToolMessage) for item in messages),
        "model_calls": sum(isinstance(item, AIMessage) for item in messages),
        "latency_seconds": time.monotonic() - start,
        "termination": "legacy_graph_end",
    }


async def run_new(question: str) -> dict[str, Any]:
    from app.agents.graph import graph

    start = time.monotonic()
    result = await graph.ainvoke(
        {"question": question, "user_id": str(uuid4()), "task_id": str(uuid4()), "profile": []},
        {"recursion_limit": 40},
    )
    budget = result.get("budget", {})
    return {
        "version": "new",
        "answer": result.get("answer", ""),
        "evidence": result.get("evidence", []),
        "tool_calls": budget.get("tool_calls", 0),
        "model_calls": budget.get("model_calls", 0),
        "latency_seconds": time.monotonic() - start,
        "termination": result.get("termination"),
        "review": result.get("review"),
    }


async def main() -> None:
    await preflight()
    cases = load_cases()
    details: dict[str, list[dict[str, Any]]] = {"legacy": [], "new": []}
    for case in cases:
        for version, runner in (("legacy", run_legacy), ("new", run_new)):
            output = await runner(case["question"])
            details[version].append({"question": case["question"], **score_case(case, output)})
    report = {"summary": {version: aggregate(rows) for version, rows in details.items()}, "details": details}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    from app.db.session import run_compatible_async

    run_compatible_async(main())
