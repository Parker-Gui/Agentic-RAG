"""Fixed-fixture engineering evaluation. No real model, external tools, or business DB.

Default: actual SDK + in-memory exporter, local JSON only. --publish explicitly
uploads the fixed synthetic dataset, experiment and scores to configured Langfuse.
"""

import argparse
import hashlib
import json
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langsmith import tracing_context
from pydantic import Field

from app.agents import executor
from app.agents.context.conversation import select_history
from app.agents.evidence import citations, from_tool_messages
from app.agents.graph import build_graph
from app.core import observability as obs
from app.core.config import settings
from app.db.session import run_compatible_async
from app.services.artifact import render_brief
from eval.paths import FIXTURES

CASES = FIXTURES / "prototype_cases.jsonl"
REPORT = Path(__file__).parents[1] / ".agent_data" / "prototype-evaluation.json"
DATASET = "intelligence-analysis-synthetic-v1"


class FixtureModel(BaseChatModel):
    """Predetermined answers check plumbing, never claim to test LLM intelligence."""

    answer: str
    model_name: str = "scripted-fixture"
    calls: int = 0
    position: int = 0
    seen: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "prototype-fixture"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FixtureModel":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        text = "\n".join(str(message.content) for message in messages)
        self.calls += 1
        self.seen.append(text)
        if "Context Extraction Assistant" in text:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="合成任务目标：核查公告。保留当前来源与限制。"))]
            )
        responses = [
            AIMessage(content='{"goals":["核查合成公告"]}'),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_rag_tool", "args": {"query": "合成公告"}, "id": "synthetic-call"}],
            ),
            AIMessage(content="查证结束"),
            AIMessage(content=self.answer),
            AIMessage(content='{"approved":true,"unsupported_claims":[],"reason":"固定夹具"}'),
        ]
        response = responses[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    if case not in load_cases():
        raise ValueError("Only the checked-in synthetic fixture cases may be uploaded")
    started = time.monotonic()
    records = [
        {
            "article_id": index,
            "title": f"合成公告 {index}",
            "url": f"https://example.org/synthetic/{index}",
            "content": f"合成公司 {index} 宣布扩产。" + ("演示材料。" * 800 if case["kind"] == "summary" else ""),
        }
        for index in case["sources"]
    ]
    payload = json.dumps({"results": records}, ensure_ascii=False)
    if case["kind"] == "tool_error":
        payload = json.dumps({"results": [], "error": "synthetic tool failure"})
    expected_evidence, _ = from_tool_messages(
        [ToolMessage(content=payload, tool_call_id="fixture", name="search_rag_tool")], "S-1"
    )
    answer = "\n".join(f"合成公司 {i + 1} 宣布扩产。[{source.id}]" for i, source in enumerate(expected_evidence))

    @tool("search_rag_tool")
    async def fixture_search(query: str) -> str:
        """Read only this evaluator's fixed fixture. No business data access."""
        del query
        return payload

    user_id, conversation_id = str(uuid4()), str(uuid4())
    model = FixtureModel(answer=answer)
    initial = {
        "question": "整理合成公告中的扩产信息",
        "user_id": user_id,
        "task_id": str(uuid4()),
        "conversation_id": conversation_id,
        "budget": {"max_model_calls": 1 if case["kind"] == "budget" else 12},
        "evidence": [],
        "trace": [],
    }

    @contextmanager
    def context_settings() -> Any:
        with (
            patch.object(executor, "business_tools", [fixture_search]),
            patch.object(executor, "BUSINESS_TOOL_NAMES", {"search_rag_tool"}),
        ):
            if case["kind"] == "summary":
                with (
                    patch.object(settings, "AGENT_SUMMARY_TRIGGER_TOKENS", 1800),
                    patch.object(settings, "AGENT_KEEP_TOKENS", 100),
                ):
                    yield
            else:
                yield

    operations: list[dict[str, Any]] = []
    history_ok = None
    with context_settings():
        with obs.observation(
            "operation.run",
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=initial["task_id"],
            operation_id=str(uuid4()),
        ) as record:
            result = await build_graph(model).ainvoke(initial, config={"recursion_limit": 40, **obs.callback_config()})
            record.update(**obs.result_metadata(result))
            operations.append({"trace_id": record.trace_id, "task_id": result["task_id"], "model_calls": model.calls})
        if case["kind"] == "followup":
            history, sources = select_history([{**result, "status": "completed"}])
            previous_answer = result["answer"]
            model = FixtureModel(answer=answer)
            task_id = str(uuid4())
            with obs.observation(
                "operation.run",
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=task_id,
                operation_id=str(uuid4()),
            ) as record:
                result = await build_graph(model).ainvoke(
                    {
                        **initial,
                        "question": "展开第一条",
                        "task_id": task_id,
                        "conversation_history": history,
                        "evidence": sources,
                    },
                    config={"recursion_limit": 40, **obs.callback_config()},
                )
                record.update(**obs.result_metadata(result))
                operations.append({"trace_id": record.trace_id, "task_id": task_id, "model_calls": model.calls})
            history_ok = all(previous_answer.splitlines()[0] in model.seen[i] for i in (0, 1, 3))
            history_ok = history_ok and result["budget"]["model_calls"] == 5
    artifact_ok = None
    if case["kind"] == "artifact":
        calls_before = model.calls
        with obs.observation(
            "operation.export",
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=result["task_id"],
            operation_id=str(uuid4()),
        ) as record:
            brief = render_brief(result, task_id=UUID(result["task_id"]), created_at=datetime.now(UTC))
            encoded = brief.encode("utf-8")
            artifact_ok = result["answer"] in brief and all(source.id in brief for source in expected_evidence)
            artifact_ok = artifact_ok and model.calls == calls_before and len(hashlib.sha256(encoded).hexdigest()) == 64
            record.update(model_calls=0, size_bytes=len(encoded))
            operations.append({"trace_id": record.trace_id, "task_id": result["task_id"], "model_calls": 0})
    budget = result.get("budget", {})
    known_ids = {item["id"] for item in result.get("evidence", [])}
    cited = citations(result.get("answer", ""))
    expected_ids = set(case["sources"])
    found_ids = {item.get("article_id") for item in result.get("evidence", [])}
    checks = {
        "termination_expected": result.get("termination") == case["expected_termination"],
        "citation_id_valid": bool(cited) and all(item in known_ids for item in cited) if expected_ids else None,
        "source_recall": len(expected_ids & found_ids) / len(expected_ids) if expected_ids else None,
        "budget_respected": budget.get("model_calls", 0) <= initial["budget"]["max_model_calls"],
        "history_continuity": history_ok,
        "artifact_integrity": artifact_ok,
        "summary_triggered": budget.get("summary_calls", 0) > 0 if case["kind"] == "summary" else None,
        "tool_failure_recorded": bool(result.get("tool_errors")) if case["kind"] == "tool_error" else None,
    }
    return {
        "case_id": case["id"],
        "checks": checks,
        "passed": all(value is None or value is True or value == 1.0 for value in checks.values()),
        "model_calls": sum(operation["model_calls"] for operation in operations),
        "final_task_model_calls": budget.get("model_calls", 0),
        "summary_calls": budget.get("summary_calls", 0),
        "wall_seconds": time.monotonic() - started,
        "cost_usd": None,
        "operations": operations,
        "answer": result.get("answer"),
        "termination": result.get("termination"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for key in rows[0]["checks"] if rows else []:
        values = [row["checks"][key] for row in rows if row["checks"][key] is not None]
        metrics[key] = {"n": len(values), "mean": sum(values) / len(values) if values else None}
    return {
        "n": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "metrics": metrics,
        "cost_usd": None,
        "evaluation_type": "synthetic_engineering_contract",
        "paid_api_calls": 0,
    }


async def run_offline(client: Any, key: str) -> list[dict[str, Any]]:
    with tracing_context(enabled=False), obs.synthetic_observations(client, public_key=key):
        return [await run_case(case) for case in load_cases()]


def publish(client: Any, key: str) -> Any:
    """Public SDK dataset + experiment path. Only fixed synthetic data accepted."""
    from langfuse import Evaluation

    cases = load_cases()
    client.create_dataset(name=DATASET, description="Fixed synthetic engineering cases; no LLM quality claim")
    for case in cases:
        client.create_dataset_item(
            id=f"prototype-v1-{case['id']}",
            dataset_name=DATASET,
            input=case,
            expected_output=case["expected_termination"],
            metadata={"fixture_version": "v1"},
        )
    dataset = client.get_dataset(DATASET)
    # The SDK records dataset inputs before calling task(): reject unexpected
    # remote items here, not only inside run_case(), to keep publication synthetic.
    if len(dataset.items) != len(cases) or any(
        item.input not in cases
        or item.expected_output != item.input["expected_termination"]
        or item.metadata != {"fixture_version": "v1"}
        for item in dataset.items
    ):
        raise ValueError("Remote dataset must contain only the fixed synthetic fixtures")

    async def task(*, item: Any, **_kwargs: Any) -> dict[str, Any]:
        with tracing_context(enabled=False), obs.synthetic_observations(client, public_key=key):
            return await run_case(item.input)

    def evaluate(*, output: dict[str, Any], **_kwargs: Any) -> list[Any]:
        return [
            Evaluation(name=name, value=float(value)) for name, value in output["checks"].items() if value is not None
        ]

    return dataset.run_experiment(
        name="prototype-contract-v1",
        task=task,
        evaluators=[evaluate],
        max_concurrency=1,
        metadata=provenance(),
    )


def provenance() -> dict[str, Any]:
    """Capture versions without uploading repository paths, remotes or source."""
    revision = None
    dirty = None
    try:
        root = Path(__file__).parents[2]
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=2)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, timeout=2)
        if head.returncode == status.returncode == 0:
            revision, dirty = head.stdout.strip(), bool(status.stdout)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "dataset_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_version": "v1",
        "scorer_version": "v1",
        "model": "scripted-fixture",
        "git_revision": revision,
        "git_dirty": dirty,
        "packages": {name: version(name) for name in ("langfuse", "deepagents", "langchain-core", "langgraph")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish", action="store_true", help="Explicitly upload fixed synthetic data to configured Langfuse"
    )
    args = parser.parse_args()
    if args.publish:
        if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
            raise SystemExit("Configure LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in backend/.env first")
        client = obs.build_client(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            base_url=settings.LANGFUSE_BASE_URL,
            synthetic=True,
        )
        try:
            if not client.auth_check():
                raise SystemExit("Langfuse authentication failed; check local project keys and region")
            result = publish(client, settings.LANGFUSE_PUBLIC_KEY)
            rows = [item.output for item in result.item_results if isinstance(item.output, dict)]
        finally:
            obs.shutdown_client(client)
    else:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        key = f"pk-offline-{uuid4()}"
        client = obs.build_client(
            public_key=key, secret_key="offline-only", base_url="http://127.0.0.1:1", exporter=exporter, synthetic=True
        )
        try:
            rows = run_compatible_async(run_offline(client, key))
            client.flush()
            spans = exporter.get_finished_spans()
            REPORT.with_suffix(".spans.json").parent.mkdir(parents=True, exist_ok=True)
            REPORT.with_suffix(".spans.json").write_text(
                json.dumps(
                    [
                        {
                            "name": span.name,
                            "trace_id": f"{span.context.trace_id:032x}",
                            "span_id": f"{span.context.span_id:016x}",
                            "parent_id": f"{span.parent.span_id:016x}" if span.parent else None,
                            "attributes": dict(span.attributes or {}),
                        }
                        for span in spans
                    ],
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        finally:
            obs.shutdown_client(client)
    report = {"provenance": provenance(), "summary": summarize(rows), "cases": rows}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]))  # noqa: T201 — CLI report
    if not all(row["passed"] for row in rows) or len(rows) != len(load_cases()):
        raise SystemExit("Prototype evaluation failed or incomplete; see local report")


if __name__ == "__main__":
    main()
