"""Test actual SDK export batches as well as fail-open application boundaries."""

import asyncio
import json
import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from app.core import observability as obs
from app.core.config import settings


@pytest.fixture
def memory_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    key = f"pk-test-{uuid4()}"
    client = obs.build_client(public_key=key, secret_key="test-only", base_url="http://127.0.0.1:1", exporter=exporter)
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", key)
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "test-only")
    monkeypatch.setattr(obs, "_client", client)
    yield client, exporter
    client.shutdown()


def test_disabled_observability_does_not_construct_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    with patch.object(obs, "build_client") as build:
        with obs.observation("operation.run", question="PRIVATE"):
            assert obs.callback_config() == {}
        build.assert_not_called()


def test_real_exporter_strips_nested_payloads_errors_and_unknown_names(
    memory_client: tuple[Any, InMemorySpanExporter],
) -> None:
    client, exporter = memory_client
    marker = "SECRET_PAYLOAD_123"
    # Deliberately bypass the helper to exercise defense at the final exporter boundary.
    with client.start_as_current_observation(
        name="operation.run",
        input={"profile": marker},
        output=marker,
        metadata={"question": marker, "model_calls": 3, "task_id": marker},
        status_message=marker,
    ):
        pass
    with client.start_as_current_observation(name=marker, input=marker):
        pass
    with client.start_as_current_observation(name="stage.plan") as event:
        event._otel_span.set_status(Status(StatusCode.ERROR, marker))
    with client.start_as_current_observation(name="stage.review") as event:
        event._otel_span.record_exception(ValueError(marker))
    client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    serialized = json.dumps(
        [
            {
                "name": s.name,
                "attrs": dict(s.attributes or {}),
                "resource": dict(s.resource.attributes),
                "events": [str(e) for e in s.events],
                "links": [str(link) for link in s.links],
            }
            for s in spans
        ]
    )
    assert marker not in serialized
    assert "model_calls" in serialized


@pytest.mark.anyio
async def test_operation_and_stage_ids_propagate_without_text(memory_client: tuple[Any, InMemorySpanExporter]) -> None:
    client, exporter = memory_client
    user, conversation, task = (uuid4() for _ in range(3))

    @obs.trace_operation("run")
    async def work(user_id: object, question: str) -> dict[str, Any]:
        obs.set_operation_ids(task_id=task, conversation_id=conversation)
        await asyncio.sleep(0)
        with obs.observation("stage.execute"):
            pass
        return {"task_id": str(task), "answer": question, "termination": "completed", "budget": {"model_calls": 2}}

    result = await work(user, "SECRET_BUSINESS_456")
    assert result["answer"] == "SECRET_BUSINESS_456"
    client.flush()
    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "operation.run")
    child = next(span for span in spans if span.name == "stage.execute")
    assert root.context.trace_id == child.context.trace_id
    assert child.parent is not None and root.attributes is not None and child.attributes is not None
    assert child.parent.span_id == root.context.span_id
    assert root.attributes["session.id"] == child.attributes["session.id"] == str(conversation)
    assert root.attributes["user.id"] == child.attributes["user.id"] == str(user)
    assert "SECRET_BUSINESS_456" not in str([dict(s.attributes or {}) for s in spans])


def test_business_exception_is_raised_once_without_sdk_exception_event(
    memory_client: tuple[Any, InMemorySpanExporter],
) -> None:
    client, exporter = memory_client
    with pytest.raises(ValueError, match="PRIVATE_ERROR"):
        with obs.observation("operation.export"):
            raise ValueError("PRIVATE_ERROR")
    client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and not spans[0].events
    assert "PRIVATE_ERROR" not in str(dict(spans[0].attributes or {}))


@pytest.mark.anyio
async def test_broken_platform_does_not_retry_business(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(obs, "get_client", Mock(side_effect=RuntimeError("PRIVATE_CONFIG")))
    calls = []

    @obs.trace_operation("export")
    async def export() -> dict[str, Any]:
        calls.append(1)
        return {"answer": "ok"}

    assert await export() == {"answer": "ok"}
    assert calls == [1]


def test_initialization_and_update_failures_are_safe(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(obs, "_client", None)
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "test-only")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "test-only")
    with patch.object(obs, "build_client", side_effect=ValueError("PRIVATE_CONFIG")):
        with obs.observation("operation.run"):
            pass
    broken = Mock()
    broken.update.side_effect = ValueError("PRIVATE_CONFIG")
    obs.Observation(broken).update(model_calls=2)
    assert "PRIVATE_CONFIG" not in caplog.text
    assert "initialization failed" in caplog.text and "update failed" in caplog.text


def test_export_failure_does_not_repeat_or_fail_business(memory_client: tuple[Any, InMemorySpanExporter]) -> None:
    from opentelemetry.sdk.trace.export import SpanExportResult

    client, exporter = memory_client
    with patch.object(exporter, "export", return_value=SpanExportResult.FAILURE) as export:
        with obs.observation("operation.run"):
            result = "successful business result"
        client.flush()
    assert result == "successful business result"
    assert export.call_count == 1


@pytest.mark.anyio
async def test_shutdown_wait_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    monkeypatch.setattr(obs, "_client", Mock(shutdown=lambda: release.wait(2)))
    monkeypatch.setattr(settings, "LANGFUSE_SHUTDOWN_SECONDS", 0.02)
    try:
        await asyncio.wait_for(obs.close_observability(), timeout=0.5)
    finally:
        release.set()


def test_usage_missing_is_not_zero_and_summary_not_double_counted() -> None:
    data = obs.result_metadata(
        {
            "budget": {
                "model_calls": 3,
                "summary_calls": 1,
                "usage": [
                    {"purpose": "execute", "usage": {"input_tokens": 10, "output_tokens": 4}},
                    {"purpose": "summary", "usage": None},
                ],
            }
        }
    )
    assert data["model_calls"] == 3
    assert data["summary_calls"] == 1
    assert data["reported_input_tokens"] == 10
    assert data["usage_missing_records"] == 1
    empty = obs.result_metadata({"budget": {"usage": [{"usage": None}]}})
    assert empty["usage_available"] is False
    assert "reported_input_tokens" not in empty


@pytest.mark.anyio
async def test_cached_resume_reports_zero_new_calls_and_retains_task_totals(
    memory_client: tuple[Any, InMemorySpanExporter],
) -> None:
    client, exporter = memory_client
    budget = {"model_calls": 5, "summary_calls": 1, "usage": [{"usage": {"input_tokens": 9}}]}

    @obs.trace_operation("resume")
    async def resume() -> dict[str, Any]:
        obs.set_operation_budget(budget)
        return {"answer": "cached", "budget": budget}

    assert (await resume())["answer"] == "cached"
    client.flush()
    attributes = exporter.get_finished_spans()[0].attributes or {}
    prefix = "langfuse.observation.metadata."
    assert attributes[f"{prefix}model_calls"] == "0"
    assert attributes[f"{prefix}task_model_calls"] == "5"
    assert attributes[f"{prefix}summary_calls"] == "0"
    assert f"{prefix}reported_input_tokens" not in attributes
