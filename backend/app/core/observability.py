"""Fail-open metadata observations; detailed callbacks are for synthetic runners only."""

import asyncio
import inspect
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig

from app.core.config import settings

logger = logging.getLogger(__name__)
_client: Any = None
_lock = threading.Lock()
_demo: ContextVar[Any] = ContextVar("synthetic_observation_client", default=None)
_demo_key: ContextVar[str | None] = ContextVar("synthetic_public_key", default=None)
_demo_handler: ContextVar[Any] = ContextVar("synthetic_callback", default=None)
_current: ContextVar[Any] = ContextVar("current_operation", default=None)
_NAMES = {f"operation.{name}" for name in ("run", "resume", "export")}
_NAMES |= {f"stage.{name}" for name in ("plan", "execute", "draft", "review", "finalize", "summary")}
_IDS = {"user_id", "conversation_id", "task_id", "operation_id", "artifact_id"}
_NUMBERS = {
    "model_calls",
    "tool_calls",
    "summary_calls",
    "elapsed_seconds",
    "wall_seconds",
    "size_bytes",
    "reported_input_tokens",
    "reported_output_tokens",
    "usage_records",
    "usage_missing_records",
    "task_model_calls",
    "task_tool_calls",
    "task_summary_calls",
    "task_elapsed_seconds",
}
_FLAGS = {"created", "approved", "usage_available"}
_TERMINATIONS = {
    "completed",
    "insufficient_evidence",
    "model_budget",
    "tool_budget",
    "time_budget",
    "review_budget",
    "execution_error",
}


def safe_metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    """Allow keys AND values. Never recursively forward arbitrary state/metadata."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key in _IDS and value is not None:
            try:
                result[key] = str(UUID(str(value)))
            except (ValueError, TypeError, AttributeError):
                continue
        elif key in _NUMBERS and type(value) in (int, float) and 0 <= value < 10**12:
            result[key] = value
        elif key in _FLAGS and isinstance(value, bool):
            result[key] = value
        elif key == "termination" and isinstance(value, str) and value in _TERMINATIONS:
            result[key] = value
        elif key == "error_type" and isinstance(value, str) and value in {"operation_failed", "cancelled"}:
            result[key] = value
        elif key == "model_name" and isinstance(value, str) and value in {"deepseek-flash", "scripted-fixture"}:
            result[key] = value
    return result


def should_export_metadata(span: Any) -> bool:
    # Names, events, resources and links cannot all be sanitized by masking.
    return span.name in _NAMES and not span.events and not span.links and not span.status.description


def mask_metadata_spans(*, params: Any) -> Any:
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        delete = []
        for key, value in span.attributes.items():
            keep = key == "langfuse.observation.type" and value == "span"
            if key in {"user.id", "session.id"}:
                keep = bool(safe_metadata({"user_id": value}))
            if key.startswith("langfuse.observation.metadata."):
                field = key.removeprefix("langfuse.observation.metadata.")
                try:
                    decoded = json.loads(value)
                    keep = field in safe_metadata({field: decoded})
                except (ValueError, TypeError):
                    keep = False
            if not keep:
                delete.append(key)
        patches[identifier] = OtelSpanPatch(delete_attributes=tuple(delete))
    return MaskOtelSpansResult(span_patches=patches)


def build_client(
    *, public_key: str, secret_key: str, base_url: str, exporter: Any = None, synthetic: bool = False
) -> Any:
    """Isolated SDK provider; passing an exporter enables real-SDK offline tests."""
    from langfuse import Langfuse
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        timeout=2,
        tracer_provider=TracerProvider(resource=Resource({"service.name": "intelligence-analysis"})),
        span_exporter=exporter,
        mask_otel_spans=None if synthetic else mask_metadata_spans,
        should_export_span=None if synthetic else should_export_metadata,
        flush_interval=1,
    )


def get_client() -> Any:
    global _client
    if _demo.get() is not None:
        return _demo.get()
    if not settings.LANGFUSE_ENABLED:
        return None
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return None
    if _client is None:
        try:
            with _lock:
                if _client is None:
                    _client = build_client(
                        public_key=settings.LANGFUSE_PUBLIC_KEY,
                        secret_key=settings.LANGFUSE_SECRET_KEY,
                        base_url=settings.LANGFUSE_BASE_URL,
                    )
        except Exception:
            logger.warning("Observation client initialization failed")
    return _client


class Observation:
    def __init__(self, span: Any = None) -> None:
        self.span = span
        self.metadata: dict[str, Any] = {}
        self.budget_before: dict[str, Any] = {}

    @property
    def trace_id(self) -> str | None:
        return self.span.trace_id if self.span is not None else None

    def update(self, **data: Any) -> None:
        self.metadata.update(safe_metadata(data))
        if self.span is not None:
            try:
                self.span.update(metadata=self.metadata)
                from langfuse import propagate_attributes

                # Apply to the active root now; children re-propagate the IDs.
                with propagate_attributes(
                    user_id=self.metadata.get("user_id"), session_id=self.metadata.get("conversation_id")
                ):
                    pass
            except Exception:
                logger.warning("Observation update failed")


@contextmanager
def observation(name: str, **metadata: Any) -> Iterator[Observation]:
    record, stack = Observation(), ExitStack()
    start = time.monotonic()
    parent = _current.get()
    try:
        client = get_client()
        if client is not None:
            record.span = stack.enter_context(client.start_as_current_observation(name=name))
            from langfuse import propagate_attributes

            inherited = parent.metadata if parent is not None else {}
            ids = safe_metadata(inherited | metadata)
            stack.enter_context(propagate_attributes(user_id=ids.get("user_id"), session_id=ids.get("conversation_id")))
            record.update(**ids)
    except Exception:
        logger.warning("Observation start failed")
    token = _current.set(record)
    try:
        yield record
    except BaseException as exc:
        record.update(error_type="cancelled" if isinstance(exc, asyncio.CancelledError) else "operation_failed")
        raise
    finally:
        record.update(wall_seconds=time.monotonic() - start)
        _current.reset(token)
        try:
            # Never hand SDK context managers an exception containing business text.
            stack.__exit__(None, None, None)
        except Exception:
            logger.warning("Observation close failed")


def set_operation_ids(**metadata: Any) -> None:
    current = _current.get()
    if current is not None:
        current.update(**metadata)


def budget_snapshot(budget: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{key: budget.get(key, 0) for key in ("model_calls", "tool_calls", "summary_calls", "elapsed_seconds")},
        "usage_offset": len(budget.get("usage", [])),
    }


def set_operation_budget(budget: Mapping[str, Any]) -> None:
    """A resumed operation consumes only the delta after the saved checkpoint."""
    current = _current.get()
    if current is not None:
        current.budget_before = budget_snapshot(budget)


def result_metadata(result: Any, *, budget_before: Mapping[str, Any] | None = None) -> dict[str, Any]:
    created = None
    if isinstance(result, tuple) and len(result) == 2:
        result, created = result
    if not isinstance(result, dict):
        return {}
    data = safe_metadata(result)
    if created is not None:
        data.update(safe_metadata({"created": created, "artifact_id": result.get("id")}))
    budget = result.get("budget", {})
    before = budget_before or {}
    for key in ("model_calls", "tool_calls", "summary_calls", "elapsed_seconds"):
        total = budget.get(key, 0)
        data[key] = max(0, total - before.get(key, 0)) if budget else 0
        if budget:
            data[f"task_{key}"] = total
    usage = budget.get("usage", [])[before.get("usage_offset", 0) :]
    known = [row["usage"] for row in usage if isinstance(row.get("usage"), dict)]
    data.update(usage_available=bool(known), usage_records=len(known), usage_missing_records=len(usage) - len(known))
    if known:
        for source, target in (("input_tokens", "reported_input_tokens"), ("output_tokens", "reported_output_tokens")):
            reported = [row[source] for row in known if type(row.get(source)) is int]
            if reported:
                data[target] = sum(reported)
    return safe_metadata(data)


def trace_operation(kind: str) -> Callable[..., Any]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = inspect.signature(function).bind(*args, **kwargs)
            metadata = safe_metadata(bound.arguments)
            with observation(f"operation.{kind}", **metadata, operation_id=str(uuid4())) as record:
                result = await function(*args, **kwargs)
                try:
                    record.update(**result_metadata(result, budget_before=record.budget_before))
                except Exception:
                    logger.warning("Observation result collection failed")
                return result

        return wrapped

    return decorate


@contextmanager
def synthetic_observations(client: Any, *, public_key: str) -> Iterator[None]:
    """Only call from fixed-fixture evaluators, never select this from HTTP input."""
    token = _demo.set(client)
    key_token = _demo_key.set(public_key)
    handler_token = _demo_handler.set(None)
    try:
        yield
    finally:
        _demo.reset(token)
        _demo_key.reset(key_token)
        _demo_handler.reset(handler_token)


def callback_config() -> RunnableConfig:
    client = _demo.get()
    if client is None:
        return {}
    try:
        from langfuse.langchain import CallbackHandler

        handler = _demo_handler.get()
        if handler is None:
            handler = CallbackHandler(public_key=_demo_key.get())
            _demo_handler.set(handler)
        return {"callbacks": [handler]}
    except Exception:
        logger.warning("Synthetic callback initialization failed")
        return {}


def shutdown_client(client: Any) -> None:
    """Bound the SDK's flush/shutdown wait in both service and CLI processes."""

    def close() -> None:
        try:
            client.shutdown()
        except Exception:
            logger.warning("Observation shutdown failed")

    worker = threading.Thread(target=close, daemon=True)
    worker.start()
    worker.join(settings.LANGFUSE_SHUTDOWN_SECONDS)


async def close_observability() -> None:
    global _client
    client, _client = _client, None
    if client is not None:
        await asyncio.to_thread(shutdown_client, client)
