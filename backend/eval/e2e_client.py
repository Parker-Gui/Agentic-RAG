"""One serial, durable HTTP-send budget; no credentials or model text in its ledger."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from eval.e2e_data import save_json


class RequestBudgetExceeded(RuntimeError):
    pass


class BudgetTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, ledger: Path, *, limit: int, wall_seconds: float = 900) -> None:
        if not 1 <= limit <= 40:
            raise ValueError("HTTP budget must be between 1 and 40")
        self.inner, self.ledger, self.limit = inner, ledger, limit
        self.deadline = time.monotonic() + wall_seconds
        self.lock = asyncio.Lock()
        self.exhausted = False
        ledger.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = ledger.with_suffix(".lock")
        self.lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            self.records: list[dict[str, Any]] = (
                json.loads(ledger.read_text(encoding="utf-8"))["requests"] if ledger.exists() else []
            )
            if not isinstance(self.records, list) or any(
                not isinstance(row, dict) or row.get("sequence") != i + 1 for i, row in enumerate(self.records)
            ):
                raise ValueError("Invalid budget ledger; never reset an uncertain count")
            self.exhausted = len(self.records) >= limit
        except Exception:
            os.close(self.lock_fd)
            self.lock_path.unlink()
            raise

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        caps = [body[key] for key in ("max_tokens", "max_completion_tokens") if key in body]
        if (
            request.method != "POST"
            or not request.url.path.endswith("/chat/completions")
            or body.get("stream")
            or not caps
            or any(type(value) is not int or not 1 <= value <= 1024 for value in caps)
            or len(request.content) > 128000
        ):
            raise RequestBudgetExceeded("Request outside approved non-streaming evaluation scope")
        async with self.lock:
            if len(self.records) >= self.limit or time.monotonic() >= self.deadline:
                self.exhausted = True
                raise RequestBudgetExceeded("Evaluation request/time budget exhausted")
            record: dict[str, Any] = {
                "sequence": len(self.records) + 1,
                "status": "reserved",
                "input_bytes": len(request.content),
                "output_cap": min(caps),
                "usage": None,
            }
            self.records.append(record)
            save_json(self.ledger, {"limit": self.limit, "requests": self.records})
            try:
                response = await self.inner.handle_async_request(request)
                raw = await response.aread()
                data = json.loads(raw) if response.is_success else {}
                usage = data.get("usage")
                record.update(
                    status="received",
                    http_status=response.status_code,
                    usage={
                        key: usage[key]
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if type(usage.get(key)) is int
                    }
                    if isinstance(usage, dict)
                    else None,
                )
                return response
            finally:
                save_json(self.ledger, {"limit": self.limit, "requests": self.records})

    async def aclose(self) -> None:
        try:
            await self.inner.aclose()
        finally:
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None
                self.lock_path.unlink()
