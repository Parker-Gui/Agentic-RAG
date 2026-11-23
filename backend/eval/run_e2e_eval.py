"""Local end-to-end evaluation. Default: fixtures only; live calls require approved data and explicit flags."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from app.agents.evidence import citations
from app.core.config import settings
from app.db.session import close_db, run_compatible_async
from eval.e2e_client import BudgetTransport
from eval.e2e_data import (
    CASES,
    OUTPUT,
    Corpus,
    E2ECase,
    fingerprint,
    load_cases,
    preflight_case,
    prepare_cases,
    read_corpus,
    save_json,
)
from eval.e2e_runtime import AppRuntime
from eval.e2e_scoring import HumanReview, score, summarize


class OutcomeUnknown(RuntimeError):
    pass


def provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=3)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, timeout=3)
    files = [*root.joinpath("backend/app").rglob("*.py"), *root.joinpath("backend/eval").glob("e2e*"), Path(__file__)]
    return {
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
            if path.is_file()
        },
        "packages": {name: version(name) for name in ("deepagents", "langgraph", "langfuse", "mem0ai")},
        "model": settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.OLLAMA_MODEL,
        "application_input_estimate_limit": settings.AGENT_INPUT_TOKENS,
        "normal_output_limit": settings.AGENT_OUTPUT_TOKENS,
        "cost_usd": None,
        "tool_scope": "internal-corpus-only; external tools denied",
        "cloud_export": False,
    }


async def create_task(
    runtime: AppRuntime, record: dict[str, Any], question: str, conversation_id: str | None = None
) -> dict[str, Any]:
    payload = {"question": question, **({"conversation_id": conversation_id} if conversation_id else {})}
    response = await runtime.api("POST", "/api/v1/agent/tasks", payload=payload)
    try:
        data = response.json()
    except ValueError as exc:
        raise OutcomeUnknown("Task creation response was not JSON; never replay it") from exc
    if not isinstance(data, dict) or not data.get("task_id"):
        raise OutcomeUnknown("Task creation did not provide an ID; never replay it")
    record["tasks"].append(data)
    response.raise_for_status()
    return data


async def exercise(case: E2ECase, runtime: AppRuntime, record: dict[str, Any]) -> None:
    runtime.configure(case)
    questions = case.input.questions
    if case.metadata.kind == "recovery":
        job = asyncio.create_task(create_task(runtime, record, questions[0]))
        try:
            await asyncio.wait_for(runtime.started.wait(), 30)
            listing = await runtime.api("GET", "/api/v1/agent/tasks")
            rows = listing.json()["items"]
            if len(rows) != 1:
                raise ValueError("Recovery fixture must have exactly one task")
            task_id = rows[0]["task_id"]
            paused = await runtime.api("POST", f"/api/v1/agent/tasks/{task_id}/pause")
            record["checks"]["pause_requested"] = paused.status_code == 202
            runtime.release.set()
            boundary = await job
            record["checks"]["safe_boundary"] = (
                boundary["status"] == "interrupted" and boundary["recovery"]["can_resume"]
            )
            before = boundary["budget"]
            resumed = await runtime.api("POST", f"/api/v1/agent/tasks/{task_id}/resume")
            resumed.raise_for_status()
            final = resumed.json()
            record["tasks"].append(final)
            record["checks"]["resume_preserves_budget"] = (
                final["budget"]["model_calls"] >= before["model_calls"]
                and final["budget"]["tool_calls"] == before["tool_calls"]
            )
            record["checks"]["resumed_same_task"] = final["task_id"] == boundary["task_id"]
        finally:
            runtime.release.set()
            if not job.done():
                job.cancel()
            await asyncio.gather(job, return_exceptions=True)
    else:
        first = await create_task(runtime, record, questions[0])
        if case.metadata.kind in {"followup_artifact", "followup", "memory"}:
            conversation = first["conversation_id"] if case.metadata.kind != "memory" else None
            second = await create_task(runtime, record, questions[1], conversation)
            record["checks"]["conversation_relation"] = (first["conversation_id"] == second["conversation_id"]) == (
                conversation is not None
            )
            if case.metadata.kind == "memory":
                record["checks"]["memory_saved_receipt"] = (first.get("memory_result") or {}).get("status") == "saved"
            if case.metadata.exports_artifact:
                calls = runtime.model_call_count()
                created = await runtime.api("POST", f"/api/v1/agent/tasks/{second['task_id']}/artifacts")
                created.raise_for_status()
                metadata = created.json()
                listing = await runtime.api("GET", f"/api/v1/agent/tasks/{second['task_id']}/artifacts")
                downloaded = await runtime.api("GET", metadata["download_url"])
                downloaded.raise_for_status()
                denied = await runtime.api("GET", metadata["download_url"], other=True)
                record["checks"].update(
                    artifact_content=second["answer"] in downloaded.text,
                    artifact_hash=hashlib.sha256(downloaded.content).hexdigest() == metadata["sha256"],
                    artifact_listed=any(row["id"] == metadata["id"] for row in listing.json()["items"]),
                    artifact_ownership=denied.status_code == 404,
                    export_no_model_calls=runtime.model_call_count() == calls,
                )
    record["checks"]["external_network_unused"] = not runtime.external_attempts


def task_checks(case: E2ECase, tasks: list[dict[str, Any]], sources: Corpus | None) -> dict[str, bool | None]:
    final = list({task["task_id"]: task for task in tasks}.values())
    ids_ok = []
    for task in final:
        if task.get("plan") and task.get("evidence"):
            ids = {row["id"] for row in task["evidence"]}
            found = citations(task.get("answer") or "")
            ids_ok.append(bool(found) and set(found) <= ids)
    options = case.expected_output.termination_options
    expected = options if options is not None else [[value] for value in case.expected_output.terminations]
    checks: dict[str, bool | None] = {
        "termination_expected": len(final) == len(expected)
        and all(task.get("termination") in values for task, values in zip(final, expected, strict=True)),
        "budget_respected": all(
            task["budget"]["model_calls"] <= task["budget"]["max_model_calls"]
            and task["budget"]["tool_calls"] <= task["budget"]["max_tool_calls"]
            for task in final
        ),
        "citation_id_valid": all(ids_ok) if ids_ok else None,
    }
    if case.metadata.kind in {"empty", "tool_error"}:
        checks["no_error_as_evidence"] = all(
            not task.get("evidence") and not citations(task.get("answer") or "") for task in final
        )
    if case.metadata.kind == "tool_error":
        checks["tool_failure_visible"] = any(
            any(item.get("errors", 0) for item in task.get("trace", [])) for task in final
        )
    if sources is not None and case.metadata.mode == "real_corpus":
        urls = {row.url for row in sources.sources if row.key in case.metadata.source_keys}
        found = {row.get("url") for task in final for row in task.get("evidence", [])}
        checks["required_sources_seen"] = (
            None if not urls and case.metadata.kind == "insufficient" else bool(urls) and urls <= found
        )
    return checks


def apply_reviews(root: Path, cases: list[E2ECase], records: list[dict[str, Any]], reviews: list[HumanReview]) -> None:
    by_id = {case.metadata.case_id: case for case in cases}
    if len({review.case_id for review in reviews}) != len(reviews):
        raise ValueError("Duplicate manual reviews")
    if any(review.case_id not in by_id for review in reviews):
        raise ValueError("Review references an unknown case")
    for record in records:
        case = by_id[record["case_id"]]
        record["split"] = case.metadata.split
        review = next((item for item in reviews if item.case_id == record["case_id"]), None)
        if review is None and record.get("human_review") is not None:
            review = HumanReview.model_validate(record["human_review"])
        if case.expected_output.negative_control and record["execution_status"] == "executed" and review is None:
            review = HumanReview(
                run_id=record["run_id"],
                case_id=record["case_id"],
                case_sha256=fingerprint(case.model_dump()),
                artifacts_sha256=fingerprint(record["artifacts"]),
                reviewer="fixed-negative-control-author",
                note="合成脚本仅有甲的来源，未提供乙的研究结果；这是已知反例，不是对真实模型的人工评分。",
                checks={"goal_completion": "fail", "evidence_support": "pass"},
            )
        record["score"] = score(case, record, review)
        record["human_review"] = review.model_dump() if review is not None else None
        save_json(root / record["case_id"] / "result.json", record)


def write_report(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"run_id": root.name, "summary": summarize(records), "cases": records}
    result["summary"]["by_split"] = {
        split: summarize([row for row in records if (row.get("split") or "unspecified") == split])["groups"]
        for split in sorted({row.get("split") or "unspecified" for row in records})
    }
    save_json(root / "report.json", result)
    lines = [
        "# 端到端验收报告",
        "",
        "真实业务质量与合成工程场景分别统计；pending/blocked 不算通过。费用未知为 null。",
        "",
        "| 场景 | 类型 | 分组 | 执行状态 | 判定 | 负例验证 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {row['case_id']} | {row['mode']} | {row.get('split') or 'unspecified'} | {row['execution_status']} | {row['score']['verdict']} | {row['score']['negative_control_verified']} |"
        for row in records
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.cases)
    if args.mode == "score":
        root = args.run_dir.resolve()
        cases = load_cases(root / "cases.jsonl")
        records = json.loads((root / "report.json").read_text(encoding="utf-8"))["cases"]
        reviews = [HumanReview.model_validate(row) for row in json.loads(args.reviews.read_text(encoding="utf-8"))]
        apply_reviews(root, cases, records, reviews)
        return write_report(root, records)
    root = OUTPUT / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    root.mkdir(parents=True)
    save_json(
        root / "manifest.json",
        {
            "run_id": root.name,
            "mode": args.mode,
            "provenance": provenance(),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    if args.mode == "prepare":
        corpus = await read_corpus()
        cases = prepare_cases(cases, corpus)
        save_json(root / "corpus.json", corpus.model_dump())
        (root / "cases.jsonl").write_text("\n".join(case.model_dump_json() for case in cases) + "\n", encoding="utf-8")
        (root / "review-guide.md").write_text(
            "# 待审标注\n\n来源和支持段保存在 corpus.json 与 cases.jsonl。所有 B 类仍为 draft；先逐条替换‘待人工核对’参考要点并审阅，不能只改 approval 标志。未批准前不得执行真实模型。\n",
            encoding="utf-8",
        )
        return {
            "prepared": str(root),
            "sources": len(corpus.sources),
            "corpus_version": corpus.version,
            "real_chat_requests": 0,
        }
    snapshot = Corpus.model_validate_json(args.sources.read_text(encoding="utf-8")) if args.sources else None
    if args.mode == "live" and (not args.allow_paid or args.ledger is None or snapshot is None):
        raise ValueError("Live mode requires --allow-paid, --ledger and reviewed --sources; no request sent")
    split = getattr(args, "split", None)
    if split is not None and args.case:
        raise ValueError("Select either --case or --split, not both")
    selected = set(
        args.case
        or [
            case.metadata.case_id
            for case in cases
            if (case.metadata.mode == "fixture") == (args.mode == "fixture")
            and (split is None or case.metadata.split == split)
        ]
    )
    if not selected:
        raise ValueError("No cases match the selected mode/split")
    if selected - {case.metadata.case_id for case in cases}:
        raise ValueError("Unknown case selection")
    (root / "cases.jsonl").write_text("\n".join(case.model_dump_json() for case in cases) + "\n", encoding="utf-8")
    if snapshot is not None:
        save_json(root / "corpus.json", snapshot.model_dump())
    budget = (
        BudgetTransport(httpx.AsyncHTTPTransport(retries=0), args.ledger, limit=args.max_requests)
        if args.mode == "live"
        else None
    )
    model_http = httpx.AsyncClient(transport=budget, timeout=httpx.Timeout(60, connect=10)) if budget else None
    records: list[dict[str, Any]] = []
    try:
        for case in cases:
            record: dict[str, Any] = {
                "run_id": root.name,
                "case_id": case.metadata.case_id,
                "mode": case.metadata.mode,
                "execution_status": "not_selected",
                "reason": None,
                "tasks": [],
                "checks": {},
                "artifacts": [],
            }
            records.append(record)
            if case.metadata.case_id not in selected:
                continue
            record["execution_status"] = "blocked"
            expected_mode = "real_corpus" if args.mode == "live" else "fixture"
            if case.metadata.mode != expected_mode:
                record["reason"] = "mode_mismatch"
                continue
            started = time.monotonic()
            try:
                if budget is not None and budget.exhausted:
                    record["reason"] = "request_budget_exhausted"
                    continue
                if case.metadata.mode == "real_corpus":
                    if snapshot is None:
                        raise ValueError("Missing corpus")
                    problems = preflight_case(case, snapshot, await read_corpus())
                    if problems:
                        record["reason"] = problems
                        continue
                    if settings.LLM_PROVIDER != "openai" or not settings.OPENAI_API_KEY or not settings.OPENAI_BASE_URL:
                        record["reason"] = "chat_configuration_missing"
                        continue
                    if urlsplit(settings.RAG_OLLAMA_BASE_URL).hostname not in {"127.0.0.1", "localhost", "::1"}:
                        record["reason"] = "embedding_endpoint_not_local"
                        continue
                    async with httpx.AsyncClient(timeout=3) as local:
                        response = await local.get(settings.RAG_OLLAMA_BASE_URL.rstrip("/") + "/api/tags")
                        response.raise_for_status()
                async with AppRuntime(
                    root / record["case_id"],
                    live=args.mode == "live",
                    model_http=model_http,
                    request_count=(lambda: len(budget.records)) if budget else None,
                ) as runtime:
                    try:
                        before_sends = len(budget.records) if budget else 0
                        await exercise(case, runtime, record)
                        record["checks"].update(task_checks(case, record["tasks"], snapshot))
                        if runtime.model is not None:
                            unique = {task["task_id"]: task for task in record["tasks"]}
                            record["checks"]["fixture_model_accounting"] = (
                                sum(task["budget"]["model_calls"] for task in unique.values()) == runtime.model.calls
                            )
                        record["execution_status"] = "executed"
                        record["http_sends"] = len(budget.records) - before_sends if budget else 0
                        if budget is not None and budget.exhausted:
                            record.update(execution_status="blocked", reason="request_budget_exhausted")
                    finally:
                        record["artifacts"] = [*runtime.steps]
                if snapshot is not None and (await read_corpus()).version != snapshot.version:
                    record.update(execution_status="blocked", reason="corpus_changed_during_run")
                    break
            except Exception as exc:
                pending_post = any(
                    step["method"] == "POST" and step["state"] == "request_pending" for step in record["artifacts"]
                )
                record.update(
                    execution_status="outcome_unknown"
                    if pending_post or isinstance(exc, OutcomeUnknown)
                    else ("failed" if record["artifacts"] else "blocked"),
                    reason=type(exc).__name__,
                )
            finally:
                record["wall_seconds"] = time.monotonic() - started
                final_tasks = list({task["task_id"]: task for task in record["tasks"]}.values())
                usage = [item.get("usage") for task in final_tasks for item in task.get("budget", {}).get("usage", [])]
                record["metrics"] = {
                    "logical_model_calls": sum(task.get("budget", {}).get("model_calls", 0) for task in final_tasks),
                    "tool_calls": sum(task.get("budget", {}).get("tool_calls", 0) for task in final_tasks),
                    "summary_calls": sum(task.get("budget", {}).get("summary_calls", 0) for task in final_tasks),
                    "usage_records": len(usage),
                    "usage_missing": sum(item is None for item in usage),
                    "reported_input_tokens": sum(
                        item.get("input_tokens", 0) for item in usage if isinstance(item, dict)
                    ),
                    "reported_output_tokens": sum(
                        item.get("output_tokens", 0) for item in usage if isinstance(item, dict)
                    ),
                    "cost_usd": None,
                }
                save_json(root / case.metadata.case_id / "raw.json", record)
    finally:
        if model_http is not None:
            await model_http.aclose()
        await close_db()
    # Preserve all planned rows even when a corpus change stopped the loop.
    for case in cases[len(records) :]:
        records.append(
            {
                "run_id": root.name,
                "case_id": case.metadata.case_id,
                "mode": case.metadata.mode,
                "execution_status": "blocked",
                "reason": "prior_corpus_change",
                "tasks": [],
                "checks": {},
                "artifacts": [],
            }
        )
    apply_reviews(root, cases, records, [])
    reviews = [
        HumanReview(
            run_id=root.name,
            case_id=case.metadata.case_id,
            case_sha256=fingerprint(case.model_dump()),
            artifacts_sha256=fingerprint(record["artifacts"]),
            reviewer="",
            note="",
            checks={},
        ).model_dump()
        for case, record in zip(cases, records, strict=True)
        if case.metadata.mode == "real_corpus" and record["execution_status"] == "executed"
    ]
    save_json(root / "human-reviews.template.json", reviews)
    report = write_report(root, records)
    report["report_path"] = str(root / "report.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixture", "prepare", "live", "score"), default="fixture")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--sources", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", action="append")
    selection.add_argument("--split", choices=("development", "holdout"))
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    if args.mode == "score" and (not args.run_dir or not args.reviews):
        parser.error("score requires --run-dir and --reviews")
    result = run_compatible_async(run(args))
    print(  # noqa: T201 -- local report summary
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key in {"summary", "report_path", "prepared", "sources", "corpus_version", "real_chat_requests"}
            },
            ensure_ascii=False,
        )
    )  # noqa: T201 -- local report summary
    if "cases" in result:
        selected = [row for row in result["cases"] if row["execution_status"] != "not_selected"]
        if any(
            row["execution_status"] == "failed"
            or row["score"]["verdict"] == "fail"
            and not row["score"]["negative_control_verified"]
            for row in selected
        ):
            raise SystemExit(1)
        if any(row["score"]["verdict"] == "pending" for row in selected):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
