"""Corpus/label barriers, independent manual scores, and durable actual-send limits."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest

from eval import run_e2e_eval as runner
from eval.e2e_client import BudgetTransport, RequestBudgetExceeded
from eval.e2e_data import (
    Corpus,
    E2ECase,
    Source,
    corpus_version,
    fingerprint,
    load_cases,
    preflight_case,
    prepare_cases,
    save_json,
)
from eval.e2e_runtime import AppRuntime
from eval.e2e_scoring import HumanReview, score, summarize


def corpus() -> Corpus:
    rows = [
        Source(
            key=f"S-{i}",
            article_id=i,
            url=f"https://example.invalid/{i}",
            title=f"公告{i}",
            body=f"原文明确结论{i}，未说明投资额。",
            body_sha256=hashlib.sha256(f"原文明确结论{i}，未说明投资额。".encode()).hexdigest(),
            embedding_done=True,
            chunks=2,
            dimensions=[1024, 1024],
        )
        for i in (1, 2)
    ]
    return Corpus(version=corpus_version(rows), sources=rows)


def approved_case() -> Any:
    case = prepare_cases(load_cases(), corpus())[0]
    case.metadata.approval = "approved"
    case.expected_output.requirements[0].requirement = "区分明确结论与未说明的投资额。"
    return case


def record() -> dict[str, Any]:
    return {
        "run_id": "run-a",
        "execution_status": "executed",
        "checks": {"citation_id_valid": True},
        "artifacts": [{"answer": "原文明确结论1。"}],
    }


def review(case: Any, output: dict[str, Any]) -> HumanReview:
    return HumanReview(
        run_id=output["run_id"],
        case_id=case.metadata.case_id,
        case_sha256=fingerprint(case.model_dump()),
        artifacts_sha256=fingerprint(output["artifacts"]),
        reviewer="reviewer",
        note="逐条对照了原文与要点。",
        checks=dict.fromkeys(case.expected_output.manual_checks, "pass"),
    )


def test_nine_versioned_cases_and_drafts_do_not_become_approved_when_prepared() -> None:
    cases = prepare_cases(load_cases(), corpus())
    assert len(cases) == 9
    assert all(case.metadata.approval == "draft" for case in cases[:4])
    assert all(case.metadata.mode == "fixture" and case.metadata.approval == "approved" for case in cases[4:])
    assert "labels_not_approved" in preflight_case(cases[0], corpus(), corpus())
    cases[0].metadata.approval = "approved"
    assert "support_quote_invalid" in preflight_case(cases[0], corpus(), corpus())


def test_approved_labels_match_body_and_survive_local_id_remapping() -> None:
    current = corpus()
    current.sources[0].article_id = 999
    assert preflight_case(approved_case(), corpus(), current) == []


def test_expanded_business_ids_keep_split_and_legacy_serialization(tmp_path: Path) -> None:
    original = approved_case()
    assert "split" not in original.model_dump()["metadata"]
    assert "termination_options" not in original.model_dump()["expected_output"]
    original.metadata.case_id = "B-132"
    original.metadata.split = "holdout"
    path = tmp_path / "expanded.jsonl"
    path.write_text(original.model_dump_json(), encoding="utf-8")
    loaded = load_cases(path)[0]
    assert loaded.metadata.split == "holdout"
    assert preflight_case(loaded, corpus(), corpus()) == []


@pytest.mark.parametrize("kind", ["empty", "tool_error", "missing_goal", "budget", "recovery"])
def test_fixture_kinds_cannot_masquerade_as_real_business_cases(tmp_path: Path, kind: str) -> None:
    value = approved_case().model_dump()
    value["metadata"]["kind"] = kind
    path = tmp_path / "mixed.jsonl"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="scenario kinds"):
        load_cases(path)


def absence_case() -> E2ECase:
    case = approved_case()
    case.metadata.case_id = "B-129"
    case.metadata.kind = "insufficient"
    case.metadata.search_scope = "snapshot"
    case.metadata.source_keys = []
    case.expected_output.requirements = []
    case.expected_output.missing_information = "公司销售佣金比例和报销上限。"
    case.expected_output.must_not_claim = ["不得编造制度数值。"]
    case.expected_output.manual_checks.append("missing_information_disclosed")
    case.expected_output.terminations = ["insufficient_evidence"]
    return case


def test_scoped_absence_needs_no_fake_source_and_still_requires_manual_review() -> None:
    case = absence_case()
    assert preflight_case(case, corpus(), corpus()) == []
    assert score(case, record())["verdict"] == "pending"
    regular = approved_case()
    regular.metadata.source_keys = []
    regular.expected_output.requirements = []
    assert "references_missing" in preflight_case(regular, corpus(), corpus())


@pytest.mark.parametrize("missing", ["scope", "information", "prohibition", "manual_check"])
def test_absence_exception_requires_explicit_scope_gap_and_prohibitions(missing: str) -> None:
    case = absence_case()
    if missing == "scope":
        case.metadata.search_scope = None
    elif missing == "information":
        case.expected_output.missing_information = " "
    elif missing == "prohibition":
        case.expected_output.must_not_claim = [" "]
    else:
        case.expected_output.manual_checks.remove("missing_information_disclosed")
    assert "absence_rubric_missing" in preflight_case(case, corpus(), corpus())


def test_absence_checks_all_documents_in_the_frozen_scope() -> None:
    case = absence_case()
    current = corpus()
    current.sources[1].embedding_done = False
    assert "source_not_indexed" in preflight_case(case, corpus(), current)
    case.metadata.approval = "draft"
    assert "labels_not_approved" in preflight_case(case, corpus(), corpus())


def test_per_step_termination_options_are_explicit_and_validate_task_count() -> None:
    case = absence_case()
    case.expected_output.terminations = []
    case.expected_output.termination_options = [["completed", "insufficient_evidence"]]
    assert preflight_case(case, corpus(), corpus()) == []
    task = {
        "task_id": "one",
        "termination": "completed",
        "budget": {"model_calls": 1, "max_model_calls": 2, "tool_calls": 0, "max_tool_calls": 2},
    }
    checks = runner.task_checks(case, [task], corpus())
    assert checks["termination_expected"] is True and checks["required_sources_seen"] is None
    assert runner.task_checks(case, [], corpus())["termination_expected"] is False
    case.expected_output.terminations = ["completed"]
    assert "termination_labels_ambiguous" in preflight_case(case, corpus(), corpus())
    case.expected_output.terminations = []
    case.expected_output.termination_options = [[]]
    assert "termination_labels_incomplete" in preflight_case(case, corpus(), corpus())


def test_multiturn_labels_must_describe_each_turn() -> None:
    case = approved_case()
    case.metadata.kind = "followup"
    case.input.questions.append("展开刚才内容。")
    assert "termination_labels_incomplete" in preflight_case(case, corpus(), corpus())
    case.expected_output.terminations = ["completed", "completed"]
    assert preflight_case(case, corpus(), corpus()) == []
    case.metadata.kind = "memory"
    case.metadata.requires_artifact = True
    assert "artifact_configuration_invalid" in preflight_case(case, corpus(), corpus())


@pytest.mark.parametrize(
    "change,problem",
    [("missing_followup", "questions_unprepared"), ("unlabeled_second_source", "references_incomplete")],
)
def test_incomplete_multistep_or_source_labels_are_blocked_before_paid_work(change: str, problem: str) -> None:
    case = approved_case()
    if change == "missing_followup":
        case.metadata.kind = "followup_artifact"
    else:
        case.metadata.kind = "compare"
        case.metadata.source_keys = [row.key for row in corpus().sources]
    assert problem in preflight_case(case, corpus(), corpus())


@pytest.mark.parametrize(
    "change,problem",
    [
        ("body", "corpus_version_mismatch"),
        ("quote", "support_quote_invalid"),
        ("index", "source_not_indexed"),
        ("dimension", "source_not_indexed"),
        ("snapshot", "snapshot_corrupted"),
        ("rubric", "manual_rubric_missing"),
    ],
)
def test_preflight_blocks_mismatched_data_before_any_model(change: str, problem: str) -> None:
    case, snapshot, current = approved_case(), corpus(), corpus()
    if change == "body":
        current.sources[0].body_sha256 = "changed"
        current.version = corpus_version(current.sources)
    elif change == "quote":
        case.expected_output.requirements[0].quote = "原文没有这句话"
    elif change == "index":
        current.sources[0].chunks = 0
    elif change == "dimension":
        current.sources[0].dimensions = [768, 768]
    elif change == "snapshot":
        snapshot.sources[0].body = "tampered"
    else:
        case.expected_output.manual_checks = []
    assert problem in preflight_case(case, snapshot, current)


def test_agent_approved_and_valid_citations_are_not_independent_semantic_success() -> None:
    case, output = approved_case(), record()
    output["review"] = {"approved": True}
    assert score(case, output)["verdict"] == "pending"
    human = review(case, output)
    assert score(case, output, human)["verdict"] == "pass"
    human.checks["goal_completion"] = "fail"
    assert score(case, output, human)["verdict"] == "fail"


@pytest.mark.parametrize("field", ["run_id", "case_id", "case_sha256", "artifacts_sha256", "reviewer", "note"])
def test_review_must_match_the_case_and_its_actual_output(field: str) -> None:
    case, output = approved_case(), record()
    human = review(case, output)
    setattr(human, field, "" if field in {"reviewer", "note"} else "another-version")
    result = score(case, output, human)
    assert result["verdict"] == "pending" and result["review_error"]


@pytest.mark.parametrize("status", ["blocked", "failed", "outcome_unknown", "not_selected"])
def test_non_executed_case_never_passes_even_with_positive_human_review(status: str) -> None:
    case, output = approved_case(), record()
    output["execution_status"] = status
    assert score(case, output, review(case, output))["verdict"] == "pending"


def test_negative_control_requires_semantic_failure_and_working_engineering_path() -> None:
    case = load_cases()[4]
    output = record()
    human = review(case, output)
    human.checks["goal_completion"] = "fail"
    assert score(case, output, human)["negative_control_verified"] is True
    output["checks"]["http_failed"] = False
    assert score(case, output, human)["negative_control_verified"] is False


def test_report_keeps_unexecuted_denominators_and_fixture_quality_separate() -> None:
    rows = []
    for case in load_cases():
        output = {**record(), "case_id": case.metadata.case_id, "mode": case.metadata.mode}
        if case.metadata.mode == "real_corpus":
            output["execution_status"] = "blocked"
        output["score"] = score(case, output)
        rows.append(output)
    result = summarize(rows)
    assert result["groups"]["real_corpus"]["planned"] == 4
    assert result["groups"]["real_corpus"]["blocked"] == 4
    assert result["groups"]["real_corpus"]["pass"] == 0
    assert result["groups"]["fixture"]["planned"] == 5
    assert result["cost_usd"] is None


@pytest.mark.anyio
async def test_actual_send_budget_is_serial_durable_and_does_not_store_secrets(tmp_path: Path) -> None:
    calls = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"usage": {"prompt_tokens": 12, "completion_tokens": 3, "private": "secret-response"}}
        )

    ledger = tmp_path / "budget.json"
    transport = BudgetTransport(httpx.MockTransport(respond), ledger, limit=2)
    async with httpx.AsyncClient(transport=transport) as client:
        for _ in range(2):
            await client.post(
                "https://model.invalid/chat/completions",
                json={"max_tokens": 100, "messages": ["secret-input"]},
                headers={"Authorization": "Bearer secret-key"},
            )
        with pytest.raises(RequestBudgetExceeded):
            await client.post("https://model.invalid/chat/completions", json={"max_tokens": 100})
    restored = BudgetTransport(httpx.MockTransport(respond), ledger, limit=2)
    async with httpx.AsyncClient(transport=restored) as client:
        with pytest.raises(RequestBudgetExceeded):
            await client.post("https://model.invalid/chat/completions", json={"max_tokens": 100})
    assert calls == 2
    data = ledger.read_text(encoding="utf-8")
    assert "secret" not in data
    assert json.loads(data)["requests"][0]["usage"] == {"prompt_tokens": 12, "completion_tokens": 3}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [{"max_tokens": 1025}, {"max_tokens": 10, "max_completion_tokens": 4096}, {"max_tokens": 10, "stream": True}, {}],
)
async def test_request_scope_rejects_before_network_or_budget_reservation(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    send = Mock(side_effect=AssertionError("Network must not be reached"))
    transport = BudgetTransport(httpx.MockTransport(send), tmp_path / "budget.json", limit=1)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RequestBudgetExceeded):
            await client.post("https://model.invalid/chat/completions", json=payload)
    assert transport.records == []
    send.assert_not_called()


@pytest.mark.anyio
async def test_failed_http_send_keeps_its_reservation_and_blocks_parallel_ledger_use(tmp_path: Path) -> None:
    async def failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("simulated unknown outcome", request=request)

    ledger = tmp_path / "budget.json"
    transport = BudgetTransport(httpx.MockTransport(failure), ledger, limit=1)
    with pytest.raises(FileExistsError):
        BudgetTransport(httpx.MockTransport(failure), ledger, limit=1)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ReadError):
            await client.post("https://model.invalid/chat/completions", json={"max_tokens": 10})
    assert json.loads(ledger.read_text())["requests"][0]["status"] == "reserved"


@pytest.mark.anyio
async def test_live_mode_requires_explicit_permission_before_runtime_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    runtime = Mock(side_effect=AssertionError("Must not initialize live runtime"))
    monkeypatch.setattr(runner, "AppRuntime", runtime)
    args = argparse.Namespace(mode="live", cases=runner.CASES, sources=None, allow_paid=False, ledger=None)
    with pytest.raises(ValueError, match="requires"):
        await runner.run(args)
    runtime.assert_not_called()


@pytest.mark.anyio
async def test_unapproved_live_labels_block_runtime_before_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "runs")
    monkeypatch.setattr(runner, "read_corpus", AsyncMock(return_value=corpus()))
    runtime = Mock(side_effect=AssertionError("Unapproved data must not reach runtime"))
    monkeypatch.setattr(runner, "AppRuntime", runtime)
    cases = prepare_cases(load_cases(), corpus())
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(case.model_dump_json() for case in cases), encoding="utf-8")
    snapshot = tmp_path / "corpus.json"
    save_json(snapshot, corpus().model_dump())
    args = argparse.Namespace(
        mode="live",
        cases=path,
        sources=snapshot,
        allow_paid=True,
        ledger=tmp_path / "budget.json",
        max_requests=8,
        case=["B-01"],
    )
    result = await runner.run(args)
    item = result["cases"][0]
    assert item["execution_status"] == "blocked" and "labels_not_approved" in item["reason"]
    assert item["score"]["verdict"] == "pending"
    assert not (tmp_path / "budget.json").exists()
    runtime.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "kind,export",
    [("followup", False), ("followup", True), ("followup_artifact", True), ("memory", False)],
)
async def test_followup_exports_only_when_requested_and_memory_uses_a_new_conversation(kind: str, export: bool) -> None:
    case = approved_case()
    case.metadata.kind = kind
    if kind != "followup_artifact":
        case.metadata.requires_artifact = export
    case.input.questions = ["第一轮问题", "展开刚才内容"]
    case.expected_output.requirements[0].requirement = "PRIVATE_EXPECTED_ANSWER"
    runtime = Mock(spec=AppRuntime)
    runtime.external_attempts = []
    runtime.model_call_count.return_value = 7
    payloads = []
    answer = "实际回答，不来自参考答案。"
    artifact = {"id": "artifact", "download_url": "/download", "sha256": hashlib.sha256(answer.encode()).hexdigest()}

    async def api(method: str, path: str, *, payload: Any = None, other: bool = False) -> httpx.Response:
        if path == "/api/v1/agent/tasks":
            payloads.append(payload)
            conversation = "second" if kind == "memory" and len(payloads) == 2 else "first"
            return httpx.Response(
                201,
                json={
                    "task_id": str(len(payloads)),
                    "conversation_id": conversation,
                    "answer": answer,
                    "memory_result": {"status": "saved"},
                },
            )
        if path == "/download":
            return httpx.Response(404 if other else 200, text=answer)
        return httpx.Response(
            200,
            json=artifact if method == "POST" else {"items": [artifact]},
            request=httpx.Request(method, "http://test" + path),
        )

    # raise_for_status needs the request attached to all mock API responses.
    async def received(method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = await api(method, path, **kwargs)
        response.request = httpx.Request(method, "http://test" + path)
        return response

    runtime.api = AsyncMock(side_effect=received)
    output: dict[str, Any] = {"tasks": [], "checks": {}}
    await runner.exercise(case, runtime, output)
    assert payloads[0] == {"question": "第一轮问题"}
    assert payloads[1] == (
        {"question": "展开刚才内容"} if kind == "memory" else {"question": "展开刚才内容", "conversation_id": "first"}
    )
    assert "PRIVATE_EXPECTED_ANSWER" not in json.dumps(payloads)
    assert runtime.api.await_count == (6 if export else 2)
    assert output["checks"]["conversation_relation"] is True
    if export:
        assert output["checks"]["artifact_hash"] is True
        assert output["checks"]["export_no_model_calls"] is True
    else:
        assert "artifact_hash" not in output["checks"]
    if kind == "memory":
        assert output["checks"]["memory_saved_receipt"] is True


@pytest.mark.anyio
async def test_split_selection_does_not_run_holdout_or_approve_drafts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = []
    for i, split in enumerate(("development", "holdout"), 101):
        case = approved_case()
        case.metadata.case_id = f"B-{i}"
        case.metadata.split = split
        case.metadata.approval = "draft"
        cases.append(case)
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(case.model_dump_json() for case in cases), encoding="utf-8")
    snapshot = tmp_path / "corpus.json"
    save_json(snapshot, corpus().model_dump())
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "runs")
    monkeypatch.setattr(runner, "read_corpus", AsyncMock(return_value=corpus()))
    runtime = Mock(side_effect=AssertionError("Draft labels must not run"))
    monkeypatch.setattr(runner, "AppRuntime", runtime)
    args = argparse.Namespace(
        mode="live",
        cases=path,
        sources=snapshot,
        allow_paid=True,
        ledger=tmp_path / "budget.json",
        max_requests=8,
        case=None,
        split="development",
    )
    result = await runner.run(args)
    assert result["cases"][0]["reason"] == ["labels_not_approved"]
    assert result["cases"][1]["execution_status"] == "not_selected"
    assert result["summary"]["by_split"]["development"]["real_corpus"]["blocked"] == 1
    assert result["summary"]["by_split"]["holdout"]["real_corpus"]["not_selected"] == 1
    runtime.assert_not_called()


def test_reports_keep_development_and_holdout_outcomes_separate(tmp_path: Path) -> None:
    rows = []
    for i, split in enumerate(("development", "holdout"), 101):
        output = {**record(), "case_id": f"B-{i}", "mode": "real_corpus", "split": split}
        output["score"] = score(approved_case(), output)
        rows.append(output)
    result = runner.write_report(tmp_path, rows)
    assert result["summary"]["groups"]["real_corpus"]["planned"] == 2
    assert result["summary"]["by_split"]["development"]["real_corpus"]["pending"] == 1
    assert result["summary"]["by_split"]["holdout"]["real_corpus"]["pending"] == 1


@pytest.mark.anyio
async def test_non_json_post_result_is_unknown_and_is_not_replayed(tmp_path: Path) -> None:
    calls = 0

    async def no_receipt(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(201, text="lost task receipt")

    runtime = AppRuntime(tmp_path)
    runtime.users = [uuid4(), uuid4()]
    output: dict[str, Any] = {"tasks": []}
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_receipt), base_url="http://test") as client:
        runtime.client = client
        with pytest.raises(runner.OutcomeUnknown):
            await runner.create_task(runtime, output, "查询公告")
    assert calls == 1 and output["tasks"] == []
    assert runtime.steps[0]["state"] == "received"


def test_rescoring_retains_independent_review_and_rejects_changed_output(tmp_path: Path) -> None:
    case, output = approved_case(), record()
    output.update(case_id=case.metadata.case_id, mode="real_corpus")
    human = review(case, output)
    runner.apply_reviews(tmp_path, [case], [output], [human])
    assert output["score"]["verdict"] == "pass"
    runner.apply_reviews(tmp_path, [case], [output], [])
    assert output["human_review"]["reviewer"] == "reviewer"
    assert output["score"]["verdict"] == "pass"
    output["artifacts"].append({"answer": "different"})
    runner.apply_reviews(tmp_path, [case], [output], [])
    assert output["score"]["verdict"] == "pending"


@pytest.mark.anyio
async def test_review_only_mode_does_not_recreate_tasks_or_read_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, output = approved_case(), record()
    output.update(case_id=case.metadata.case_id, mode="real_corpus")
    (tmp_path / "cases.jsonl").write_text(case.model_dump_json(), encoding="utf-8")
    save_json(tmp_path / "report.json", {"cases": [output]})
    save_json(tmp_path / "reviews.json", [review(case, output).model_dump()])
    runtime = Mock(side_effect=AssertionError("Never replay application side effects when scoring"))
    database = AsyncMock(side_effect=AssertionError("Scoring is local only"))
    monkeypatch.setattr(runner, "AppRuntime", runtime)
    monkeypatch.setattr(runner, "read_corpus", database)
    args = argparse.Namespace(mode="score", cases=runner.CASES, run_dir=tmp_path, reviews=tmp_path / "reviews.json")
    result = await runner.run(args)
    assert result["cases"][0]["score"]["verdict"] == "pass"
    runtime.assert_not_called()
    database.assert_not_awaited()


@pytest.mark.anyio
async def test_time_budget_blocks_before_sending(tmp_path: Path) -> None:
    send = Mock(side_effect=AssertionError("No send after deadline"))
    transport = BudgetTransport(httpx.MockTransport(send), tmp_path / "budget.json", limit=1, wall_seconds=0)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RequestBudgetExceeded):
            await client.post("https://model.invalid/chat/completions", json={"max_tokens": 10})
    assert transport.records == []
    send.assert_not_called()
