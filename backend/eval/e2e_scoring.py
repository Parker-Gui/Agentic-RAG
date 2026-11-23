"""Independent local checks: incomplete runs and unreviewed semantics never become passes."""

from typing import Any, Literal

from pydantic import BaseModel

from eval.e2e_data import E2ECase, fingerprint

Verdict = Literal["pass", "fail", "pending"]


class HumanReview(BaseModel):
    run_id: str
    case_id: str
    case_sha256: str
    artifacts_sha256: str
    reviewer: str
    note: str
    checks: dict[str, Literal["pass", "fail"]]


def score(case: E2ECase, record: dict[str, Any], review: HumanReview | None = None) -> dict[str, Any]:
    checks = dict(record.get("checks", {}))
    manual: dict[str, str] = dict.fromkeys(case.expected_output.manual_checks, "pending")
    review_error = None
    if review is not None:
        if (
            review.run_id != record["run_id"]
            or review.case_id != case.metadata.case_id
            or review.case_sha256 != fingerprint(case.model_dump())
            or review.artifacts_sha256 != fingerprint(record.get("artifacts", []))
            or not review.reviewer.strip()
            or not review.note.strip()
        ):
            review_error = "review_identity_or_artifacts_mismatch"
        else:
            manual.update({key: value for key, value in review.checks.items() if key in manual})
    verdict: Verdict = "pending"
    if record.get("execution_status") == "executed":
        if any(value is False for value in checks.values()) or "fail" in manual.values():
            verdict = "fail"
        elif (
            any(value is True for value in checks.values())
            and all(value is True or value is None for value in checks.values())
            and all(value == "pass" for value in manual.values())
            and not review_error
        ):
            verdict = "pass"
    return {
        "verdict": verdict,
        "checks": checks,
        "manual_checks": manual,
        "review_error": review_error,
        "negative_control_verified": case.expected_output.negative_control
        and verdict == "fail"
        and manual.get("goal_completion") == "fail"
        and not any(value is False for value in checks.values())
        and not review_error,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for mode in ("real_corpus", "fixture"):
        rows = [row for row in records if row["mode"] == mode]
        groups[mode] = {
            "planned": len(rows),
            "executed": sum(row["execution_status"] == "executed" for row in rows),
            "blocked": sum(row["execution_status"] == "blocked" for row in rows),
            "outcome_unknown": sum(row["execution_status"] == "outcome_unknown" for row in rows),
            "not_selected": sum(row["execution_status"] == "not_selected" for row in rows),
            "failed_execution": sum(row["execution_status"] == "failed" for row in rows),
            "pass": sum(row["score"]["verdict"] == "pass" for row in rows),
            "fail": sum(row["score"]["verdict"] == "fail" for row in rows),
            "pending": sum(row["score"]["verdict"] == "pending" for row in rows),
            "negative_controls_verified": sum(row["score"]["negative_control_verified"] for row in rows),
        }
    return {"groups": groups, "cost_usd": None}
