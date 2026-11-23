"""Small, reproducible task and trajectory metrics. No invented LLM costs."""

from typing import Any

from app.agents.evidence import citations


def score_case(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    answer = str(run.get("answer") or "")
    evidence = run.get("evidence") or []
    actual_ids = {int(item["article_id"]) for item in evidence if item.get("article_id") is not None}
    expected_ids = {int(item) for item in case.get("expected_article_ids", [])}
    required = [str(term).casefold() for term in case.get("required_terms", [])]
    source_recall = len(actual_ids & expected_ids) / len(expected_ids) if expected_ids else None
    terms_present = all(term in answer.casefold() for term in required)
    known_citations = {str(item["id"]) for item in evidence if item.get("id")}
    cited = citations(answer)
    citation_valid = bool(cited) and all(item in known_citations for item in cited)
    if run.get("version") == "new" and run.get("review"):
        citation_valid = citation_valid and bool(run["review"].get("approved"))
    complete = run.get("version") != "new" or run.get("termination") == "completed"
    return {
        "task_success": bool(answer) and complete and terms_present and expected_ids <= actual_ids,
        "source_recall": source_recall,
        "citation_valid": citation_valid if run.get("version") == "new" else None,
        "tool_calls": int(run.get("tool_calls", 0)),
        "model_calls": int(run.get("model_calls", 0)),
        "latency_seconds": float(run.get("latency_seconds", 0.0)),
        "termination": run.get("termination"),
        "cost_usd": None,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    evaluable = [item["citation_valid"] for item in rows if item["citation_valid"] is not None]
    source_scores = [item["source_recall"] for item in rows if item["source_recall"] is not None]
    return {
        "n": n,
        "task_success_rate": sum(item["task_success"] for item in rows) / n,
        "mean_source_recall": sum(source_scores) / len(source_scores) if source_scores else None,
        "source_recall_n": len(source_scores),
        "citation_valid_rate": sum(evaluable) / len(evaluable) if evaluable else None,
        "mean_tool_calls": sum(item["tool_calls"] for item in rows) / n,
        "mean_model_calls": sum(item["model_calls"] for item in rows) / n,
        "mean_latency_seconds": sum(item["latency_seconds"] for item in rows) / n,
        "cost_usd": None,
    }
