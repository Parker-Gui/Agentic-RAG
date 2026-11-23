"""Task metrics count observed facts and sources, not guessed model quality."""

from eval.agent_metrics import aggregate, score_case


def test_task_success_requires_sources_and_terms() -> None:
    case = {"expected_article_ids": [23, 96], "required_terms": ["GRAS", "Nukoko"]}
    run = {
        "version": "new",
        "answer": "GRAS 和 Nukoko [E-aaaaaaaaaaaa]",
        "evidence": [{"id": "E-aaaaaaaaaaaa", "article_id": 23}],
        "tool_calls": 1,
        "model_calls": 3,
        "latency_seconds": 2.0,
    }
    score = score_case(case, run)
    assert score["task_success"] is False
    assert score["source_recall"] == 0.5
    assert score["citation_valid"] is True


def test_aggregate_marks_unmeasured_cost_and_legacy_citations() -> None:
    row = score_case(
        {"expected_article_ids": [1], "required_terms": []},
        {"version": "legacy", "answer": "结果", "evidence": [{"article_id": 1}], "latency_seconds": 1.0},
    )
    summary = aggregate([row])
    assert summary["citation_valid_rate"] is None
    assert summary["cost_usd"] is None


def test_missing_source_reference_is_excluded_from_denominator() -> None:
    missing = score_case({}, {"answer": "No material", "version": "new"})
    measured = score_case({"expected_article_ids": [1, 2]}, {"evidence": [{"article_id": 1}]})
    assert missing["source_recall"] is None
    summary = aggregate([missing, measured])
    assert summary["source_recall_n"] == 1
    assert summary["mean_source_recall"] == 0.5
