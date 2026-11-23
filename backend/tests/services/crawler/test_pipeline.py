"""Watermarks follow committed data and HTTP-only scans do not start Chrome."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.services.crawler.common.models import ArticleContent, Candidate, Outcome
from app.services.crawler.orchestration import pipeline


def test_http_only_scan_does_not_start_browser_or_save_watermark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    browser, save = Mock(side_effect=AssertionError("Browser should not be started")), Mock()
    monkeypatch.setattr(
        pipeline, "load_sites", lambda: [{"name": "sample", "news_url": "https://example.org", "fetch_mode": "http"}]
    )
    monkeypatch.setattr(pipeline, "load_last_times", lambda: {})
    monkeypatch.setattr(pipeline, "setup_logging", lambda: (tmp_path / "run.log", tmp_path / "summary.log"))
    monkeypatch.setattr(pipeline, "write_summary", lambda *args: None)
    monkeypatch.setattr(pipeline, "browser_session", browser)
    monkeypatch.setattr(pipeline, "save_last_times", save)
    monkeypatch.setattr(
        pipeline,
        "scan_site",
        lambda *args, **kwargs: pipeline.SiteScan(
            Outcome("sample", "ok", "one"), [{"title": "test"}], datetime(2026, 1, 1)
        ),
    )
    run = pipeline.scan_all_sites()
    assert run.next_times == {"sample": "2026-01-01 00:00:00"}
    browser.assert_not_called()
    save.assert_not_called()


def test_acknowledgement_merges_watermarks_without_going_backwards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    save = Mock()
    monkeypatch.setattr(pipeline, "BASELINE_OVERRIDE", None)
    monkeypatch.setattr(
        pipeline, "load_last_times", lambda: {"newer": "2026-02-01 00:00:00", "other": "2025-01-01 00:00:00"}
    )
    monkeypatch.setattr(pipeline, "save_last_times", save)
    run = pipeline.MonitorRun(
        [],
        [],
        {},
        tmp_path / "run",
        tmp_path / "summary",
        {"newer": "2026-01-01 00:00:00", "sample": "2026-01-02 00:00:00"},
    )
    pipeline.acknowledge_run(run)
    save.assert_called_once_with(
        {"newer": "2026-02-01 00:00:00", "other": "2025-01-01 00:00:00", "sample": "2026-01-02 00:00:00"}
    )


def test_failed_body_stops_before_advancing_past_that_article() -> None:
    candidates = [(datetime(2026, 1, i), Candidate(f"Article {i}", f"https://example.org/{i}")) for i in (1, 2, 3)]

    def fetch(candidate: Candidate) -> ArticleContent:
        return ArticleContent(body_text="real body" if candidate.url.endswith("/1") else "")

    hits, latest = pipeline._open_dated(candidates, {"name": "sample"}, fetch)
    assert len(hits) == 1
    assert latest == datetime(2026, 1, 1)
