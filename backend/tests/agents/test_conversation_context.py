"""Complete-turn history retains references without promoting answers to evidence."""

from uuid import uuid4

import pytest

from app.agents.context.conversation import select_history
from app.core.config import settings
from app.core.exceptions import BadRequestError


def test_history_is_chronological_and_preserves_source_ids() -> None:
    older, newer = uuid4(), uuid4()
    source = {"id": "E-123456789abc", "content": "原文", "url": "https://example.org"}
    history, evidence = select_history(
        [
            {
                "task_id": newer,
                "question": "第二轮",
                "answer": "展开分析[E-123456789abc]",
                "evidence": [source],
                "status": "completed",
            },
            {
                "task_id": older,
                "question": "第一轮",
                "answer": "列出两条[E-123456789abc]",
                "evidence": [source],
                "status": "completed",
            },
        ]
    )
    assert history.index("第一轮") < history.index("第二轮")
    assert len(evidence) == 1
    assert evidence[0]["source_task_id"] == str(newer)
    assert "历史答案本身不是事实证据" in history


def test_oversized_latest_history_is_rejected_not_silently_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AGENT_HISTORY_TOKENS", 10)
    with pytest.raises(BadRequestError):
        select_history(
            [
                {
                    "task_id": uuid4(),
                    "question": "问题",
                    "answer": "长答案" * 100,
                    "evidence": [],
                    "status": "completed",
                }
            ]
        )


def test_failed_history_is_not_reused() -> None:
    assert select_history([{"status": "failed"}]) == ("", [])
