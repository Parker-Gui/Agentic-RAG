"""爬虫 CLI：抓取后默认写库，``--scrape-only`` 只抓不写。

全部离线：不起浏览器、不连数据库。抓取被 stub，断言的是抓完之后 hits 的去向。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.crawler.delivery.cli import main, parse_args, persist


@pytest.fixture
def fake_hit() -> dict:
    return {
        "site": "Acme Bio",
        "title": "Acme launches a thing",
        "url": "https://acme.example/news/1",
        "publish_date": "2026-08-20",
        "body_text": "Acme announced a thing today.",
        "external": False,
        "group": "competitor",
    }


class TestParseArgs:
    def test_default_is_not_scrape_only(self):
        assert parse_args([]).scrape_only is False

    def test_scrape_only_flag_sets_it(self):
        assert parse_args(["--scrape-only"]).scrape_only is True


class TestPersist:
    def test_returns_true_after_writing(self):
        # persist 在调用时才解析 save_hits_to_db，所以要 patch persistence 模块本身。
        save = AsyncMock(return_value=1)
        with patch("app.services.crawler.delivery.persistence.save_hits_to_db", save):
            assert persist([{"url": "https://x.com/a", "title": "a"}]) is True

        save.assert_awaited_once()

    def test_db_failure_is_reported_not_raised(self):
        with patch(
            "app.services.crawler.delivery.persistence.save_hits_to_db",
            AsyncMock(side_effect=OSError("connection refused")),
        ):
            assert persist([{"url": "https://x.com/a", "title": "a"}]) is False


class TestMain:
    """端到端接线：给定命令行，hits 抓完之后的去向。抓取被 stub。"""

    def test_default_scrapes_and_persists(self, fake_hit: dict):
        save = AsyncMock(return_value=1)
        with (
            patch(
                "app.services.crawler.delivery.cli.scan_all_sites",
                return_value=MagicMock(hits=[fake_hit]),
            ),
            patch("app.services.crawler.delivery.persistence.save_hits_to_db", save),
        ):
            assert main([]) == 0

        assert save.await_args is not None
        assert save.await_args.args == ([fake_hit],)

    def test_db_failure_exits_1(self, fake_hit: dict):
        with (
            patch(
                "app.services.crawler.delivery.cli.scan_all_sites",
                return_value=MagicMock(hits=[fake_hit]),
            ),
            patch(
                "app.services.crawler.delivery.persistence.save_hits_to_db",
                AsyncMock(side_effect=OSError("connection refused")),
            ),
        ):
            assert main([]) == 1

    def test_scrape_only_does_not_persist(self, fake_hit: dict):
        save = AsyncMock()
        with (
            patch(
                "app.services.crawler.delivery.cli.scan_all_sites",
                return_value=MagicMock(hits=[fake_hit]),
            ),
            patch("app.services.crawler.delivery.persistence.save_hits_to_db", save),
        ):
            assert main(["--scrape-only"]) == 0

        save.assert_not_awaited()

    def test_no_hits_skips_persist(self):
        save = AsyncMock()
        with (
            patch(
                "app.services.crawler.delivery.cli.scan_all_sites",
                return_value=MagicMock(hits=[]),
            ),
            patch("app.services.crawler.delivery.persistence.save_hits_to_db", save),
        ):
            assert main([]) == 0

        save.assert_not_awaited()
