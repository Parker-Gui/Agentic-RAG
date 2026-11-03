"""tavily_tool 的单测。全部离线：Tavily 客户端一律被替换掉，不发任何外部请求。

这里钉住的是两件事，别的都次要：

1. 返回值永远是 `{"results": [...]}` 形状的 JSON —— 包括缺 key、格式错、
   接口抛异常这些失败路径。失败时报错要落在 error 字段里，绝不能漏成一句
   冒充正文的字符串（见 test_exception_is_caught_and_returned_as_error）。
2. Tavily 的 published_date 必须映射成结果里的 published_at。
   漏了这一步不会报错，只会让所有结果的发布时间静默变成 None。
"""

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.agent_tools import tavily_tool, tools
from app.agents.agent_tools.tavily_tool import (
    MAX_EXTRACT_URLS,
    _normalize_published_at,
    tavily_extract,
    tavily_search,
    tavily_tools,
)
from app.core.config import settings

SEARCH_RESPONSE: dict[str, Any] = {
    "query": "甜味剂 出口",
    "results": [
        {
            "title": "甜味剂行业月报",
            "url": "https://example.cn/report",
            "content": "2026 年 8 月出口量同比增长 12.3%。",
            "published_date": "2026-08-01",
            "score": 0.81,
        }
    ],
}


def fake_client(**methods: AsyncMock) -> AsyncMock:
    """替身客户端。只挂上被测方法，其余属性不该被碰到。"""
    client = AsyncMock()
    for name, method in methods.items():
        setattr(client, name, method)
    return client


@pytest.fixture
def with_key():
    """默认配置里没有 key（也不该依赖开发机的 .env），测试里显式给一把假的。"""
    with patch.object(settings, "TAVILY_API_KEY", "tvly-test-key"):
        yield


@pytest.fixture(autouse=True)
def no_cached_client():
    """清掉惰性单例，避免上一个测试的替身漏到下一个测试。"""
    tavily_tool._client = None
    yield
    tavily_tool._client = None


class TestRegistration:
    def test_tavily_tools_are_registered_in_the_agent_toolset(self):
        """不挂进 agent 的 tools 聚合，接入了也等于没接 —— 模型看不到这两个工具。"""
        names = [t.name for t in tools]
        assert [t.name for t in tavily_tools] == ["tavily_search", "tavily_extract"]
        assert {t.name for t in tavily_tools} <= set(names)

    def test_placeholder_scraper_is_gone(self):
        """占位爬虫早该由 tavily_extract 取代，不能还留在工具集里。"""
        assert "web_scraper_tool" not in [t.name for t in tools]

    def test_model_facing_args(self):
        assert set(tavily_search.args) == {"query", "topic", "time_range", "start_date", "end_date", "max_results"}
        assert set(tavily_extract.args) == {"urls"}


class TestNormalizePublishedAt:
    """发布时间必须落成 YYYY-MM-DD —— 下游的日期核对只认这类写法。"""

    def test_rfc_1123_from_news_search(self):
        # topic="news" 实际返回的就是这个格式
        assert _normalize_published_at("Wed, 05 Aug 2026 07:00:00 GMT") == "2026-08-05"

    def test_iso_datetime_is_cut_to_the_date(self):
        assert _normalize_published_at("2026-08-01T00:00:00Z") == "2026-08-01"

    def test_plain_iso_date_is_kept(self):
        assert _normalize_published_at("2026-08-01") == "2026-08-01"

    def test_missing_becomes_none(self):
        assert _normalize_published_at(None) is None
        assert _normalize_published_at("") is None

    def test_unparsable_is_kept_verbatim(self):
        """解析不了也别丢 —— 下游至少还能自己读一眼。"""
        assert _normalize_published_at("上周三") == "上周三"


class TestSearch:
    @pytest.mark.anyio
    async def test_maps_tavily_fields_onto_result_fields(self, with_key: None):
        search = AsyncMock(return_value=SEARCH_RESPONSE)
        with patch.object(tavily_tool, "_get_client", return_value=fake_client(search=search)):
            payload = json.loads(await tavily_search.ainvoke({"query": "甜味剂 出口", "topic": "news"}))

        result = payload["results"][0]
        assert payload["total"] == 1
        assert result["published_at"] == "2026-08-01"  # Tavily 给的是 published_date
        assert result["url"] == "https://example.cn/report"
        assert result["score"] == 0.81
        assert result["truncated"] is False

    @pytest.mark.anyio
    async def test_output_json_carries_title_and_published_at(self, with_key: None):
        """JSON 契约：每条结果都带上 title 和归一化后的 published_at。"""
        search = AsyncMock(return_value=SEARCH_RESPONSE)
        with patch.object(tavily_tool, "_get_client", return_value=fake_client(search=search)):
            payload = json.loads(await tavily_search.ainvoke({"query": "甜味剂 出口"}))

        result = payload["results"][0]
        assert result["title"] == "甜味剂行业月报"
        assert result["published_at"] == "2026-08-01"

    @pytest.mark.anyio
    async def test_raw_content_wins_over_snippet_and_gets_clipped(self, with_key: None):
        long_text = "甜" * 50
        search = AsyncMock(return_value={"results": [{"content": "摘要", "raw_content": long_text}]})
        with (
            patch.object(settings, "TAVILY_MAX_CONTENT_CHARS", 20),
            patch.object(tavily_tool, "_get_client", return_value=fake_client(search=search)),
        ):
            payload = json.loads(await tavily_search.ainvoke({"query": "甜味剂"}))

        result = payload["results"][0]
        assert result["content"] == "甜" * 20
        assert result["truncated"] is True

    @pytest.mark.anyio
    async def test_max_results_is_clamped_to_the_configured_cap(self, with_key: None):
        search = AsyncMock(return_value={"results": []})
        with (
            patch.object(settings, "TAVILY_MAX_RESULTS", 5),
            patch.object(tavily_tool, "_get_client", return_value=fake_client(search=search)),
        ):
            await tavily_search.ainvoke({"query": "甜味剂", "max_results": 20})

        assert search.await_args is not None
        assert search.await_args.kwargs["max_results"] == 5

    @pytest.mark.anyio
    async def test_empty_results_say_so_instead_of_looking_like_evidence(self, with_key: None):
        search = AsyncMock(return_value={"results": []})
        with patch.object(tavily_tool, "_get_client", return_value=fake_client(search=search)):
            raw = await tavily_search.ainvoke({"query": "查不到的东西"})

        assert json.loads(raw)["results"] == []


class TestSearchFailures:
    @pytest.mark.anyio
    async def test_missing_api_key_returns_empty_results_without_calling_out(self):
        search = AsyncMock()
        with (
            patch.object(settings, "TAVILY_API_KEY", None),
            patch.object(tavily_tool, "_get_client", return_value=fake_client(search=search)),
        ):
            payload = json.loads(await tavily_search.ainvoke({"query": "甜味剂"}))

        assert payload["results"] == []
        assert "TAVILY_API_KEY" in payload["error"]
        search.assert_not_awaited()

    @pytest.mark.anyio
    async def test_bad_date_format_is_rejected_locally(self, with_key: None):
        """本地挡住，省一次注定 400 的请求，也让模型看到「格式错」而不是「接口错」。"""
        search = AsyncMock()
        with patch.object(tavily_tool, "_get_client", return_value=fake_client(search=search)):
            payload = json.loads(await tavily_search.ainvoke({"query": "甜味剂", "start_date": "2026年5月1日"}))

        assert payload["results"] == []
        assert "YYYY-MM-DD" in payload["error"]
        search.assert_not_awaited()

    @pytest.mark.anyio
    async def test_exception_is_caught_and_returned_as_error_not_content(self, with_key: None):
        """接口抛异常时，报错落在 error 字段、results 保持空 —— 不能冒充成一条正文。"""
        search = AsyncMock(side_effect=RuntimeError("429 Too Many Requests"))
        with patch.object(tavily_tool, "_get_client", return_value=fake_client(search=search)):
            raw = await tavily_search.ainvoke({"query": "甜味剂"})

        payload = json.loads(raw)
        assert payload["results"] == []
        assert "429" in payload["error"]


class TestExtract:
    @pytest.mark.anyio
    async def test_falls_back_to_url_as_title(self, with_key: None):
        """/extract 不返回标题，留空会得到一串分不清彼此的空结果。"""
        extract = AsyncMock(return_value={"results": [{"url": "https://example.cn/a", "raw_content": "正文"}]})
        with patch.object(tavily_tool, "_get_client", return_value=fake_client(extract=extract)):
            payload = json.loads(await tavily_extract.ainvoke({"urls": ["https://example.cn/a"]}))

        assert payload["results"][0]["title"] == "https://example.cn/a"
        assert payload["results"][0]["content"] == "正文"

    @pytest.mark.anyio
    async def test_extra_urls_are_dropped_but_reported(self, with_key: None):
        urls = [f"https://example.cn/{i}" for i in range(MAX_EXTRACT_URLS + 2)]
        extract = AsyncMock(return_value={"results": []})
        with patch.object(tavily_tool, "_get_client", return_value=fake_client(extract=extract)):
            payload = json.loads(await tavily_extract.ainvoke({"urls": urls}))

        assert extract.await_args is not None
        assert len(extract.await_args.kwargs["urls"]) == MAX_EXTRACT_URLS
        assert payload["dropped_urls"] == urls[MAX_EXTRACT_URLS:]

    @pytest.mark.anyio
    async def test_failed_urls_are_surfaced(self, with_key: None):
        extract = AsyncMock(
            return_value={"results": [], "failed_results": [{"url": "https://example.cn/x", "error": "403"}]}
        )
        with patch.object(tavily_tool, "_get_client", return_value=fake_client(extract=extract)):
            payload = json.loads(await tavily_extract.ainvoke({"urls": ["https://example.cn/x"]}))

        assert payload["failed"][0]["error"] == "403"
        assert payload["results"] == []

    @pytest.mark.anyio
    async def test_no_urls_returns_empty_results(self, with_key: None):
        extract = AsyncMock()
        with patch.object(tavily_tool, "_get_client", return_value=fake_client(extract=extract)):
            payload = json.loads(await tavily_extract.ainvoke({"urls": []}))

        assert payload["results"] == []
        extract.assert_not_awaited()
