"""rag_tool 的单测。全部离线：DB 会话与 RagService 都被替换，不连 Postgres、不加载模型。

钉住的是「模型传错参数时会发生什么」这一组行为，因为工具参数是 LLM 采样出来的，
不是可信调用方。曾经发生过的真实故障：模型给 group_type 编了个 "news"，
chunk_repo.search 照此加 WHERE，于是稳定返回 0 条；而 0 条和「库里真的没有」
在返回值里长得一模一样，模型于是连换 5 轮关键词、一次没碰真凶，直到撞穿轮次上限。

所以这里分别钉：
1. 非法 group_type 进不到函数体（schema 层就挡住），且合法值出现在模型能看到的
   JSON Schema enum 里 —— 这是唯一能「预防」而非事后补救的一层。
2. 零命中且带过滤器时，note 必须点出「可能是过滤器造成的」，模型才有自纠线索。
3. 命中记录必须带 article_id，否则 read_article_tool 那条链是死的。
"""

import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.agents.agent_tools.rag_tool import MAX_TOP_K, search_rag_tool

HIT: dict[str, Any] = {
    "article_id": 30,
    "title": "Recommended cash acquisition of Tate & Lyle PLC by Ingredion",
    "url": "https://example.com/tate-lyle",
    "site_name": "Tate & Lyle",
    "publish_date": "2026-07-28",
    "chunk_text": "Ingredion has agreed to acquire Tate & Lyle PLC for cash.",
    "score": 0.714,
}


@asynccontextmanager
async def _fake_db():
    yield AsyncMock()


def _patch_rag(hits: list[dict[str, Any]]):
    """把 DB 会话与 RagService 一起替换掉，返回 RagService 的 mock 供断言入参。"""
    service = MagicMock()
    service.query = AsyncMock(return_value=hits)
    return (
        patch("app.agents.agent_tools.rag_tool.get_worker_db_context", _fake_db),
        patch("app.agents.agent_tools.rag_tool.RagService", MagicMock(return_value=service)),
        service,
    )


async def _call(**kwargs: Any) -> dict[str, Any]:
    """跑一次工具，返回解析后的 JSON。默认命中一条。"""
    hits = kwargs.pop("_hits", [HIT])
    db_patch, svc_patch, _ = _patch_rag(hits)
    with db_patch, svc_patch:
        return json.loads(await search_rag_tool.ainvoke(kwargs))


class TestGroupTypeValidation:
    def test_schema_exposes_valid_values_as_enum(self):
        """模型看到的 JSON Schema 必须含 enum —— 它只能在合法值里选，没法自己编。"""
        props = search_rag_tool.args_schema.model_json_schema()["properties"]
        enums = [branch["enum"] for branch in props["group_type"]["anyOf"] if "enum" in branch]

        assert enums == [["competitor", "regulatory"]]

    @pytest.mark.anyio
    async def test_invalid_group_type_never_reaches_the_service(self):
        """非法值在 schema 层就被挡下，不会变成一次「合法但零命中」的检索。"""
        db_patch, svc_patch, service = _patch_rag([HIT])
        with db_patch, svc_patch, pytest.raises(ValidationError):
            await search_rag_tool.ainvoke({"query": "Tate & Lyle", "group_type": "news"})

        service.query.assert_not_awaited()

    @pytest.mark.anyio
    async def test_valid_group_type_is_forwarded(self):
        db_patch, svc_patch, service = _patch_rag([HIT])
        with db_patch, svc_patch:
            await search_rag_tool.ainvoke({"query": "Tate & Lyle", "group_type": "competitor"})

        assert service.query.await_args is not None
        assert service.query.await_args.kwargs["group_type"] == "competitor"

    @pytest.mark.anyio
    async def test_omitting_group_type_searches_whole_kb(self):
        """不传 = 全库检索，是默认且通常正确的选择；别默默替模型补一个过滤器。"""
        db_patch, svc_patch, service = _patch_rag([HIT])
        with db_patch, svc_patch:
            await search_rag_tool.ainvoke({"query": "Tate & Lyle"})

        assert service.query.await_args is not None
        assert service.query.await_args.kwargs["group_type"] is None


class TestTopKBounds:
    @pytest.mark.anyio
    @pytest.mark.parametrize("bad", [0, -1, MAX_TOP_K + 1, 1000])
    async def test_out_of_range_top_k_rejected(self, bad: int):
        """无上界时模型会自己往上加，而每条命中都要进 messages、每轮 react 重付 token。"""
        with pytest.raises(ValidationError):
            await search_rag_tool.ainvoke({"query": "x", "top_k": bad})

    @pytest.mark.anyio
    @pytest.mark.parametrize("ok", [1, 5, MAX_TOP_K])
    async def test_in_range_top_k_accepted(self, ok: int):
        payload = await _call(query="x", top_k=ok)

        assert payload["top_k"] == ok


class TestOutputShape:
    @pytest.mark.anyio
    async def test_record_carries_article_id(self):
        """漏掉 article_id，read_article_tool 就没有可传的 id —— 读全文那条链直接断掉。"""
        payload = await _call(query="Tate & Lyle")

        assert payload["results"][0]["article_id"] == 30

    @pytest.mark.anyio
    async def test_chunk_text_becomes_content_and_publish_date_becomes_published_at(self):
        record = (await _call(query="Tate & Lyle"))["results"][0]

        assert record["content"] == HIT["chunk_text"]
        assert record["published_at"] == HIT["publish_date"]

    @pytest.mark.anyio
    async def test_service_failure_returns_zero_results_not_error_text(self):
        """报错文案漏成 results 里的一条，就会被下游当成一条可引用证据。"""
        service = MagicMock()
        service.query = AsyncMock(side_effect=RuntimeError("connection refused"))
        with (
            patch("app.agents.agent_tools.rag_tool.get_worker_db_context", _fake_db),
            patch("app.agents.agent_tools.rag_tool.RagService", MagicMock(return_value=service)),
        ):
            payload = json.loads(await search_rag_tool.ainvoke({"query": "x"}))

        assert payload["results"] == []
        assert "RuntimeError" in payload["error"]


class TestEmptyResultNote:
    @pytest.mark.anyio
    async def test_zero_hits_with_filter_points_at_the_filter(self):
        """带过滤器的零命中要明说可能是过滤器造成的，否则模型只会反复换关键词。"""
        payload = await _call(query="x", group_type="regulatory", _hits=[])

        assert payload["total"] == 0
        assert "group_type" in payload["note"]
        assert "去掉" in payload["note"]

    @pytest.mark.anyio
    async def test_zero_hits_without_filter_does_not_blame_a_filter(self):
        payload = await _call(query="x", _hits=[])

        assert "全库检索" in payload["note"]
        assert "去掉 group_type" not in payload["note"]

    @pytest.mark.anyio
    async def test_hits_note_mentions_how_to_read_full_text(self):
        payload = await _call(query="x")

        assert "read_article_tool" in payload["note"]
