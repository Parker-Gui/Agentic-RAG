"""Tavily 外部检索工具 —— ori_tools 里唯一真的会出网取信息的部分。

改这个文件之前先读完这三条，它们都不是风格问题：

1. 返回值必须是 `{"results": [...]}` 形状的 JSON 字符串。
   critic_tool._parse_tool_content 就是按这个形状把 ToolMessage 还原成 Evidence 的
   （逐键取 title / url / published_at / content / score）。返回散文、返回裸列表、
   或者让异常冒出去由 ToolNode 兜成一句报错字符串，都会被解析成「一条内容是该报错
   文案的证据」塞进证据集 —— 裁判接着就拿报错文案去跟答案对账。所以这里每条路径都
   返回 JSON，取不到东西时 results 给空列表（空列表解析出来就是 0 条证据）。

2. 客户端惰性创建。模块级 `AsyncTavilyClient()` 在缺 key 时会把 import 打挂，
   连带 graph、`langgraph dev`、整个测试套一起起不来；而且它在 __init__ 里就建好了
   httpx.AsyncClient 连接池，import 期建等于把池绑到建它的那个 event loop 上。

3. 正文必须截断。单页 raw_content 动辄几万字，一旦进了 messages，之后每一轮
   react / critic 都要重付一遍这份 token，而裁判要核对的只是其中几句。
"""

import json
import logging
import re
from email.utils import parsedate_to_datetime
from typing import Any, Literal

from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from app.core.config import settings

logger = logging.getLogger(__name__)

# tavily_extract 单次接受的 URL 上限。超出的部分不静默丢弃，会回报在 dropped_urls 里。
MAX_EXTRACT_URLS = 5

# start_date / end_date 只收 Tavily 认的 YYYY-MM-DD。本地先挡一道：模型很爱写
# 「5.1」「2026年5月1日」，直接发出去换回来的是一句 BadRequestError，
# 模型看不出是自己格式写错了，只会原样重试。
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_NO_EVIDENCE_NOTE = "本次检索没有取到任何结果，即没有任何可引用证据；不要凭记忆补足，换关键词或换时间范围再试。"

_client: AsyncTavilyClient | None = None


def _get_client() -> AsyncTavilyClient:
    """惰性单例。首次调用发生在图运行时，连接池因此绑在正确的 event loop 上。

    单线程 asyncio 下判空和赋值之间没有 await，所以不需要锁。
    """
    global _client
    if _client is None:
        _client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
    return _client


def _dump(payload: dict[str, Any]) -> str:
    """工具返回值统一成 JSON 字符串 —— ToolMessage.content 只吃字符串。"""
    return json.dumps(payload, ensure_ascii=False)


def _fail(context: dict[str, Any], reason: str) -> str:
    """失败也要保持 `{"results": [...]}` 契约，理由见模块 docstring 第 1 条。"""
    return _dump({**context, "results": [], "error": reason, "note": _NO_EVIDENCE_NOTE})


def _clip(text: str) -> tuple[str, bool]:
    """按 TAVILY_MAX_CONTENT_CHARS 截断，同时告诉调用方截没截过。"""
    limit = settings.TAVILY_MAX_CONTENT_CHARS
    return text[:limit].strip(), len(text) > limit


def _normalize_published_at(value: Any) -> str | None:
    """发布时间统一折成 YYYY-MM-DD。

    Tavily 的 published_date 有两种写法：ISO 8601，以及 news 检索常见的 RFC 1123
    （"Wed, 05 Aug 2026 07:00:00 GMT"）。后者原样留着不报错，但 critic_tool 的日期
    核对只认 2026-08-05 / 2026年8月5日 这类写法，模型一旦引用发布时间就会被判成
    「日期在证据中找不到」。实在解析不了的按原样留着，不静默丢信息。
    """
    if not value:
        return None
    text = str(value).strip()
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError):
        pass
    if _ISO_DATE_RE.match(text[:10]):  # 2026-08-01T00:00:00Z → 2026-08-01
        return text[:10]
    return text


def _normalize_search_result(record: dict[str, Any]) -> dict[str, Any]:
    """一条 Tavily 搜索结果 → 一条 Evidence 能直接吃下的记录。

    键名不能照抄 Tavily：它给的是 published_date，Evidence 读的是 published_at，
    对不上的话所有证据的发布时间都会是 None —— 情报场景里时间失效等于证据失效。
    """
    content, truncated = _clip(str(record.get("raw_content") or record.get("content") or ""))
    return {
        "title": str(record.get("title") or ""),
        "url": record.get("url"),
        # Tavily 只在 topic="news" 时给 published_date，general 检索这里就是 None
        "published_at": _normalize_published_at(record.get("published_date")),
        "content": content,
        "truncated": truncated,
        "score": record.get("score"),
    }


def _normalize_extract_result(record: dict[str, Any]) -> dict[str, Any]:
    """一条 Tavily 正文抓取结果 → Evidence 记录。

    /extract 不返回标题，用 URL 顶上：裁判 list_evidence 时看到的是标题列，
    留空会得到一串分不清彼此的空证据。
    """
    url = record.get("url")
    content, truncated = _clip(str(record.get("raw_content") or record.get("content") or ""))
    return {
        "title": str(record.get("title") or url or ""),
        "url": url,
        "published_at": _normalize_published_at(record.get("published_date")),
        "content": content,
        "truncated": truncated,
        "score": None,
    }


@tool(
    description=(
        "联网搜索外部公开网页与新闻。按关键词检索，返回每条结果的标题、URL、发布时间与正文摘要，"
        "用于获取模型自身不掌握的外部事实；需要时间范围时用 time_range 或 start_date/end_date"
    )
)
async def tavily_search(
    query: str,
    topic: Literal["general", "news"] = "general",
    time_range: Literal["day", "week", "month", "year"] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_results: int = 5,
) -> str:
    """Tavily 联网搜索

    Args:
        query (str): 检索词，用具体的关键词组合，不要传整段问题
        topic (str): general 通用网页；news 新闻（只有 news 会带发布时间）
        time_range (str | None): 相对时间范围 day/week/month/year，与 start_date 二选一
        start_date (str | None): 起始日期，必须是 YYYY-MM-DD
        end_date (str | None): 截止日期，必须是 YYYY-MM-DD
        max_results (int): 期望返回条数，上限由 TAVILY_MAX_RESULTS 控制

    Returns:
        str: JSON，含 results（title/url/published_at/content/score）与 total
    """
    context: dict[str, Any] = {"query": query, "topic": topic}

    if not settings.TAVILY_API_KEY:
        logger.error("[tavily] TAVILY_API_KEY 未配置，检索被跳过")
        return _fail(context, "TAVILY_API_KEY 未配置，无法发起外部检索")

    for label, value in (("start_date", start_date), ("end_date", end_date)):
        if value and not _ISO_DATE_RE.match(value):
            return _fail(context, f"{label}={value!r} 格式不对，必须是 YYYY-MM-DD，例如 2026-05-01")

    # 夹到配置上限：模型经常张口要 20 条，那是白烧 credit 又把上下文顶爆
    wanted = max(1, min(max_results, settings.TAVILY_MAX_RESULTS))

    # 只放真正有值的键。tavily 的可选参数标注成 `str = None`（标注与默认值不一致），
    # 显式传 None 过不了 ty，而它自己在 _search 里本来也会把 None 全过滤掉。
    params: dict[str, Any] = {
        "query": query,
        "topic": topic,
        "search_depth": settings.TAVILY_SEARCH_DEPTH,
        "max_results": wanted,
        "timeout": settings.TAVILY_TIMEOUT,
    }
    if time_range:
        params["time_range"] = time_range
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if settings.TAVILY_INCLUDE_RAW_CONTENT:
        params["include_raw_content"] = "markdown"

    try:
        response = await _get_client().search(**params)
    # 兜到 Exception 是刻意的：tavily 的异常没有公共基类（UsageLimitExceeded /
    # InvalidAPIKey / BadRequest / Forbidden / Timeout 各自继承 Exception），
    # 漏一个就会让报错文案经 ToolNode 变成一条「证据」。见模块 docstring 第 1 条。
    except Exception as exc:
        logger.warning("[tavily] 检索失败 query=%r: %s", query, exc)
        return _fail(context, f"检索失败：{type(exc).__name__}: {exc}")

    raw_results = response.get("results") or []
    results = [_normalize_search_result(item) for item in raw_results if isinstance(item, dict)]
    logger.info("[tavily] query=%r topic=%s 命中 %d 条", query, topic, len(results))

    payload: dict[str, Any] = {**context, "total": len(results), "results": results}
    if not results:
        payload["note"] = _NO_EVIDENCE_NOTE
    else:
        payload["note"] = "只能引用这里出现的内容；正文被截断（truncated=true）时用 tavily_extract 读全文再下判断。"
    return _dump(payload)


@tool(
    description=(
        "抓取指定 URL 的网页正文。用于把搜索命中的页面读全（搜索结果的正文是摘要，可能被截断），"
        f"一次最多 {MAX_EXTRACT_URLS} 个 URL"
    )
)
async def tavily_extract(urls: list[str]) -> str:
    """Tavily 网页正文抓取

    Args:
        urls (list[str]): 要抓取的 URL 列表，必须是搜索结果里出现过的真实链接

    Returns:
        str: JSON，含 results（title/url/content）、failed（抓取失败的 URL）与 dropped_urls
    """
    wanted = [url for url in urls if url]
    dropped = wanted[MAX_EXTRACT_URLS:]
    wanted = wanted[:MAX_EXTRACT_URLS]
    context: dict[str, Any] = {"requested_urls": wanted, "dropped_urls": dropped}

    if not wanted:
        return _fail(context, "没有传入任何 URL")

    if not settings.TAVILY_API_KEY:
        logger.error("[tavily] TAVILY_API_KEY 未配置，正文抓取被跳过")
        return _fail(context, "TAVILY_API_KEY 未配置，无法抓取网页正文")

    try:
        response = await _get_client().extract(
            urls=wanted,
            extract_depth=settings.TAVILY_SEARCH_DEPTH,
            format="markdown",
            timeout=settings.TAVILY_TIMEOUT,
        )
    except Exception as exc:  # 同上：保住 JSON 契约，别让报错变成证据
        logger.warning("[tavily] 正文抓取失败 urls=%s: %s", wanted, exc)
        return _fail(context, f"抓取失败：{type(exc).__name__}: {exc}")

    raw_results = response.get("results") or []
    results = [_normalize_extract_result(item) for item in raw_results if isinstance(item, dict)]
    failed = [item for item in (response.get("failed_results") or []) if isinstance(item, dict)]
    logger.info("[tavily] 抓取 %d 个 URL，成功 %d 个，失败 %d 个", len(wanted), len(results), len(failed))

    payload: dict[str, Any] = {**context, "total": len(results), "results": results, "failed": failed}
    if not results:
        payload["note"] = _NO_EVIDENCE_NOTE
    return _dump(payload)


tavily_tools = [tavily_search, tavily_extract]
