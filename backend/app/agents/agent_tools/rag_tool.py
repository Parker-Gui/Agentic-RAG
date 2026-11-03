"""内部情报知识库工具（pgvector RAG）。

薄适配层：重逻辑在 app/services/rag（RagService/facade/repo），这里只把结果转成
工具输出。返回统一为 {"results":[{title,url,published_at,content,score}]} 的 JSON 字符串，
和 tavily_tool 保持同一形状，方便 reasoning 节点统一消费、并在最终答案里引用 source URL。

工具内自开只读会话（get_worker_db_context），不依赖请求级 DI；任何异常都兜成
{"results": [], "error": ...}（0 条结果），避免报错文本被当成检索内容。

导入一律放模块级、不做函数内懒导入：本模块在建图时（事件循环之外）被导入，而
psycopg / torch 这类包的 import 本身就会读文件、取 cwd——放在协程里首次触发，
langgraph dev 的 blockbuster 会直接判定为阻塞调用并抛 BlockingError。
"""

import json
import logging
from typing import Annotated, Any

from langchain_core.tools import tool
from pydantic import Field

from app.db.session import get_worker_db_context
from app.repository import article as article_repo
from app.schemas.article import GroupType
from app.services.rag.facade import RagService

logger = logging.getLogger(__name__)

# top_k 上界。参数由模型采样产生，不是可信调用方：没有上界时它会自己往上加
# （实测传过 10），而每条命中都要进 messages、每轮 react 重付一遍 token。
MAX_TOP_K = 20


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _fail(context: dict[str, Any], reason: str) -> str:
    return _dump({**context, "results": [], "error": reason})


def _hit_to_record(hit: dict[str, Any]) -> dict[str, Any]:
    """RagService 命中 → 工具输出记录（content=chunk_text, url 供引用）。

    article_id 必须带上：命中的是片段，模型要读全文只能靠它调 read_article_tool。
    漏掉这个字段，那条链就是死的——模型手上只有 url，没有 id 可传。
    """
    return {
        "article_id": hit.get("article_id"),
        "title": hit.get("title") or "",
        "url": hit.get("url"),
        "published_at": hit.get("publish_date"),
        "content": hit.get("chunk_text") or "",
        "score": hit.get("score"),
        "source_kind": "internal_kb",
    }


@tool(description="内部情报知识库检索：对已抓取入库的竞品/法规文章做语义检索（pgvector）")
async def search_rag_tool(
    query: str,
    top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)] = 5,
    group_type: GroupType | None = None,
) -> str:
    """内部知识库语义检索。

    Args:
        query: 检索词。用具体关键词或专名，别传整段问题——一个 query 里塞多个主题会
            互相稀释语义（实测「A 收购 B 甜味剂 竞品 最新动态」得分 0.62，拆成
            「A 被 B 收购」升到 0.71）。多个意图请分多轮检索。
        top_k: 返回条数，1..20
        group_type: 可选，按分类过滤。不传 = 全库检索，这是默认且通常正确的选择；
            只在明确只要竞品或只要法规时才传。
    Returns:
        str: {"results":[...]} JSON；空 results 表示库内无相关。
    """
    context: dict[str, Any] = {"query": query, "top_k": top_k, "group_type": group_type}
    try:
        async with get_worker_db_context() as db:
            hits = await RagService(db).query(query, top_k=top_k, group_type=group_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[search_rag] 检索失败 query=%r: %s", query, exc)
        return _fail(context, f"内部检索失败：{type(exc).__name__}: {exc}")

    results = [_hit_to_record(h) for h in hits]
    logger.info("[search_rag] query=%r group_type=%r 命中 %d 条", query, group_type, len(results))
    return _dump({**context, "total": len(results), "results": results, "note": _note_for(results, group_type)})


def _note_for(results: list[dict[str, Any]], group_type: str | None) -> str:
    """给模型的下一步提示。空结果最难自纠，所以要说清「零命中可能是过滤器造成的」。

    零命中和「库里真的没有」在返回值里长得一样。上一版只说「换关键词」，模型于是
    换了 5 轮关键词、一次没碰那个才是真凶的 group_type，直到撞穿轮次上限。
    """
    if results:
        return "内部库结果；正文是片段，要全文用 read_article_tool（传 article_id）。"
    if group_type is not None:
        return (
            f"库内无相关。注意本次按 group_type={group_type!r} 过滤过，"
            "零命中可能只是这个分类下没有数据——建议先去掉 group_type 重试一次全库检索，"
            "再考虑换关键词或用 tavily_search 走外网。"
        )
    return "库内无相关（已全库检索，无分类过滤）；可换更具体的关键词，或用 tavily_search 走外网。"


@tool(description="按 article_id 读取内部文章全文（检索命中的是片段，需要全文时用它）")
async def read_article_tool(
    article_id: int,
    max_chars: Annotated[int, Field(ge=200, le=20000)] = 2000,
) -> str:
    """内部版正文抓取：取整篇 body_text。

    Args:
        article_id: 文章 id，取 search_rag_tool 结果里的 article_id 字段，不要猜
        max_chars: 截断上限（200..20000），超出标 truncated

    Returns:
        str: {"results":[{title,url,published_at,content,truncated}]} JSON。
    """
    context: dict[str, Any] = {"article_id": article_id}
    try:
        async with get_worker_db_context() as db:
            article = await article_repo.get_by_id(db, article_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[read_article] 读取失败 id=%s: %s", article_id, exc)
        return _fail(context, f"读取文章失败：{type(exc).__name__}: {exc}")

    if article is None:
        return _fail(context, f"article {article_id} 不存在")

    body = article.body_text or ""
    record = {
        "title": article.title,
        "url": article.url,
        "published_at": str(article.publish_date) if article.publish_date else None,
        "content": body[:max_chars],
        "truncated": len(body) > max_chars,
        "source_kind": "internal_kb",
    }
    return _dump({**context, "total": 1, "results": [record]})
