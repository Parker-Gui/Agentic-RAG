"""把抓取到的 hits 写入 articles 表。

这是爬虫与 FastAPI / SQLAlchemy 栈之间唯一的集成点——本包其余部分都是不依赖
数据库的纯同步 Python，因此调用方懒加载本模块。

命令行上这是默认（非 ``--scrape-only``）行为::

    python -m app.services.crawler              # 抓取并入库
    python -m app.services.crawler --scrape-only  # 只抓不写

在代码里::

    from app.db.session import run_compatible_async
    from app.services.crawler import fetch_all_new_hits, save_hits_to_db

    hits = fetch_all_new_hits()
    run_compatible_async(save_hits_to_db(hits))
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING

from app.db.session import get_worker_db_context
from app.repository import article as article_repo
from app.services.crawler.common.models import Hit

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def parse_publish_date(value: object) -> date | None:
    """把 hit 的 publish_date 解析成 ``date``。

    抓取存的是 "2026-08-18" 或 "2026-08-18 00:00:00"，时间部分是 CMS 噪声。
    无法解析的存 NULL，而不是让整批失败。
    """
    if value is None:
        return None
    text = str(value).strip().split(" ")[0]
    try:
        return date.fromisoformat(text)
    except ValueError:
        logger.warning("Could not parse publish_date %r — storing as NULL", value)
        return None


async def save_hits_to_db(hits: Sequence[Hit]) -> int:
    """按 URL upsert 一批 hits；返回写入的行数。

    ``get_worker_db_context`` 是短生命周期的 NullPool 会话，成功即提交、出错回滚，
    用于请求之外的后台工作。
    """
    if not hits:
        return 0

    async with get_worker_db_context() as db:
        count = await _save(db, hits)
        logger.info("Saved %d articles to database", count)
        return count


async def _save(db: AsyncSession, hits: Sequence[Hit]) -> int:
    saved = 0
    for hit in hits:
        url = hit.get("url", "").strip()
        title = hit.get("title", "").strip()
        if not url or not title:
            logger.warning("Skipping hit with missing url/title: %r", hit)
            continue

        await article_repo.upsert_article(
            db,
            site_name=hit.get("site", ""),
            title=title,
            url=url,
            publish_date=parse_publish_date(hit.get("publish_date")),
            body_text=hit.get("body_text") or None,
            group_type=hit.get("group", "competitor"),
            is_external=bool(hit.get("external", False)),
        )
        saved += 1
    return saved
