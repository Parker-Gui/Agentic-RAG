"""RAG 嵌入 worker 的命令行入口。

    python -m app.services.rag

把 DB 中 embedding_done=False 的文章切块、编码、写入 article_chunks。供 cron 定时或手动触发。
"""

from __future__ import annotations

import logging

from app.db.session import run_compatible_async
from app.services.rag.indexer import run_indexing

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    count = run_compatible_async(run_indexing())
    logger.info("全部完成，共处理 %d 篇文章", count)


if __name__ == "__main__":
    main()
