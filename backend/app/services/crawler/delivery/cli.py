"""命令行入口：抓取被监测站点，然后把结果写入 articles 表。

抓取始终执行。默认把抓到的文章写入数据库；加 ``--scrape-only`` 则只抓不写
（用于排查站点抓取问题）。运行日志与站点抓取汇总（``delivery/reporting.py``）
每次都会生成，不是可选阶段。

新鲜度来自每个站点保存的时间戳（config/monitor_last_times.json），而非关键词
匹配。抓取见 ``pipeline``，写库见 ``persistence``；本模块只负责命令行编排。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from app.services.crawler.common.models import Hit
from app.services.crawler.delivery.reporting import configure_console
from app.services.crawler.orchestration.pipeline import acknowledge_run, scan_all_sites

logger = logging.getLogger(__name__)

_RULE = "=" * 60


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.services.crawler",
        description="竞品/法规情报监测：抓取新文章并写入 articles 表。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="只抓取，不写库（用于排查站点抓取问题）",
    )
    parser.add_argument("--site", action="append", dest="sites", help="只抓指定站点，可重复传入，名称需与配置一致")
    parser.add_argument("--limit", type=int, choices=range(1, 21), metavar="1..20", help="每个站点最多读取多少篇文章")
    return parser.parse_args(argv)


def persist(hits: Sequence[Hit]) -> bool:
    """把这批文章写入 articles 表；成功返回 True。

    在函数内部懒加载：``--scrape-only`` 运行不会引入 SQLAlchemy 栈，也不需要
    设置 DATABASE_URL。
    """
    from app.db.session import run_compatible_async
    from app.services.crawler.delivery.persistence import save_hits_to_db

    try:
        count = run_compatible_async(save_hits_to_db(hits))
    except Exception:
        logger.exception(
            "写库失败，%d 条未入库。站点时间戳未推进，可以修复数据库后重跑。",
            len(hits),
        )
        return False
    logger.info("已写入 articles 表 %d 条。", count)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_console()

    mode = "抓取（仅日志与站点汇总，不写库）" if args.scrape_only else "抓取 → 写入数据库"
    logger.info("%s\n本次运行: %s\n%s", _RULE, mode, _RULE)

    run = scan_all_sites(site_names=args.sites, max_items=args.limit)
    failed = any(outcome.status not in {"ok", "no_new"} for outcome in run.outcomes)
    if not run.hits:
        if failed:
            logger.error("没有取得文章，部分站点抓取失败，请查看站点汇总。")
            return 1
        logger.info("\n没有新文章（都不晚于各站点上次记录的时间）。")
        return 0

    if args.scrape_only:
        logger.info("\n--scrape-only：跳过写库，本次抓到 %d 条。", len(run.hits))
        return 1 if failed else 0

    if not persist(run.hits):
        return 1
    acknowledge_run(run)
    return 1 if failed else 0
