"""Crawler package facade.

命令行用法见 ``delivery/cli.py`` 或 ``python -m app.services.crawler --help``。
默认抓取并写入 articles 表；加 ``--scrape-only`` 只抓不写。

在代码里调用::

    from app.db.session import run_compatible_async
    from app.services.crawler import fetch_all_new_hits, save_hits_to_db

    hits = fetch_all_new_hits()
    run_compatible_async(save_hits_to_db(hits))
"""

from app.services.crawler.common.constants import BASELINE_OVERRIDE
from app.services.crawler.common.models import Candidate, Hit, Outcome
from app.services.crawler.config.sites import load_sites
from app.services.crawler.config.state import load_last_times, save_last_times
from app.services.crawler.delivery.cli import main
from app.services.crawler.orchestration.pipeline import (
    MonitorRun,
    SiteScan,
    fetch_all_new_hits,
    scan_all_sites,
    scan_site,
)

__all__ = [
    "BASELINE_OVERRIDE",
    "Candidate",
    "Hit",
    "MonitorRun",
    "Outcome",
    "SiteScan",
    "fetch_all_new_hits",
    "load_last_times",
    "load_sites",
    "main",
    "save_hits_to_db",
    "save_last_times",
    "scan_all_sites",
    "scan_site",
]


def __getattr__(name: str):
    # 懒加载：导入 persistence 会拉入 SQLAlchemy 并要求 DATABASE_URL，
    # 而 --scrape-only 运行不该需要它。
    if name == "save_hits_to_db":
        from app.services.crawler.delivery.persistence import save_hits_to_db

        return save_hits_to_db
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
