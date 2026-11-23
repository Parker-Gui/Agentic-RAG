"""Fetch server-rendered public pages without starting Chrome."""

import logging

import httpx

from app.services.crawler.common.models import ArticleContent
from app.services.crawler.fetch.detail import parse_article

logger = logging.getLogger(__name__)


def fetch_html(url: str) -> str | None:
    try:
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=20,
            headers={"User-Agent": "IntelligenceAnalysisResearch/0.1", "Accept": "text/html"},
        )
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as exc:
        logger.warning("HTTP fetch failed for %s: %s", url, type(exc).__name__)
        return None


def fetch_http_article(url: str) -> ArticleContent:
    html = fetch_html(url)
    return parse_article(html) if html else ArticleContent()
