"""Article schemas: request filters and API response shapes."""

from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.base import BaseSchema, TimestampSchema

# articles.group_type 的取值域，唯一出处。
# 声明成 Literal 而非 frozenset，是为了能同时喂两个消费者：
#   - services.article.VALID_GROUP_TYPES（运行时校验，由此派生）
#   - agent_tools.rag_tool 的工具签名 —— Literal 会被编进工具的 JSON Schema
#     `enum`，模型于是只能在合法值里选，而不是自己编一个（曾编出 "news"）。
GroupType = Literal["competitor", "regulatory"]


class ArticleRead(TimestampSchema):
    """Single article returned by the API."""

    id: int
    site_name: str
    title: str
    url: str
    publish_date: date | None = None
    summary: str | None = None
    group_type: str
    is_external: bool
    embedding_done: bool


class ArticleDetail(ArticleRead):
    """Article with full body text, returned by the detail endpoint."""

    body_text: str | None = None


class ArticleList(BaseSchema):
    """Paginated list of articles."""

    items: list[ArticleRead]
    total: int
    skip: int
    limit: int


class ArticleQuery(BaseSchema):
    """Query parameters for the history / list endpoint."""

    # 故意保持 str 而非 GroupType：非法值要走 service 的领域校验（→ 400），
    # 收成 Literal 会让 Pydantic 先拦成 422，改掉这个端点的对外契约。
    group_type: str | None = Field(default=None, description="'competitor' or 'regulatory'")
    site_name: str | None = Field(default=None, description="Filter by exact site name")
    date_from: date | None = Field(default=None, description="Publish date ≥ this date")
    date_to: date | None = Field(default=None, description="Publish date ≤ this date")
    search: str | None = Field(default=None, description="Keyword search in title / site name")
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)
