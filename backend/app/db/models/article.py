"""Article database model.

Stores crawled news articles from monitored sites.
Each row is one article; url is the natural deduplication key.
"""

from datetime import date

from sqlalchemy import BigInteger, Boolean, Date, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Article(Base, TimestampMixin):
    """Crawled article from a monitored site."""

    __tablename__ = "articles"

    # BigInteger, not the bare `int` default: the column is `bigint` in the DB and
    # `article_chunks.article_id` references it. A bare `int` maps to `Integer`, so
    # `alembic revision --autogenerate` would emit an ALTER narrowing it back to
    # int4 -- which the foreign key then rejects.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Source
    site_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # Content
    publish_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    body_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Classification
    group_type: Mapped[str] = mapped_column(String(50), nullable=False, default="competitor", index=True)
    is_external: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # RAG pipeline status
    embedding_done: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, site={self.site_name!r}, title={self.title[:40]!r})>"
