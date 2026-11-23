"""Public artifact metadata; the Markdown body is delivered only by download."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class ArtifactRead(BaseModel):
    id: UUID
    task_id: UUID
    conversation_id: UUID | None
    title: str
    format: Literal["markdown"]
    filename: str
    size_bytes: int
    sha256: str
    created_at: datetime
    download_url: str


class ArtifactList(BaseModel):
    items: list[ArtifactRead]
    total: int
