"""Versioned parser output and source coordinates, independent of persistence."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.exceptions import ValidationError

DocumentFormat = Literal["html", "md", "docx", "pdf", "text"]
BlockKind = Literal["heading", "paragraph", "list", "table", "code"]
PARSER_VERSION = "structured-v1"


class DocumentError(ValidationError):
    def __init__(self, code: str, message: str, *, stage: str = "parsing") -> None:
        super().__init__(message=message, code=code)
        self.stage = stage


class ContentBlock(BaseModel):
    kind: BlockKind = "paragraph"
    text: str
    heading_path: list[str] = Field(default_factory=list)
    locator: dict[str, Any] = Field(default_factory=dict)
    start: int = 0
    end: int = 0
    header: str = ""


class ParsedDocument(BaseModel):
    title: str
    format: DocumentFormat
    text: str = ""
    blocks: list[ContentBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    parser_version: str = PARSER_VERSION

    def canonicalize(self) -> "ParsedDocument":
        """Assign offsets only after the canonical text is fixed."""
        self.blocks = [block for block in self.blocks if block.text.strip()]
        parts: list[str] = []
        offset = 0
        for block in self.blocks:
            block.text = block.text.strip("\r\n") if block.kind == "code" else block.text.strip()
            block.start = offset
            block.end = offset + len(block.text)
            parts.append(block.text)
            offset = block.end + 2
        self.text = "\n\n".join(parts)
        if not self.text:
            raise DocumentError("empty_document", "Document contains no indexable text")
        return self


class DocumentChunk(BaseModel):
    index: int
    text: str
    start: int
    end: int
    kind: str
    heading_path: list[str]
    locator: dict[str, Any]
    parent_start: int
    parent_end: int
    embedding_text: str
    split_reason: str = "structure"
    context_ranges: list[dict[str, int]] = Field(default_factory=list)
