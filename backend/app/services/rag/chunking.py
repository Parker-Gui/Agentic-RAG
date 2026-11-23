"""Source-preserving chunking with the maintained LangChain splitter.

The default budget is characters, not BGE or chat-model tokens. Embedding
adapters must additionally reject inputs beyond their actual model limit.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.rag.document_types import DocumentChunk, DocumentError, ParsedDocument

CHUNKING_VERSION = "structure-char-v1"
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", "，", ", ", " ", ""]


def _splitter(size: int, overlap: int, *, lines: bool = False) -> RecursiveCharacterTextSplitter:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n", " ", ""] if lines else _SEPARATORS,
        keep_separator="end",
        strip_whitespace=False,
        add_start_index=True,
    )


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Compatibility entry point; preserve paragraph/line structure and real words."""
    return [piece for piece in _splitter(chunk_size, overlap).split_text(text.strip()) if piece.strip()]


def chunk_document(parsed: ParsedDocument, *, chunk_size: int = 500, overlap: int = 50) -> list[DocumentChunk]:
    _splitter(chunk_size, overlap)
    result: list[DocumentChunk] = []
    for block in parsed.blocks:
        prefix = " / ".join(value for value in [parsed.title, *block.heading_path] if value)[:120]
        if block.header:
            prefix += "\n" + block.header[:120]
        prefix = prefix[: min(240, chunk_size // 2)]
        prefix = prefix + "\n" if prefix else ""
        budget = chunk_size - len(prefix)
        if budget < 1:
            raise DocumentError("prefix_budget", "Prefix leaves no body space", stage="chunking")
        if block.kind == "table" and any(len(line) > budget for line in block.text.splitlines(keepends=True)):
            raise DocumentError("table_row_budget", "A table row exceeds the body budget", stage="chunking")
        splitter = _splitter(budget, min(overlap, budget - 1), lines=block.kind in {"code", "table"})
        for doc in splitter.create_documents([block.text]):
            text, local_start = doc.page_content, doc.metadata["start_index"]
            if not text.strip():
                continue
            if local_start < 0 or block.text[local_start : local_start + len(text)] != text:
                raise DocumentError("offset_mismatch", "Invalid source offset", stage="chunking")
            start, end = block.start + local_start, block.start + local_start + len(text)
            if parsed.text[start:end] != text:
                raise DocumentError("offset_mismatch", "Chunk differs from canonical text", stage="chunking")
            result.append(
                DocumentChunk(
                    index=len(result),
                    text=text,
                    start=start,
                    end=end,
                    kind=block.kind,
                    heading_path=block.heading_path,
                    locator=block.locator,
                    parent_start=block.start,
                    parent_end=block.end,
                    embedding_text=prefix + text,
                    split_reason="length" if len(block.text) > budget else "structure",
                    context_ranges=[{"start": block.start, "end": block.start + len(block.header)}]
                    if block.header
                    else [],
                )
            )
    return result
