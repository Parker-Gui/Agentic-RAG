"""Format adapters: retain structure before the common chunking pipeline."""

import io
import multiprocessing
import re
import zipfile
from typing import Any

import pymupdf
from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from markdown_it import MarkdownIt

from app.services.rag.document_types import BlockKind, ContentBlock, DocumentError, DocumentFormat, ParsedDocument

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_EXPANDED_BYTES = 100 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000
_BLOCK_TAGS = {"p", "div", "section", "article", "main", "blockquote", "ul", "ol", "li", "pre", "table"}


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_html(html: str, *, title: str = "") -> ParsedDocument:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "nav", "footer", "noscript", "aside"]):
        node.decompose()
    blocks: list[ContentBlock] = []
    headings: list[tuple[int, str]] = []

    def add(text: str, kind: BlockKind = "paragraph", **extra: Any) -> None:
        if text.strip():
            blocks.append(ContentBlock(kind=kind, text=text, heading_path=[v for _, v in headings], **extra))

    def walk(node: Tag) -> None:
        name = node.name or ""
        heading_match = re.fullmatch(r"h([1-6])", name)
        if heading_match:
            value, level = _flat(node.get_text(" ", strip=True)), int(heading_match[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, value))
            add(value, "heading")
            return
        if name == "table":
            rows = []
            for row in node.find_all("tr"):
                cells = row.find_all(["th", "td"], recursive=False)
                if any(cell.has_attr("rowspan") or cell.has_attr("colspan") for cell in cells):
                    raise DocumentError("complex_table", "Merged HTML table requires a richer parser")
                if cells:
                    rows.append("\t".join(_flat(cell.get_text(" ", strip=True)) for cell in cells))
            if rows:
                add("\n".join(rows), "table", header=rows[0])
            return
        if name in {"pre", "li"}:
            add(
                node.get_text("", strip=False) if name == "pre" else _flat(node.get_text(" ", strip=True)),
                "code" if name == "pre" else "list",
            )
            return
        buffer: list[str] = []

        def flush() -> None:
            add(_flat("".join(buffer)))
            buffer.clear()

        for child in node.children:
            if isinstance(child, NavigableString):
                buffer.append(str(child))
            elif isinstance(child, Tag):
                if child.name in _BLOCK_TAGS or re.fullmatch(r"h[1-6]", child.name or ""):
                    flush()
                    walk(child)
                elif child.name == "br":
                    buffer.append("\n")
                else:
                    buffer.append(child.get_text("", strip=False))
        flush()

    walk(soup.body or soup)
    for i, block in enumerate(blocks):
        block.locator = {"block": i}
    return ParsedDocument(title=title, format="html", blocks=blocks).canonicalize()


def parse_markdown(text: str, *, title: str = "") -> ParsedDocument:
    tokens = MarkdownIt("commonmark").enable("table").parse(text)
    headings: list[tuple[int, str]] = []
    blocks: list[ContentBlock] = []
    i = 0
    list_depth = 0
    while i < len(tokens):
        token = tokens[i]
        kind, content = "paragraph", ""
        start_line, end_line = token.map or [0, 0]
        if token.type in {"bullet_list_open", "ordered_list_open"}:
            list_depth += 1
        elif token.type in {"bullet_list_close", "ordered_list_close"}:
            list_depth -= 1
        elif token.type == "heading_open":
            level, content = int(token.tag[1]), tokens[i + 1].content
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, content))
            kind = "heading"
            i += 2
        elif token.type in {"fence", "code_block"}:
            kind, content = "code", token.content.rstrip("\n")
        elif token.type == "table_open":
            rows: list[str] = []
            cells: list[str] = []
            while i + 1 < len(tokens):
                i += 1
                t = tokens[i]
                if t.type == "inline":
                    cells.append(t.content)
                elif t.type == "tr_close":
                    rows.append("\t".join(cells))
                    cells = []
                elif t.type == "table_close":
                    break
            kind, content = "table", "\n".join(rows)
        elif token.type == "inline":
            content = token.content
            kind = "list" if list_depth else "paragraph"
        elif token.type == "html_block":
            nested = parse_html(token.content)
            blocks.extend(
                block.model_copy(update={"heading_path": [v for _, v in headings]}) for block in nested.blocks
            )
        if content.strip():
            locator: dict[str, Any] = {"lines": [start_line + 1, end_line]}
            if kind == "code" and token.info.strip():
                locator["language"] = token.info.strip().split()[0]
            blocks.append(
                ContentBlock(
                    kind=kind,
                    text=content,
                    heading_path=[v for _, v in headings],
                    locator=locator,
                    header=content.splitlines()[0] if kind == "table" else "",
                )
            )
        i += 1
    return ParsedDocument(title=title, format="md", blocks=blocks).canonicalize()


def parse_docx(data: bytes, *, title: str = "") -> ParsedDocument:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > MAX_EXPANDED_BYTES:
                raise DocumentError("expanded_size_limit", "DOCX expanded data exceeds limit")
    except zipfile.BadZipFile as exc:
        raise DocumentError("invalid_docx", "Invalid DOCX archive") from exc
    doc = Document(io.BytesIO(data))
    blocks: list[ContentBlock] = []
    headings: list[tuple[int, str]] = []
    for i, item in enumerate(doc.iter_inner_content()):
        kind, header = "paragraph", ""
        if isinstance(item, Paragraph):
            content = item.text
            match = re.fullmatch(r"Heading ([1-9])", item.style.name if item.style else "")
            if match:
                level = int(match[1])
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, content))
                kind = "heading"
        elif isinstance(item, Table):
            rows = []
            for row in item.rows:
                if row.grid_cols_before or row.grid_cols_after:
                    raise DocumentError("complex_table", "Omitted DOCX cells require a richer parser")
                if any(cell.tables or cell._tc.grid_span != 1 or cell._tc.vMerge is not None for cell in row.cells):
                    raise DocumentError("complex_table", "Merged/nested DOCX table requires a richer parser")
                rows.append("\t".join(_flat(cell.text) for cell in row.cells))
            content, kind = "\n".join(rows), "table"
            header = rows[0] if rows else ""
        else:
            continue
        blocks.append(
            ContentBlock(
                kind=kind, text=content, header=header, heading_path=[v for _, v in headings], locator={"block": i}
            )
        )
    return ParsedDocument(
        title=title,
        format="docx",
        blocks=blocks,
        warnings=["Main body only; comments, revisions, text boxes, footnotes and embedded objects are not guaranteed"],
    ).canonicalize()


def parse_pdf(data: bytes, *, title: str = "") -> ParsedDocument:
    blocks: list[ContentBlock] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise DocumentError("encrypted_pdf", "Password-protected PDF is unsupported")
        if len(doc) > MAX_PDF_PAGES:
            raise DocumentError("page_limit", "PDF exceeds page limit")
        for page_number, page in enumerate(doc, 1):
            page_blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0 and str(b[4]).strip()]
            text_chars = sum(len(str(block[4]).strip()) for block in page_blocks)
            large_image = any(
                pymupdf.Rect(image["bbox"]).get_area() > page.rect.get_area() * 0.4 for image in page.get_image_info()
            )
            if large_image and text_chars < 200:
                raise DocumentError("needs_ocr", f"Page {page_number} has an image-dominated body requiring OCR/review")
            if not page_blocks and (page.get_images() or page.get_drawings()):
                raise DocumentError("needs_ocr", f"Page {page_number} requires OCR or layout processing")
            for block in page_blocks:
                blocks.append(
                    ContentBlock(text=_flat(str(block[4])), locator={"page": page_number, "bbox": list(block[:4])})
                )
    return ParsedDocument(
        title=title,
        format="pdf",
        blocks=blocks,
        warnings=["Text-only, simple reading order; PDF tables, columns and image-based facts require review"],
    ).canonicalize()


def parse_document(data: bytes, format: DocumentFormat, *, title: str = "") -> ParsedDocument:
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError("file_size_limit", "Input exceeds 20 MiB")
    try:
        if format == "docx":
            parsed = parse_docx(data, title=title)
        elif format == "pdf":
            parsed = parse_pdf(data, title=title)
        else:
            text = data.decode("utf-8-sig")
            if format == "html":
                parsed = parse_html(text, title=title)
            elif format == "md":
                parsed = parse_markdown(text, title=title)
            elif format == "text":
                parsed = ParsedDocument(
                    title=title,
                    format="text",
                    blocks=[ContentBlock(text=text)],
                    warnings=["Unstructured plain text; original layout unavailable"],
                ).canonicalize()
            else:
                raise DocumentError("unsupported", "Unsupported document format")
        if len(parsed.text) > MAX_TEXT_CHARS:
            raise DocumentError("text_size_limit", "Parsed text exceeds character limit")
        return parsed
    except DocumentError:
        raise
    except UnicodeDecodeError as exc:
        raise DocumentError("unsupported_encoding", "Text input must be UTF-8") from exc
    except Exception as exc:
        raise DocumentError("parse_failed", f"Document parser failed ({type(exc).__name__})") from exc


def _parse_child(connection: Any, data: bytes, format: DocumentFormat, title: str) -> None:
    try:
        connection.send({"parsed": parse_document(data, format, title=title).model_dump(mode="json")})
    except DocumentError as exc:
        connection.send({"code": exc.code, "message": str(exc)})
    finally:
        connection.close()


def parse_isolated(data: bytes, format: DocumentFormat, *, title: str = "", timeout: float = 60) -> ParsedDocument:
    """Bound parser lifetime; a timed-out native parser cannot outlive its source lock."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_parse_child, args=(sender, data, format, title), daemon=True)
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout):
            raise DocumentError("parse_timeout", "Parser exceeded its time budget")
        try:
            result = receiver.recv()
        except EOFError as exc:
            raise DocumentError("parser_crashed", "Parser process terminated unexpectedly") from exc
        if "parsed" not in result:
            raise DocumentError(result["code"], result["message"])
        return ParsedDocument.model_validate(result["parsed"])
    finally:
        receiver.close()
        sender.close()
        if process.pid is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
