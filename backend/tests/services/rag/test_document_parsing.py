"""Real tiny documents exercise structure, coordinates and failure boundaries."""

import io

import pymupdf
import pytest
from docx import Document

from app.services.rag.chunking import chunk_document, chunk_text
from app.services.rag.document_types import DocumentError
from app.services.rag.parsers import parse_document


def test_html_keeps_inline_words_paragraphs_and_tail() -> None:
    html = "<article><h2>Policy</h2><p>Read <b>this</b> rule.</p><p>" + "x" * 9000 + "END</p></article>"
    parsed = parse_document(html.encode(), "html")
    assert [block.kind for block in parsed.blocks] == ["heading", "paragraph", "paragraph"]
    assert parsed.blocks[1].text == "Read this rule."
    assert parsed.text.endswith("END")
    assert parsed.blocks[-1].heading_path == ["Policy"]


def test_markdown_fenced_heading_is_code_and_table_is_ordered() -> None:
    parsed = parse_document(b"# Policy\n\n```py\n# not heading\n```\n\n|City|Limit|\n|---|---|\n|A|600|\n\nEnd", "md")
    assert [b.kind for b in parsed.blocks] == ["heading", "code", "table", "paragraph"]
    assert parsed.blocks[1].heading_path == ["Policy"]
    assert parsed.blocks[2].text == "City\tLimit\nA\t600"


def test_docx_keeps_paragraph_table_order_and_heading() -> None:
    document = Document()
    document.add_heading("Travel", 1)
    document.add_paragraph("Before")
    table = document.add_table(rows=2, cols=2)
    values = ["City", "Limit", "A", "600"]
    for cell, value in zip([c for row in table.rows for c in row.cells], values, strict=True):
        cell.text = value
    document.add_paragraph("After")
    output = io.BytesIO()
    document.save(output)
    parsed = parse_document(output.getvalue(), "docx")
    assert [b.kind for b in parsed.blocks] == ["heading", "paragraph", "table", "paragraph"]
    assert parsed.blocks[2].text == "City\tLimit\nA\t600"
    assert parsed.blocks[2].heading_path == ["Travel"]


def test_pdf_retains_page_and_position() -> None:
    with pymupdf.open() as doc:
        doc.new_page().insert_text((50, 50), "First page")
        doc.new_page().insert_text((50, 50), "Second page")
        data = doc.tobytes()
    parsed = parse_document(data, "pdf")
    assert [b.locator["page"] for b in parsed.blocks] == [1, 2]
    assert len(parsed.blocks[0].locator["bbox"]) == 4


def test_pdf_image_only_page_does_not_become_empty_success() -> None:
    with pymupdf.open() as doc:
        page = doc.new_page()
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 8, 8), False)
        pixmap.clear_with(255)
        page.insert_image(page.rect, pixmap=pixmap)
        data = doc.tobytes()
    with pytest.raises(DocumentError, match="OCR") as exc:
        parse_document(data, "pdf")
    assert exc.value.code == "needs_ocr"


def test_chunks_preserve_repeated_text_offsets_and_prefix_budget() -> None:
    paragraph = "Same paragraph. " * 35
    parsed = parse_document(("# Policy\n\n" + paragraph + "\n\n" + paragraph).encode(), "md", title="Manual")
    chunks = chunk_document(parsed, chunk_size=120, overlap=20)
    assert len(chunks) > 4
    for chunk in chunks:
        assert chunk.text == parsed.text[chunk.start : chunk.end]
        assert len(chunk.embedding_text) <= 120


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (20, -1), (20, 20)])
def test_invalid_chunk_budget_fails_early(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("data", chunk_size=size, overlap=overlap)


def test_docx_merged_table_is_explicitly_unsupported() -> None:
    doc = Document()
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Merged"
    output = io.BytesIO()
    doc.save(output)
    with pytest.raises(DocumentError) as exc:
        parse_document(output.getvalue(), "docx")
    assert exc.value.code == "complex_table"


def test_markdown_list_items_keep_their_type() -> None:
    parsed = parse_document(b"# Policy\n\n- Item one\n- Item two\n", "md")
    assert [block.kind for block in parsed.blocks] == ["heading", "list", "list"]


def test_markdown_code_keeps_indentation_and_language() -> None:
    parsed = parse_document(b"```python\n    indented()\n```\n", "md")
    assert parsed.blocks[0].text == "    indented()"
    assert parsed.blocks[0].locator["language"] == "python"


def test_pdf_header_does_not_hide_a_scanned_body() -> None:
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((30, 30), "Page header only")
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 8, 8), False)
        pixmap.clear_with(255)
        page.insert_image(pymupdf.Rect(30, 60, 560, 800), pixmap=pixmap)
        data = doc.tobytes()
    with pytest.raises(DocumentError) as exc:
        parse_document(data, "pdf")
    assert exc.value.code == "needs_ocr"
