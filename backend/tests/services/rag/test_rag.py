"""app/services/test_rag.py 的单元测试。

纯函数、无 I/O，因此无需数据库或 anyio —— 普通同步测试即可。
"""

import pytest

from app.services.rag.chunking import chunk_text

# 一段含段落（\n\n）、换行（\n）与句号的中文正文，重复以确保远超单块大小。
_LONG_TEXT = (
    "公司今日宣布在印度新建一座原料药工厂。该工厂预计明年投产，产能将翻倍。\n"
    "此举旨在应对东南亚市场的快速增长。管理层表示，扩产是长期战略的一部分。\n\n"
    "与此同时，公司也在推进数字化转型。研发投入较去年增长两成。"
    "分析师普遍看好其未来两年的营收表现。"
) * 10


# --- 正常路径 ---


def test_chunk_long_text_returns_multiple_chunks():
    chunks = chunk_text(_LONG_TEXT, chunk_size=200, overlap=50)
    assert len(chunks) > 1


def test_chunk_long_text_every_chunk_is_non_empty():
    chunks = chunk_text(_LONG_TEXT, chunk_size=200, overlap=50)
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_long_text_respects_soft_size_upper_bound():
    chunk_size, overlap = 200, 50
    chunks = chunk_text(_LONG_TEXT, chunk_size=chunk_size, overlap=overlap)
    # 块大小是软约束：新块会带上一块末尾的 overlap 段，故上界是 chunk_size + overlap。
    assert all(len(chunk) <= chunk_size + overlap for chunk in chunks)


def test_chunk_preserves_sentence_boundaries_instead_of_fixed_character_tails():
    sentences = [f"Sentence number {index} contains its own fact. " for index in range(12)]
    chunks = chunk_text("".join(sentences), chunk_size=120, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.startswith("Sentence") for chunk in chunks)
    assert all(any(sentence.strip() in chunk for chunk in chunks) for sentence in sentences)


def test_chunk_preserves_lines_after_structure_aware_extraction():
    # HTML行内元素由解析器处理；切块层不再把真实结构换行全部抹平。
    text = "公司宣布\n将在印度\n新建工厂。"
    assert chunk_text(text, chunk_size=200) == [text]


_ENGLISH_TEXT = (
    "Ingredion will showcase several innovations at IFT First 2026. "
    "The company focuses on clean label and texture. "
    "It also supports sugar reduction and protein fortification. "
) * 8


def test_chunk_english_text_splits_into_multiple_chunks():
    chunks = chunk_text(_ENGLISH_TEXT, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


# --- 边界情况 ---


def test_chunk_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_shorter_than_size_returns_single_chunk():
    text = "很短的一段话。"
    assert chunk_text(text, chunk_size=200) == [text]


def test_chunk_strips_leading_and_trailing_whitespace():
    assert chunk_text("  首尾有空白  ", chunk_size=200) == ["首尾有空白"]


# --- 非法参数 ---


def test_chunk_overlap_equal_to_chunk_size_raises_value_error():
    with pytest.raises(ValueError):
        chunk_text("任意文本", chunk_size=100, overlap=100)


def test_chunk_overlap_greater_than_chunk_size_raises_value_error():
    with pytest.raises(ValueError):
        chunk_text("任意文本", chunk_size=100, overlap=200)
