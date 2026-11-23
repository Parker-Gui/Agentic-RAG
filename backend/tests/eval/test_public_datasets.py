"""Public benchmark preparation preserves the candidate set and official splits."""

import json
import zipfile
from pathlib import Path

import pytest

from eval.datasets import prepare_scifact as scifact


def small_data() -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    corpus = [
        {"_id": "D1", "title": "One", "text": "Relevant to test"},
        {"_id": "D2", "title": "Two", "text": "Another candidate"},
    ]
    queries = [{"_id": "Q1", "text": "Test question"}, {"_id": "Q2", "text": "Train question"}]
    labels = {
        "test": [{"query-id": "Q1", "corpus-id": "D1", "score": 1}],
        "train": [{"query-id": "Q2", "corpus-id": "D2", "score": 1}],
    }
    return corpus, queries, labels


def test_selection_preserves_full_corpus_and_official_split() -> None:
    corpus, queries, labels = small_data()
    before = json.dumps(corpus, sort_keys=True)
    selected = scifact.validate(corpus, queries, labels)
    assert selected["test"] == [queries[0]] and selected["train"] == [queries[1]]
    assert json.dumps(corpus, sort_keys=True) == before and len(corpus) == 2


def test_missing_positive_document_is_rejected() -> None:
    corpus, queries, labels = small_data()
    labels["test"][0]["corpus-id"] = "MISSING"
    with pytest.raises(ValueError, match="missing"):
        scifact.validate(corpus, queries, labels)


def test_query_overlap_between_splits_is_rejected() -> None:
    corpus, queries, labels = small_data()
    labels["train"][0]["query-id"] = "Q1"
    with pytest.raises(ValueError, match="overlap"):
        scifact.validate(corpus, queries, labels)


def test_duplicate_document_id_is_rejected() -> None:
    corpus, queries, labels = small_data()
    corpus.append(dict(corpus[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        scifact.validate(corpus, queries, labels)


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.json", "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        scifact.extract_required(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside.json").exists()


def test_bad_cached_archive_is_preserved_and_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "scifact.zip"
    archive.write_bytes(b"not the official archive")
    with pytest.raises(ValueError, match="checksum"):
        scifact.download_archive(archive)
    assert archive.read_bytes() == b"not the official archive"
