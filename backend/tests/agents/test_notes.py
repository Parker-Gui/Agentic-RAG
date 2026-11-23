"""Task notes persist while user/task namespaces stay isolated."""

from pathlib import Path
from uuid import uuid4

import pytest

from app.agents.memory import notes


def test_notes_survive_new_backend_and_other_user_cannot_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notes, "DATA_ROOT", tmp_path)
    user_a, user_b, task = str(uuid4()), str(uuid4()), str(uuid4())
    first = notes.notes_backend(user_a, task)
    assert first.write("/notes/progress.md", "已检查内部库").error is None
    reopened = notes.notes_backend(user_a, task)
    read = reopened.read("/notes/progress.md")
    assert read.file_data is not None and read.file_data["content"] == "已检查内部库"
    other = notes.notes_backend(user_b, task)
    assert other.read("/notes/progress.md").error is not None


def test_notes_reject_non_uuid_path_component(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notes, "DATA_ROOT", tmp_path)
    with pytest.raises(ValueError):
        notes.notes_backend("../../other", str(uuid4()))


def test_notes_backend_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notes, "DATA_ROOT", tmp_path)
    backend = notes.notes_backend(str(uuid4()), str(uuid4()))
    with pytest.raises(ValueError, match="Path traversal"):
        backend.write("/../outside.txt", "not allowed")
    assert not (tmp_path / "outside.txt").exists()
