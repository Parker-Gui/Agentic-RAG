"""A task-scoped virtual filesystem rooted under the local agent data directory."""

from pathlib import Path
from uuid import UUID

from deepagents.backends import FilesystemBackend

DATA_ROOT = Path(__file__).resolve().parents[3] / ".agent_data"


def notes_backend(user_id: str, task_id: str) -> FilesystemBackend:
    """UUID normalization prevents a caller from escaping the intended directory."""
    user = str(UUID(user_id))
    task = str(UUID(task_id))
    root = DATA_ROOT / "notes" / user / task
    root.mkdir(parents=True, exist_ok=True)
    return FilesystemBackend(root_dir=root, virtual_mode=True)


def checkpoint_path() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT / "checkpoints.sqlite"
