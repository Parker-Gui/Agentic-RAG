"""Explicit offline cutover. Preserve IDs, parent links, versions and pending writes.

Stop the backend before running --confirm-stopped. No fallback or deletion of the
source file. New backend writes after cutover require reconciliation before rollback.
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.checkpoints import open_checkpointer, setup_checkpoints
from app.agents.memory.notes import DATA_ROOT, checkpoint_path
from app.db.session import run_compatible_async


async def migrate(*, setup_only: bool = False) -> dict[str, Any]:
    source = checkpoint_path()
    if not source.exists() and not setup_only:
        raise RuntimeError("Source checkpoint file is missing; use --setup-only for a fresh installation")
    backup = None
    if not setup_only:
        directory = DATA_ROOT / "backups" / datetime.now(UTC).strftime("checkpoints-%Y%m%dT%H%M%S%fZ")
        directory.mkdir(parents=True)
        backup = directory / "checkpoints.sqlite"
        with (
            sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True) as existing,
            sqlite3.connect(backup) as copy,
        ):
            existing.backup(copy)
    await setup_checkpoints()
    copied = reused = 0
    threads = set()
    if not setup_only:
        async with AsyncSqliteSaver.from_conn_string(str(backup)) as old, open_checkpointer() as new:
            records = [item async for item in old.alist(None)]
            for record in reversed(records):
                config = record.config
                thread = config["configurable"]["thread_id"]
                threads.add(thread)
                present = await new.aget_tuple(config)
                if present is None:
                    parent = record.parent_config or {
                        "configurable": {
                            "thread_id": thread,
                            "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
                        }
                    }
                    await new.aput(
                        parent, record.checkpoint, record.metadata, record.checkpoint.get("channel_versions", {})
                    )
                    grouped: dict[str, list[tuple[str, Any]]] = defaultdict(list)
                    for task, channel, value in record.pending_writes or []:
                        grouped[task].append((channel, value))
                    for task, writes in grouped.items():
                        await new.aput_writes(config, writes, task)
                    copied += 1
                    present = await new.aget_tuple(config)
                else:
                    reused += 1
                if (
                    present is None
                    or present.checkpoint != record.checkpoint
                    or present.metadata != record.metadata
                    or present.parent_config != record.parent_config
                ):
                    raise RuntimeError("Checkpoint verification failed; source and backup retained")
                original: dict[tuple[str, str], list[Any]] = defaultdict(list)
                migrated: dict[tuple[str, str], list[Any]] = defaultdict(list)
                for task, channel, value in record.pending_writes or []:
                    original[(task, channel)].append(value)
                for task, channel, value in present.pending_writes or []:
                    migrated[(task, channel)].append(value)
                if original != migrated:
                    raise RuntimeError("Pending-write verification failed; source and backup retained")
    report = {
        "copied": copied,
        "already_present": reused,
        "threads": len(threads),
        "backup": str(backup) if backup else None,
        "verified": True,
        "source_deleted": False,
        "backend": "postgresql",
        "schema": "agent_checkpoints",
    }
    DATA_ROOT.mkdir(exist_ok=True)
    (DATA_ROOT / "checkpoint-migration-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-stopped", action="store_true", required=True)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_compatible_async(migrate(setup_only=args.setup_only))))  # noqa: T201 -- maintenance CLI


if __name__ == "__main__":
    main()
