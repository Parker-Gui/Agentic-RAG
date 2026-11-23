"""Local RAG import/status/resume/read/query and legacy indexing commands."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.db.session import get_worker_db_context, run_compatible_async
from app.services.rag import KnowledgeService, RagService
from app.services.rag.indexer import run_indexing


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-source", choices=["legacy", "documents"])
    commands = parser.add_subparsers(dest="command")
    command = commands.add_parser("import")
    command.add_argument("file", type=Path)
    command.add_argument("--source-key", required=True)
    command.add_argument("--group-type", default="internal")
    command = commands.add_parser("status")
    command.add_argument("--document-id", type=UUID, required=True)
    command = commands.add_parser("resume")
    command.add_argument("--revision-id", type=UUID, required=True)
    command = commands.add_parser("read")
    command.add_argument("--document-id", type=UUID, required=True)
    command.add_argument("--revision-id", type=UUID)
    command.add_argument("--start", type=int, default=0)
    command.add_argument("--max-chars", type=int, default=2000)
    command = commands.add_parser("query")
    command.add_argument("question")
    command.add_argument("--top-k", type=int, default=5, choices=range(1, 51))
    commands.add_parser("index-articles")
    return parser


async def execute(args: argparse.Namespace) -> Any:
    if args.index_source:
        settings.RAG_INDEX_SOURCE = args.index_source
    service = KnowledgeService()
    match args.command:
        case "import":
            return await service.import_file(args.file, source_key=args.source_key, group_type=args.group_type)
        case "status":
            return await service.status(args.document_id)
        case "resume":
            return await service.resume(args.revision_id)
        case "read":
            return await service.read(
                args.document_id, revision_id=args.revision_id, start=args.start, max_chars=args.max_chars
            )
        case "query":
            async with get_worker_db_context() as db:
                return await RagService(db).query(args.question, top_k=args.top_k)
        case "index-articles":
            return {"processed": await service.sync_articles(), "index_source": settings.RAG_INDEX_SOURCE}
        case _:
            return {"processed": await run_indexing(), "index_source": settings.RAG_INDEX_SOURCE}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        result = run_compatible_async(execute(arguments().parse_args()))
    except Exception as exc:
        sys.stdout.write(
            json.dumps(
                {"error": type(exc).__name__, "message": getattr(exc, "message", "Command failed")}, ensure_ascii=False
            )
            + "\n"
        )
        raise SystemExit(1) from None
    sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    if isinstance(result, dict) and result.get("state") in {"failed", "unsupported", "needs_ocr"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
