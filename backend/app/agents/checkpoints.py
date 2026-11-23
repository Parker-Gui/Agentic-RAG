"""Official PostgreSQL checkpoints in an isolated schema of the existing database."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection, Error
from psycopg.conninfo import make_conninfo
from sqlalchemy.engine import make_url

from app.core.config import settings
from app.core.exceptions import ExternalServiceError

SCHEMA = "agent_checkpoints"


def connection_string(*, scoped: bool = True) -> str:
    uri = make_url(settings.DATABASE_URL).set(drivername="postgresql").render_as_string(hide_password=False)
    return make_conninfo(uri, options=f"-csearch_path={SCHEMA}") if scoped else uri


async def setup_checkpoints() -> None:
    """Explicit maintenance command only; never perform DDL during a request."""
    async with await AsyncConnection.connect(connection_string(scoped=False), autocommit=True) as connection:
        await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    async with AsyncPostgresSaver.from_conn_string(connection_string()) as saver:
        await saver.setup()


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[Any]:
    try:
        async with AsyncPostgresSaver.from_conn_string(connection_string()) as saver:
            yield saver
    except (Error, TimeoutError) as exc:
        raise ExternalServiceError(message="任务检查点存储暂不可用。") from exc
