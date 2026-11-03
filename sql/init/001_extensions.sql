-- Runs once, on an empty data directory, via /docker-entrypoint-initdb.d.
-- The entrypoint executes this as POSTGRES_USER (a superuser) against
-- POSTGRES_DB, which is exactly the privilege CREATE EXTENSION needs.
--
-- Scope on purpose: cluster-level setup only. Tables are NOT created here --
-- they belong to alembic (`scripts/prestart.sh` runs `alembic upgrade head`),
-- and a table created here would collide with `op.create_table` on the very
-- first migration run.

-- Required by article_chunks.embedding (pgvector). The migration also issues
-- this statement, so doing it here is a head start, not a duplicate: whichever
-- runs first wins and the other is a no-op.
CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram index support, available in the image and handy for fuzzy title
-- search on `articles`. Harmless if nothing uses it yet.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
