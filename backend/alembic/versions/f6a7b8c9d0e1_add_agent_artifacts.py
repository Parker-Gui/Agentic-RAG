"""Persist a single Markdown artifact per research task."""

import sqlalchemy as sa

from alembic import op

revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_artifacts",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task_id", sa.UUID(), nullable=False, unique=True),
        sa.Column("conversation_id", sa.UUID(), sa.ForeignKey("agent_conversations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("format", sa.String(20), nullable=False),
        sa.Column("filename", sa.String(80), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("agent_artifacts_user_id_idx", "agent_artifacts", ["user_id"])


def downgrade() -> None:
    op.drop_table("agent_artifacts")
