"""Add conversation and turn metadata for sequential follow-up questions."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_conversations",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("agent_conversations_user_id_idx", "agent_conversations", ["user_id"])
    op.create_table(
        "agent_conversation_turns",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("conversation_id", sa.UUID(), sa.ForeignKey("agent_conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task_id", sa.UUID(), nullable=False, unique=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("agent_conversation_turns_conversation_id_idx", "agent_conversation_turns", ["conversation_id"])


def downgrade() -> None:
    op.drop_table("agent_conversation_turns")
    op.drop_table("agent_conversations")
