"""Add task lifecycle and safe-resume receipt.

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "a7b8c9d0e1f2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_conversation_turns", sa.Column("termination", sa.String(40), nullable=True))
    op.add_column("agent_conversation_turns", sa.Column("recovery", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")))


def downgrade() -> None:
    op.drop_column("agent_conversation_turns", "recovery")
    op.drop_column("agent_conversation_turns", "termination")
