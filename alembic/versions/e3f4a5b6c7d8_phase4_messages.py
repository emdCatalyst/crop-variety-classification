"""phase4_messages

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-05-08 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e3f4a5b6c7d8"
down_revision: Union[str, None] = "d2e3f4a5b6c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sender_id", sa.Integer(), nullable=False),
        sa.Column("recipient_id", sa.Integer(), nullable=False),
        sa.Column("thread_key", sa.String(length=64), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("attachment_path", sa.String(length=500), nullable=True),
        sa.Column("attachment_mime", sa.String(length=64), nullable=True),
        sa.Column("attachment_name", sa.String(length=200), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "archived",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["sender_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["recipient_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_messages_sender_id"), "messages", ["sender_id"], unique=False)
    op.create_index(op.f("ix_messages_recipient_id"), "messages", ["recipient_id"], unique=False)
    op.create_index(op.f("ix_messages_thread_key"), "messages", ["thread_key"], unique=False)
    op.create_index(
        "ix_messages_thread_created", "messages", ["thread_key", "created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_messages_thread_created", table_name="messages")
    op.drop_index(op.f("ix_messages_thread_key"), table_name="messages")
    op.drop_index(op.f("ix_messages_recipient_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_sender_id"), table_name="messages")
    op.drop_table("messages")
