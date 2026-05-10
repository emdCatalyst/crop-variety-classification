"""message_conversation_id

Revision ID: c7d8e9fab1c2
Revises: b6c7d8e9fab1
Create Date: 2026-05-09 01:30:00.000000

Adds conversation_id to messages so a single thread_key can hold multiple
sealed-and-frozen conversations. Existing rows are backfilled with one
conversation_id per thread_key (each pair currently has a single conversation
in flight, by definition).
"""
from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa


revision: str = "c7d8e9fab1c2"
down_revision: Union[str, None] = "b6c7d8e9fab1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("messages") as batch:
        batch.add_column(
            sa.Column("conversation_id", sa.String(length=40), nullable=True)
        )

    bind = op.get_bind()
    rows = bind.execute(
        sa.text("SELECT DISTINCT thread_key FROM messages WHERE conversation_id IS NULL")
    ).fetchall()
    for (key,) in rows:
        cid = uuid.uuid4().hex
        bind.execute(
            sa.text(
                "UPDATE messages SET conversation_id = :cid WHERE thread_key = :k"
            ),
            {"cid": cid, "k": key},
        )

    with op.batch_alter_table("messages") as batch:
        batch.alter_column("conversation_id", existing_type=sa.String(length=40), nullable=False)
        batch.create_index(
            "ix_messages_conversation_id", ["conversation_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("messages") as batch:
        batch.drop_index("ix_messages_conversation_id")
        batch.drop_column("conversation_id")
