"""email_verification_and_reset

Revision ID: d8e9fab1c2d3
Revises: c7d8e9fab1c2
Create Date: 2026-05-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8e9fab1c2d3"
down_revision: Union[str, None] = "c7d8e9fab1c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        # Existing accounts default to verified ("1") so seeded test users and
        # any users created before this migration aren't retroactively locked
        # out. New signups set email_verified=False explicitly in app code.
        batch.add_column(
            sa.Column(
                "email_verified",
                sa.Boolean(),
                nullable=False,
                server_default="1",
            )
        )
        batch.add_column(sa.Column("otp_hash", sa.String(length=128), nullable=True))
        batch.add_column(sa.Column("otp_purpose", sa.String(length=16), nullable=True))
        batch.add_column(sa.Column("otp_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(
            sa.Column(
                "otp_attempts",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_column("otp_attempts")
        batch.drop_column("otp_expires_at")
        batch.drop_column("otp_purpose")
        batch.drop_column("otp_hash")
        batch.drop_column("email_verified")
