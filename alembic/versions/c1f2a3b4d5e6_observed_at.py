"""phase2_observed_at

Revision ID: c1f2a3b4d5e6
Revises: 73b80da1fdb5
Create Date: 2026-05-07 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1f2a3b4d5e6"
down_revision: Union[str, None] = "73b80da1fdb5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("results") as batch:
        batch.add_column(sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("results") as batch:
        batch.drop_column("observed_at")
