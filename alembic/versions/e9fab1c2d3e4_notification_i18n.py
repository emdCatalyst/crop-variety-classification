"""notification_i18n

Revision ID: e9fab1c2d3e4
Revises: d8e9fab1c2d3
Create Date: 2026-05-09 13:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e9fab1c2d3e4"
down_revision: Union[str, None] = "d8e9fab1c2d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("notifications") as batch:
        batch.add_column(sa.Column("i18n_key", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("i18n_params", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("notifications") as batch:
        batch.drop_column("i18n_params")
        batch.drop_column("i18n_key")
