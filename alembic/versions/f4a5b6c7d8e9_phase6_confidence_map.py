"""phase6_confidence_map

Revision ID: f4a5b6c7d8e9
Revises: e3f4a5b6c7d8
Create Date: 2026-05-08 22:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f4a5b6c7d8e9"
down_revision: Union[str, None] = "e3f4a5b6c7d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("results") as batch:
        batch.add_column(
            sa.Column("confidence_png_path", sa.String(length=512), nullable=True)
        )
        batch.add_column(
            sa.Column("confidence_url", sa.String(length=512), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("results") as batch:
        batch.drop_column("confidence_url")
        batch.drop_column("confidence_png_path")
