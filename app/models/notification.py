from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class Notification(Base, TimestampMixin):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    analysis_id: Mapped[int | None] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), nullable=True, index=True
    )
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # When set, the frontend renders title/body via the matching i18n keys
    # ("notifications.system.<i18n_key>.{title,body}") with these params,
    # so notification text follows the user's *current* language. Admin
    # notices and broadcasts leave these null, so their verbatim title/body
    # is shown as the admin wrote it.
    i18n_key: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    i18n_params: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True, default=None)
