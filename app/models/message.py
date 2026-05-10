from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class Message(Base, TimestampMixin):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    sender_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    recipient_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    thread_key: Mapped[str] = mapped_column(String(64), index=True)
    # One thread_key may span multiple conversations over time — each archive
    # seals the current conversation and the next user message starts a new
    # one. Conversations are the primary grouping unit shown in the UI.
    conversation_id: Mapped[str] = mapped_column(String(40), index=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    attachment_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    attachment_mime: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attachment_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")

    __table_args__ = (
        Index("ix_messages_thread_created", "thread_key", "created_at"),
    )
