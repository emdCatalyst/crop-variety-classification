from datetime import datetime

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    language: Mapped[str] = mapped_column(String(8), default="en")
    # Bumped on password change so any old JWTs (which carry the previous
    # token_version) fail the check in get_current_user.
    token_version: Mapped[int] = mapped_column(default=1, server_default="1")
    # Existing rows are migrated as verified (server_default "1"); new signups
    # default to False in application code so they must enter a code first.
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default="1")
    otp_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, default=None)
    otp_purpose: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
    otp_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    otp_attempts: Mapped[int] = mapped_column(default=0, server_default="0")

    analyses: Mapped[list["Analysis"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # type: ignore  # noqa: F821
