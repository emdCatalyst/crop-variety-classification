import enum

from sqlalchemy import Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class AnalysisStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Analysis(Base, TimestampMixin):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    source_name: Mapped[str] = mapped_column(String(200), default="uploaded_sequence")
    status: Mapped[AnalysisStatus] = mapped_column(Enum(AnalysisStatus, native_enum=False), default=AnalysisStatus.queued, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    upload_dir: Mapped[str] = mapped_column(String(512))
    smooth: Mapped[bool] = mapped_column(default=False)
    sigma: Mapped[float] = mapped_column(default=3.0)

    user: Mapped["User"] = relationship(back_populates="analyses")  # type: ignore  # noqa: F821
    images: Mapped[list["Image"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")  # type: ignore  # noqa: F821
    result: Mapped["Result | None"] = relationship(back_populates="analysis", uselist=False, cascade="all, delete-orphan")  # type: ignore  # noqa: F821
