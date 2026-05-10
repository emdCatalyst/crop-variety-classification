from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class Result(Base, TimestampMixin):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), unique=True, index=True)
    predicted_crop: Mapped[str] = mapped_column(String(120))
    health_status: Mapped[str] = mapped_column(String(32))
    trajectory: Mapped[str] = mapped_column(String(64))
    advice: Mapped[str] = mapped_column(Text)
    mean_confidence: Mapped[float] = mapped_column()
    mean_ndvi: Mapped[float] = mapped_column()
    mean_ndre: Mapped[float] = mapped_column()
    mean_savi: Mapped[float] = mapped_column()
    temporal_trend: Mapped[float] = mapped_column()
    map_png_path: Mapped[str] = mapped_column(String(512))
    map_url: Mapped[str] = mapped_column(String(512))
    confidence_png_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    confidence_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    geotiff_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    class_distribution: Mapped[dict] = mapped_column(JSON)
    legend: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    farmer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    analysis: Mapped["Analysis"] = relationship(back_populates="result")  # type: ignore  # noqa: F821
