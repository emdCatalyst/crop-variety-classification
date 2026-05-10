from datetime import datetime

from pydantic import BaseModel


class AnalysisOut(BaseModel):
    id: int
    source_name: str
    status: str
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ResultOut(BaseModel):
    predicted_crop: str
    health_status: str
    trajectory: str
    advice: str
    mean_confidence: float
    mean_ndvi: float
    mean_ndre: float
    mean_savi: float
    temporal_trend: float
    map_url: str
    confidence_url: str | None = None
    class_distribution: dict
    legend: dict | None = None
    farmer_notes: str | None = None
    observed_at: datetime | None = None

    class Config:
        from_attributes = True


class AnalysisDetailOut(AnalysisOut):
    result: ResultOut | None = None


class ReportListOut(BaseModel):
    id: int
    source_name: str
    status: str
    created_at: datetime
    predicted_crop: str | None = None
    health_status: str | None = None
    farmer_notes: str | None = None
    observed_at: datetime | None = None
    has_result: bool = False


class FarmerNotesIn(BaseModel):
    farmer_notes: str | None = None
    observed_at: datetime | None = None
