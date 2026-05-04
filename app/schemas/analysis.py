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
    class_distribution: dict
    farmer_notes: str | None = None

    class Config:
        from_attributes = True


class AnalysisDetailOut(AnalysisOut):
    result: ResultOut | None = None
