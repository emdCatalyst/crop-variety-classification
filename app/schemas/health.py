from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["CRITICAL", "STRESSED", "MODERATE", "HEALTHY", "VIGOROUS"]
Trajectory = Literal[
    "STRONG_DECLINE", "DECLINE", "STABLE", "GROWTH", "STRONG_GROWTH"
]


class HealthAssessment(BaseModel):
    health_status: HealthStatus
    trajectory: Trajectory
    advice: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    citations: list[str] = Field(default_factory=list)
    source: str = "heuristic"
