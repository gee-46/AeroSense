from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from app.schemas.audio import AudioMetadata


class Explanation(BaseModel):
    method: str = Field(..., description="Explainability method used, e.g. 'grad_cam'")
    important_frequency_range_hz: Optional[List[float]] = None
    important_time_range_seconds: Optional[List[float]] = None
    heatmap_path: Optional[str] = Field(None, description="Server-relative path to the saved attribution overlay image")
    notes: Optional[str] = None


class PredictionResult(BaseModel):
    prediction_id: str
    status: str = Field(..., description="NORMAL | ANOMALY | UNKNOWN_ANOMALY | ERROR")
    fault_type: Optional[str] = None
    fault_confidence: Optional[float] = None
    anomaly_score: float
    anomaly_threshold: float
    severity: Optional[str] = Field(None, description="LOW | MEDIUM | HIGH | CRITICAL")
    model_version: str
    autoencoder_version: str
    processing_time_ms: float
    timestamp: datetime
    audio: AudioMetadata
    class_probabilities: dict
    explanation: Optional[Explanation] = None
    device_id: Optional[str] = None
    environment: Optional[str] = None
    operating_condition: Optional[str] = None
    rpm: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class PredictionListItem(BaseModel):
    prediction_id: str
    status: str
    fault_type: Optional[str]
    fault_confidence: Optional[float]
    anomaly_score: float
    severity: Optional[str]
    model_version: str
    timestamp: datetime
    device_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PredictionListResponse(BaseModel):
    total: int
    items: List[PredictionListItem]
