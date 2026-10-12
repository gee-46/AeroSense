from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


class AnalyticsSummary(BaseModel):
    total_predictions: int
    avg_processing_time_ms: float
    avg_anomaly_score: float
    status_distribution: Dict[str, int]


class AnomalyTrendPoint(BaseModel):
    timestamp: datetime
    anomaly_score: float
    status: str


class ModelInfo(BaseModel):
    name: str
    type: str                       # "classifier" | "anomaly_detector"
    configured: bool                # a checkpoint file exists at the configured path
    loaded: bool                    # weights were loaded into memory successfully
    trained: bool                   # loaded AND declared trained in registry.json
    meta: Dict[str, Any]


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    registry: Dict[str, Any]


class ActiveModelsResponse(BaseModel):
    classifier: Optional[str] = None
    autoencoder: Optional[str] = None
    anomaly_threshold: float
    anomaly_threshold_source: str
