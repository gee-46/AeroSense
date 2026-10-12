from typing import Optional, Any, List
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    app_env: str
    version: str


class ModelHealthResponse(BaseModel):
    classifier_loaded: bool
    classifier_version: Optional[str] = None
    classifier_is_trained: bool   # loaded AND declared trained in registry.json
    autoencoder_loaded: bool
    autoencoder_version: Optional[str] = None
    autoencoder_is_trained: bool
    device: str
    anomaly_threshold: Optional[float] = None
    anomaly_threshold_source: str = "default"   # "registry.json" | "default"
    feature_stats_loaded: bool = False
    consistency_warnings: List[str] = []
    details: Optional[Any] = None
