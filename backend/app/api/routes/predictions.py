from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import repository
from app.db.models import Prediction
from app.schemas.prediction import (
    PredictionResult, PredictionListResponse, PredictionListItem, Explanation, PredictionStatus,
)
from app.schemas.audio import AudioMetadata
from app.services.explainability_service import delete_overlay
from app.utils.validation import validate_pagination

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


def to_prediction_result(p: Prediction) -> PredictionResult:
    """Single mapping from a stored record to the API response, shared by
    /audio/analyze and GET /predictions/{id} so the two can never disagree."""
    explanation: Optional[Explanation] = None
    if p.explanation:
        explanation = Explanation(**p.explanation)
    elif p.explanation_path:   # rows written before full explanations were persisted
        explanation = Explanation(method="grad_cam", heatmap_path=p.explanation_path)
    return PredictionResult(
        prediction_id=p.prediction_id,
        status=p.status,
        fault_type=p.fault_type,
        fault_confidence=p.fault_confidence,
        anomaly_score=p.anomaly_score,
        anomaly_threshold=p.anomaly_threshold,
        severity=p.severity,
        model_version=p.model_version,
        autoencoder_version=p.autoencoder_version,
        processing_time_ms=p.processing_time_ms,
        timestamp=p.timestamp,
        audio=AudioMetadata(
            filename=p.filename,
            duration_seconds=p.duration_seconds,
            sample_rate=p.sample_rate,
            # Rows predating these columns have no recorded value; report that
            # honestly instead of inventing one.
            channels=p.channels if p.channels is not None else 0,
            format=p.audio_format or "unknown",
            size_bytes=p.size_bytes if p.size_bytes is not None else 0,
        ),
        class_probabilities=p.class_probabilities or {},
        explanation=explanation,
        device_id=p.device_id,
        environment=p.environment,
        operating_condition=p.operating_condition,
        rpm=p.rpm,
    )


@router.get("", response_model=PredictionListResponse, summary="List past predictions (newest first)")
def list_predictions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[PredictionStatus] = Query(None, description="Filter by status"),
    device_id: Optional[str] = Query(None, max_length=128),
    db: Session = Depends(get_db),
):
    limit, offset = validate_pagination(limit, offset)
    total, items = repository.list_predictions(db, limit=limit, offset=offset, status=status, device_id=device_id)
    return PredictionListResponse(
        total=total,
        items=[PredictionListItem.model_validate(p) for p in items],
    )


@router.get("/{prediction_id}", response_model=PredictionResult, summary="Get one prediction by ID")
def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    return to_prediction_result(repository.get_prediction(db, prediction_id))


@router.delete("/{prediction_id}", status_code=204, summary="Delete a prediction record and its Grad-CAM image")
def delete_prediction(prediction_id: str, db: Session = Depends(get_db)):
    heatmap = repository.delete_prediction(db, prediction_id)
    delete_overlay(heatmap)
    return None
