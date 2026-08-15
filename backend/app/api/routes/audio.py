import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, Depends, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import AeroSenseError
from app.core.logging import logger
from app.db.database import get_db
from app.db import repository
from app.services import audio_service, inference_service
from app.schemas.audio import AudioMetadata
from app.schemas.prediction import PredictionResult, Explanation

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])


@router.post(
    "/upload",
    response_model=AudioMetadata,
    summary="Upload an audio file and get back its validated metadata",
    description="Validates format/size and returns decoded metadata (duration, sample rate, "
                "channels) without running inference. Useful for a frontend to sanity-check a "
                "file before calling /analyze.",
)
async def upload_audio(file: UploadFile = File(...)):
    audio_service.validate_extension(file.filename)
    content = await file.read()
    audio_service.validate_size(len(content))

    path = audio_service.save_upload(content, file.filename)
    try:
        meta = audio_service.probe_audio(path)
    finally:
        audio_service.cleanup_file(path)

    return AudioMetadata(
        filename=file.filename,
        duration_seconds=meta["duration_seconds"],
        sample_rate=meta["sample_rate"],
        channels=meta["channels"],
        format=meta["format"],
        size_bytes=len(content),
    )


@router.post(
    "/analyze",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Run full anomaly-detection pipeline on an uploaded audio clip",
    description="Upload -> validate -> preprocess -> mel-spectrogram -> trained CNN classifier "
                "+ trained autoencoder -> decision engine -> Grad-CAM explanation -> persist -> "
                "JSON response. All model outputs come from real trained checkpoints; if a "
                "checkpoint is missing this endpoint returns 503 MODEL_UNAVAILABLE rather than "
                "a fabricated result.",
    responses={
        422: {"description": "Invalid or undecodable audio"},
        413: {"description": "File exceeds MAX_UPLOAD_SIZE_MB"},
        415: {"description": "Unsupported file extension"},
        503: {"description": "Model not loaded / not yet trained"},
    },
)
async def analyze_audio(
    file: UploadFile = File(...),
    device_id: str | None = Form(None),
    environment: str | None = Form(None),
    operating_condition: str | None = Form(None),
    rpm: float | None = Form(None),
    model_version: str | None = Form(None),
    db: Session = Depends(get_db),
):
    audio_service.validate_extension(file.filename)
    content = await file.read()
    audio_service.validate_size(len(content))

    prediction_id = str(uuid.uuid4())
    path = audio_service.save_upload(content, file.filename)

    try:
        meta = audio_service.probe_audio(path)
        result = inference_service.run_full_analysis(path, prediction_id, with_explanation=True)

        explanation_path = None
        explanation_obj = None
        if result["explanation"]:
            explanation_path = result["explanation"]["heatmap_path"]
            explanation_obj = Explanation(**result["explanation"])

        record = repository.create_prediction(
            db,
            prediction_id=prediction_id,
            filename=file.filename,
            stored_path=path,
            device_id=device_id,
            environment=environment,
            operating_condition=operating_condition,
            rpm=rpm,
            duration_seconds=meta["duration_seconds"],
            sample_rate=meta["sample_rate"],
            model_version=result["model_version"],
            autoencoder_version=result["autoencoder_version"],
            status=result["status"],
            fault_type=result["fault_type"],
            fault_confidence=result["fault_confidence"],
            anomaly_score=result["anomaly_score"],
            anomaly_threshold=result["anomaly_threshold"],
            severity=result["severity"],
            class_probabilities=result["class_probabilities"],
            explanation_path=explanation_path,
            processing_time_ms=result["processing_time_ms"],
        )

        return PredictionResult(
            prediction_id=record.prediction_id,
            status=record.status,
            fault_type=record.fault_type,
            fault_confidence=record.fault_confidence,
            anomaly_score=record.anomaly_score,
            anomaly_threshold=record.anomaly_threshold,
            severity=record.severity,
            model_version=record.model_version,
            autoencoder_version=record.autoencoder_version,
            processing_time_ms=record.processing_time_ms,
            timestamp=record.timestamp,
            audio=AudioMetadata(
                filename=file.filename,
                duration_seconds=meta["duration_seconds"],
                sample_rate=meta["sample_rate"],
                channels=meta["channels"],
                format=meta["format"],
                size_bytes=len(content),
            ),
            class_probabilities=record.class_probabilities,
            explanation=explanation_obj,
            device_id=record.device_id,
            environment=record.environment,
            operating_condition=record.operating_condition,
            rpm=record.rpm,
        )
    except AeroSenseError:
        audio_service.cleanup_file(path)
        raise
    except Exception as e:
        audio_service.cleanup_file(path)
        logger.exception(f"Unexpected error analyzing {file.filename}")
        raise
