import uuid

from fastapi import APIRouter, UploadFile, File, Form, Depends, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.core.exceptions import AeroSenseError
from app.core.logging import logger
from app.db.database import get_db
from app.db import repository
from app.services import audio_service, inference_service
from app.services.explainability_service import delete_overlay
from app.schemas.audio import AudioMetadata
from app.schemas.prediction import PredictionResult
from app.api.routes.predictions import to_prediction_result

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])

_MAX_RPM = 200_000


@router.post(
    "/upload",
    response_model=AudioMetadata,
    summary="Upload an audio file and get back its validated metadata",
    description="Validates format/size/duration and returns decoded metadata (duration, sample rate, "
                "channels) without running inference. The file is deleted before the response is "
                "returned. Useful for a frontend to sanity-check a file before calling /analyze.",
    responses={
        413: {"description": "File exceeds MAX_UPLOAD_SIZE_MB"},
        415: {"description": "Unsupported file extension"},
        422: {"description": "Empty, undecodable or too-long audio"},
    },
)
async def upload_audio(file: UploadFile = File(...)):
    audio_service.validate_extension(file.filename)
    content = await audio_service.read_upload(file)

    path = audio_service.save_upload(content, file.filename)
    try:
        meta = await run_in_threadpool(audio_service.probe_audio, path)
    finally:
        audio_service.cleanup_file(path)

    return AudioMetadata(
        filename=audio_service.display_filename(file.filename),
        size_bytes=len(content),
        **meta,
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
                "a fabricated result. The uploaded audio is deleted after the request, whether "
                "it succeeded or failed; only derived results are retained.",
    responses={
        422: {"description": "Invalid or undecodable audio, or invalid form field"},
        413: {"description": "File exceeds MAX_UPLOAD_SIZE_MB"},
        415: {"description": "Unsupported file extension"},
        503: {"description": "Model not loaded / not yet trained"},
    },
)
async def analyze_audio(
    file: UploadFile = File(...),
    device_id: str | None = Form(None, max_length=128),
    environment: str | None = Form(None, max_length=128),
    operating_condition: str | None = Form(None, max_length=128),
    rpm: float | None = Form(None, ge=0, le=_MAX_RPM, allow_inf_nan=False),
    db: Session = Depends(get_db),
):
    audio_service.validate_extension(file.filename)
    content = await audio_service.read_upload(file)
    filename = audio_service.display_filename(file.filename)

    prediction_id = str(uuid.uuid4())
    path = audio_service.save_upload(content, file.filename)
    heatmap_path = None
    persisted = False

    try:
        meta = await run_in_threadpool(audio_service.probe_audio, path)
        # Inference is CPU-bound; run it off the event loop so health checks and
        # other requests stay responsive while a clip is being analyzed.
        result = await run_in_threadpool(
            inference_service.run_full_analysis, path, prediction_id, True
        )

        explanation_dict = result["explanation"]
        if explanation_dict:
            heatmap_path = explanation_dict["heatmap_path"]

        record = repository.create_prediction(
            db,
            prediction_id=prediction_id,
            filename=filename,
            stored_path=None,
            device_id=device_id or None,
            environment=environment or None,
            operating_condition=operating_condition or None,
            rpm=rpm,
            duration_seconds=meta["duration_seconds"],
            sample_rate=meta["sample_rate"],
            channels=meta["channels"],
            audio_format=meta["format"],
            size_bytes=len(content),
            model_version=result["model_version"],
            autoencoder_version=result["autoencoder_version"],
            status=result["status"],
            fault_type=result["fault_type"],
            fault_confidence=result["fault_confidence"],
            anomaly_score=result["anomaly_score"],
            anomaly_threshold=result["anomaly_threshold"],
            severity=result["severity"],
            class_probabilities=result["class_probabilities"],
            explanation_path=heatmap_path,
            explanation=explanation_dict,
            processing_time_ms=result["processing_time_ms"],
        )
        persisted = True
        return to_prediction_result(record)
    except Exception as e:
        if not isinstance(e, AeroSenseError):
            logger.exception("Unexpected error analyzing an upload")
        raise
    finally:
        audio_service.cleanup_file(path)
        if heatmap_path and not persisted:
            # The record was never written, so nothing would ever reference or
            # delete this image; remove it rather than orphan it.
            delete_overlay(heatmap_path)
