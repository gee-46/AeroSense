from fastapi import APIRouter

from app.core.config import settings
from app.ml.model_loader import get_model_registry
from app.schemas.common import HealthResponse, ModelHealthResponse

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Basic liveness check")
def health():
    """Returns 200 if the API process is up. Does not guarantee models are loaded —
    use GET /api/v1/health/model for that."""
    return HealthResponse(status="ok", app_env=settings.APP_ENV, version="0.1.0")


@router.get("/model", response_model=ModelHealthResponse, summary="Model loading status")
def model_health():
    """
    Reports whether the classifier and autoencoder checkpoints are actually
    loaded in memory, and whether they are real trained weights
    (is_trained=True) as opposed to an uninitialized architecture.
    """
    registry = get_model_registry()
    return ModelHealthResponse(
        classifier_loaded=registry.classifier_loaded,
        classifier_version=registry.classifier_meta.get("version"),
        classifier_is_trained=registry.classifier_loaded,
        autoencoder_loaded=registry.autoencoder_loaded,
        autoencoder_version=registry.autoencoder_meta.get("version"),
        autoencoder_is_trained=registry.autoencoder_loaded,
        device=str(registry.device),
        anomaly_threshold=registry.anomaly_threshold,
        details={
            "classifier_meta": registry.classifier_meta,
            "autoencoder_meta": registry.autoencoder_meta,
        },
    )
