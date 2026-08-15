from fastapi import APIRouter

from app.ml.model_loader import get_model_registry

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", summary="List all known models and their real training metadata")
def list_models():
    registry = get_model_registry()
    return {
        "models": [
            {
                "name": "aerosense_cnn_v1",
                "type": "classifier",
                "loaded": registry.classifier_loaded,
                "meta": registry.classifier_meta,
            },
            {
                "name": "aerosense_autoencoder_v1",
                "type": "anomaly_detector",
                "loaded": registry.autoencoder_loaded,
                "meta": registry.autoencoder_meta,
            },
        ],
        "registry": registry.registry_meta,
    }


@router.get("/active", summary="Currently active model versions")
def active_models():
    registry = get_model_registry()
    return {
        "classifier": registry.classifier_meta.get("version") if registry.classifier_loaded else None,
        "autoencoder": registry.autoencoder_meta.get("version") if registry.autoencoder_loaded else None,
        "anomaly_threshold": registry.anomaly_threshold,
    }
