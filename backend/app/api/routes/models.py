import os

from fastapi import APIRouter

from app.core.config import settings
from app.ml.model_loader import get_model_registry
from app.schemas.analytics import ActiveModelsResponse, ModelsListResponse

router = APIRouter(prefix="/api/v1/models", tags=["models"])

# Keys of registry.json that hold server-side file locations. They are not
# useful to API clients and would expose the deployment's directory layout.
_PATH_KEYS = {"path", "manifest_path", "feature_stats_path"}


def _public_registry(node):
    if isinstance(node, dict):
        return {k: _public_registry(v) for k, v in node.items() if k not in _PATH_KEYS}
    if isinstance(node, list):
        return [_public_registry(v) for v in node]
    return node


@router.get("", response_model=ModelsListResponse,
            summary="List known models: configured vs loaded vs trained, with real metadata")
def list_models():
    registry = get_model_registry()
    return {
        "models": [
            {
                "name": "aerosense_cnn_v1",
                "type": "classifier",
                "configured": os.path.exists(settings.CLASSIFIER_MODEL_PATH),
                "loaded": registry.classifier_loaded,
                "trained": registry.classifier_is_trained(),
                "meta": registry.classifier_meta,
            },
            {
                "name": "aerosense_autoencoder_v1",
                "type": "anomaly_detector",
                "configured": os.path.exists(settings.AUTOENCODER_MODEL_PATH),
                "loaded": registry.autoencoder_loaded,
                "trained": registry.autoencoder_is_trained(),
                "meta": registry.autoencoder_meta,
            },
        ],
        "registry": _public_registry(registry.registry_meta),
    }


@router.get("/active", response_model=ActiveModelsResponse, summary="Currently active model versions")
def active_models():
    registry = get_model_registry()
    return {
        "classifier": registry.classifier_meta.get("version") if registry.classifier_loaded else None,
        "autoencoder": registry.autoencoder_meta.get("version") if registry.autoencoder_loaded else None,
        "anomaly_threshold": registry.anomaly_threshold,
        "anomaly_threshold_source": registry.anomaly_threshold_source,
    }
