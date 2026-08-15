import os
import sys

# Make the repo root importable so we can reuse common/ (shared with training/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.config import AUDIO_CONFIG, CLASS_NAMES, NUM_CLASSES, DEFAULT_ANOMALY_THRESHOLD  # noqa: E402


class Settings:
    APP_ENV: str = os.environ.get("APP_ENV", "development")
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", 8000))

    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./aerosense.db")

    MODEL_DIR: str = os.environ.get("MODEL_DIR", os.path.join(_REPO_ROOT, "models"))
    CLASSIFIER_MODEL_PATH: str = os.environ.get(
        "CLASSIFIER_MODEL_PATH", os.path.join(MODEL_DIR, "aerosense_cnn_v1.pt")
    )
    AUTOENCODER_MODEL_PATH: str = os.environ.get(
        "AUTOENCODER_MODEL_PATH", os.path.join(MODEL_DIR, "aerosense_autoencoder_v1.pt")
    )
    FEATURE_STATS_PATH: str = os.environ.get(
        "FEATURE_STATS_PATH", os.path.join(MODEL_DIR, "feature_stats.json")
    )
    REGISTRY_PATH: str = os.environ.get(
        "REGISTRY_PATH", os.path.join(MODEL_DIR, "registry.json")
    )

    UPLOAD_DIR: str = os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "uploads"))
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "outputs"))

    MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 25))
    ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}

    # Anomaly threshold: overridden by models/registry.json if present
    # (computed from real validation data by evaluation/evaluate.py).
    ANOMALY_THRESHOLD: float = DEFAULT_ANOMALY_THRESHOLD

    # Confidence below which a classifier prediction is not treated as a
    # confident "known fault" call — feeds the UNKNOWN_ANOMALY decision path.
    MIN_CLASSIFIER_CONFIDENCE: float = float(os.environ.get("MIN_CLASSIFIER_CONFIDENCE", 0.5))

    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

    AUDIO = AUDIO_CONFIG
    CLASS_NAMES = CLASS_NAMES
    NUM_CLASSES = NUM_CLASSES


settings = Settings()
