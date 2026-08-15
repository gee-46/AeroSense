"""
Loads trained model checkpoints ONCE and keeps them in memory (singleton
pattern) so inference requests never pay model-loading cost.

If a checkpoint file is missing, the corresponding `is_loaded` flag is
False and app.core.exceptions.ModelUnavailableError is raised by the
inference service rather than the API silently returning fabricated
predictions.
"""
import json
import os
import threading

import torch

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ModelUnavailableError
from app.ml.models.cnn import AeroSenseCNN
from app.ml.models.autoencoder import ConvAutoencoder


class ModelRegistry:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelRegistry initializing on device={self.device}")

        self.classifier = None
        self.classifier_meta = {}
        self.classifier_loaded = False

        self.autoencoder = None
        self.autoencoder_meta = {}
        self.autoencoder_loaded = False

        self.class_names = settings.CLASS_NAMES
        self.feature_mean = 0.0
        self.feature_std = 1.0
        self.anomaly_threshold = settings.ANOMALY_THRESHOLD
        self.registry_meta = {}

        self._load_feature_stats()
        self._load_registry()
        self._load_classifier()
        self._load_autoencoder()

    def _load_feature_stats(self):
        try:
            with open(settings.FEATURE_STATS_PATH) as f:
                stats = json.load(f)
            self.feature_mean = stats["log_mel_mean"]
            self.feature_std = stats["log_mel_std"]
            logger.info(f"Loaded feature stats: mean={self.feature_mean:.3f} std={self.feature_std:.3f}")
        except FileNotFoundError:
            logger.warning(f"feature_stats.json not found at {settings.FEATURE_STATS_PATH}; "
                            f"using mean=0, std=1 (UNSTANDARDIZED — predictions will be unreliable)")

    def _load_registry(self):
        try:
            with open(settings.REGISTRY_PATH) as f:
                self.registry_meta = json.load(f)
            ae_threshold = self.registry_meta.get("autoencoder", {}).get("anomaly_threshold")
            if ae_threshold is not None:
                self.anomaly_threshold = ae_threshold
                logger.info(f"Loaded anomaly threshold from registry.json: {self.anomaly_threshold}")
        except FileNotFoundError:
            logger.warning(f"registry.json not found at {settings.REGISTRY_PATH}; "
                            f"using default anomaly threshold {self.anomaly_threshold} "
                            f"(run evaluation/evaluate.py to fit a real one)")

    def _load_classifier(self):
        path = settings.CLASSIFIER_MODEL_PATH
        if not os.path.exists(path):
            logger.error(f"Classifier checkpoint not found at {path}. "
                         f"Classifier endpoints will return MODEL_UNAVAILABLE until trained "
                         f"(see training/train_cnn.py).")
            return
        try:
            ckpt = torch.load(path, map_location=self.device)
            model = AeroSenseCNN(num_classes=ckpt.get("num_classes", len(self.class_names)))
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.classifier = model
            self.classifier_meta = {
                "version": "aerosense_cnn_v1",
                "epoch": ckpt.get("epoch"),
                "val_macro_f1": ckpt.get("val_macro_f1"),
                "val_accuracy": ckpt.get("val_accuracy"),
                "class_names": ckpt.get("class_names", self.class_names),
            }
            self.classifier_loaded = True
            logger.info(f"Loaded classifier from {path}: {self.classifier_meta}")
        except Exception as e:
            logger.exception(f"Failed to load classifier from {path}: {e}")

    def _load_autoencoder(self):
        path = settings.AUTOENCODER_MODEL_PATH
        if not os.path.exists(path):
            logger.error(f"Autoencoder checkpoint not found at {path}. "
                         f"Anomaly-score endpoints will return MODEL_UNAVAILABLE until trained "
                         f"(see training/train_autoencoder.py).")
            return
        try:
            ckpt = torch.load(path, map_location=self.device)
            model = ConvAutoencoder()
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.autoencoder = model
            self.autoencoder_meta = {
                "version": "aerosense_autoencoder_v1",
                "epoch": ckpt.get("epoch"),
                "val_loss": ckpt.get("val_loss"),
            }
            self.autoencoder_loaded = True
            logger.info(f"Loaded autoencoder from {path}: {self.autoencoder_meta}")
        except Exception as e:
            logger.exception(f"Failed to load autoencoder from {path}: {e}")

    def require_classifier(self):
        if not self.classifier_loaded:
            raise ModelUnavailableError(
                "Classifier model is not loaded. Train it with `python -m training.train_cnn` "
                "and ensure CLASSIFIER_MODEL_PATH points at the resulting checkpoint."
            )
        return self.classifier

    def require_autoencoder(self):
        if not self.autoencoder_loaded:
            raise ModelUnavailableError(
                "Autoencoder model is not loaded. Train it with `python -m training.train_autoencoder` "
                "and ensure AUTOENCODER_MODEL_PATH points at the resulting checkpoint."
            )
        return self.autoencoder


_registry_instance = None
_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """Module-level accessor guaranteeing the registry (and its checkpoints)
    is constructed exactly once for the life of the process."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ModelRegistry()
    return _registry_instance
