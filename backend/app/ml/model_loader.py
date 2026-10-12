"""
Loads trained model checkpoints ONCE and keeps them in memory (singleton
pattern) so inference requests never pay model-loading cost.

If a checkpoint file is missing, the corresponding `is_loaded` flag is
False and app.core.exceptions.ModelUnavailableError is raised by the
inference service rather than the API silently returning fabricated
predictions.

Three distinct states are tracked per model and must not be conflated:
  * configured  - a checkpoint path is configured (see settings)
  * loaded      - the checkpoint exists and its weights fit the architecture
  * trained     - loaded AND declared trained in models/registry.json
"""
import json
import math
import os
import threading

import torch

from app.core.config import settings
from common.config import AUDIO_CONFIG
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
        self.anomaly_threshold_source = "default"   # becomes "registry.json" when loaded from it
        self.registry_meta = {}
        # False when feature_stats.json is missing/invalid/inconsistent with the
        # active audio config. Inference is refused in that case, because
        # silently using mean=0/std=1 would feed the models inputs unlike
        # anything they were trained on.
        self.feature_stats_loaded = False
        self.consistency_warnings: list[str] = []
        # Serializes forward passes and Grad-CAM (which hooks the shared model)
        # so concurrent requests cannot interleave on the same module, and so
        # peak CPU memory stays bounded to one inference at a time.
        self.inference_lock = threading.RLock()

        self._load_feature_stats()
        self._load_registry()
        self._load_classifier()
        self._load_autoencoder()

    # ------------------------------------------------------------------
    def _load_feature_stats(self):
        try:
            with open(settings.FEATURE_STATS_PATH) as f:
                stats = json.load(f)
            mean = float(stats["log_mel_mean"])
            std = float(stats["log_mel_std"])
        except FileNotFoundError:
            logger.error(f"feature_stats.json not found at {settings.FEATURE_STATS_PATH}; "
                         f"inference is disabled (predictions would be unstandardized).")
            return
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"feature_stats.json at {settings.FEATURE_STATS_PATH} is invalid: {e!r}; "
                         f"inference is disabled.")
            return
        if not (math.isfinite(mean) and math.isfinite(std) and std > 0):
            logger.error(f"feature_stats.json has non-finite mean/std or std<=0 "
                         f"(mean={mean}, std={std}); inference is disabled.")
            return

        # Training/inference consistency: the stats record the audio config they
        # were computed with. If it differs from the config the backend runs
        # with, the standardization is wrong for these features.
        recorded = stats.get("audio_config") or {}
        for key, active in (
            ("sample_rate", AUDIO_CONFIG.sample_rate),
            ("n_mels", AUDIO_CONFIG.n_mels),
            ("n_fft", AUDIO_CONFIG.n_fft),
            ("hop_length", AUDIO_CONFIG.hop_length),
            ("segment_duration", AUDIO_CONFIG.segment_duration),
        ):
            if key in recorded and recorded[key] != active:
                logger.error(f"feature_stats.json was computed with {key}={recorded[key]} but the "
                             f"active audio config has {key}={active}; inference is disabled.")
                return

        self.feature_mean, self.feature_std = mean, std
        self.feature_stats_loaded = True
        logger.info(f"Loaded feature stats: mean={mean:.3f} std={std:.3f}")

    def _load_registry(self):
        try:
            with open(settings.REGISTRY_PATH) as f:
                self.registry_meta = json.load(f)
        except FileNotFoundError:
            logger.warning(f"registry.json not found at {settings.REGISTRY_PATH}; "
                           f"using default anomaly threshold {self.anomaly_threshold} "
                           f"(run evaluation/evaluate.py to fit a real one)")
            return
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Could not read registry.json: {e!r}; using default anomaly threshold")
            self.registry_meta = {}
            return

        ae_threshold = (self.registry_meta.get("autoencoder") or {}).get("anomaly_threshold")
        if ae_threshold is None:
            logger.warning(f"registry.json has no autoencoder.anomaly_threshold; "
                           f"using default {self.anomaly_threshold}")
        elif isinstance(ae_threshold, (int, float)) and math.isfinite(ae_threshold) and ae_threshold > 0:
            self.anomaly_threshold = float(ae_threshold)
            self.anomaly_threshold_source = "registry.json"
            logger.info(f"Loaded anomaly threshold from registry.json: {self.anomaly_threshold}")
        else:
            logger.error(f"Ignoring invalid anomaly_threshold in registry.json: {ae_threshold!r}; "
                         f"falling back to default {self.anomaly_threshold}")

    # ------------------------------------------------------------------
    @staticmethod
    def _torch_load(path: str, device):
        # weights_only=True: checkpoints are plain dicts of tensors/primitives,
        # so refuse to unpickle arbitrary objects from the model directory.
        return torch.load(path, map_location=device, weights_only=True)

    def _note(self, msg: str):
        logger.warning(msg)
        self.consistency_warnings.append(msg)

    def _compare_with_registry(self, section: str, ckpt: dict, loaded_params: int, fields: dict):
        """A mismatch means registry.json describes a different checkpoint
        than the one that was loaded."""
        reg = self.registry_meta.get(section) or {}
        for ckpt_key, reg_key in fields.items():
            a, b = ckpt.get(ckpt_key), reg.get(reg_key)
            if a is None or b is None:
                continue
            same = math.isclose(a, b, rel_tol=1e-6) if isinstance(a, float) or isinstance(b, float) else a == b
            if not same:
                self._note(f"registry.json {section}.{reg_key}={b!r} does not match checkpoint "
                           f"{ckpt_key}={a!r}")
        reg_params = reg.get("n_params")
        if reg_params is not None and reg_params != loaded_params:
            self._note(f"registry.json {section}.n_params={reg_params} but the loaded model has {loaded_params}")

    def _load_classifier(self):
        path = settings.CLASSIFIER_MODEL_PATH
        if not os.path.exists(path):
            logger.error(f"Classifier checkpoint not found at {path}. "
                         f"Classifier endpoints will return MODEL_UNAVAILABLE until trained "
                         f"(see training/train_cnn.py).")
            return
        try:
            ckpt = self._torch_load(path, self.device)
            ckpt_classes = list(ckpt.get("class_names", self.class_names))
            # Class ordering maps output index -> label. A checkpoint trained
            # with a different order would silently mislabel every prediction.
            if ckpt_classes != list(self.class_names):
                logger.error(f"Classifier checkpoint class order {ckpt_classes} differs from the "
                             f"configured class order {list(self.class_names)}; NOT loading it.")
                return
            num_classes = int(ckpt.get("num_classes", len(self.class_names)))
            if num_classes != len(self.class_names):
                logger.error(f"Classifier checkpoint has {num_classes} classes but "
                             f"{len(self.class_names)} are configured; NOT loading it.")
                return
            model = AeroSenseCNN(num_classes=num_classes)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            self.classifier = model
            self.classifier_meta = {
                "version": "aerosense_cnn_v1",
                "epoch": ckpt.get("epoch"),
                "val_macro_f1": ckpt.get("val_macro_f1"),
                "val_accuracy": ckpt.get("val_accuracy"),
                "class_names": ckpt_classes,
                "n_params": n_params,
            }
            self._compare_with_registry("classifier", ckpt, n_params, {
                "epoch": "trained_epoch", "val_macro_f1": "val_macro_f1", "val_accuracy": "val_accuracy",
            })
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
            ckpt = self._torch_load(path, self.device)
            model = ConvAutoencoder()
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            self.autoencoder = model
            self.autoencoder_meta = {
                "version": "aerosense_autoencoder_v1",
                "epoch": ckpt.get("epoch"),
                "val_loss": ckpt.get("val_loss"),
                "n_params": n_params,
            }
            self._compare_with_registry("autoencoder", ckpt, n_params, {
                "epoch": "trained_epoch", "val_loss": "val_loss",
            })
            self.autoencoder_loaded = True
            logger.info(f"Loaded autoencoder from {path}: {self.autoencoder_meta}")
        except Exception as e:
            logger.exception(f"Failed to load autoencoder from {path}: {e}")

    # ------------------------------------------------------------------
    def classifier_is_trained(self) -> bool:
        """Loaded AND declared trained in registry.json. 'Loaded' alone only
        proves the weights fit the architecture, not that they were trained."""
        return bool(self.classifier_loaded and (self.registry_meta.get("classifier") or {}).get("is_trained") is True)

    def autoencoder_is_trained(self) -> bool:
        return bool(self.autoencoder_loaded and (self.registry_meta.get("autoencoder") or {}).get("is_trained") is True)

    def _require_features(self):
        if not self.feature_stats_loaded:
            raise ModelUnavailableError(
                "Feature statistics (feature_stats.json) are missing or inconsistent with the "
                "audio configuration; inference is disabled to avoid unreliable predictions."
            )

    def require_classifier(self):
        if not self.classifier_loaded:
            raise ModelUnavailableError(
                "Classifier model is not loaded. Train it with `python -m training.train_cnn` "
                "and ensure CLASSIFIER_MODEL_PATH points at the resulting checkpoint."
            )
        self._require_features()
        return self.classifier

    def require_autoencoder(self):
        if not self.autoencoder_loaded:
            raise ModelUnavailableError(
                "Autoencoder model is not loaded. Train it with `python -m training.train_autoencoder` "
                "and ensure AUTOENCODER_MODEL_PATH points at the resulting checkpoint."
            )
        self._require_features()
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
