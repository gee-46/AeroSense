"""
AeroSense shared configuration.

This module is the SINGLE SOURCE OF TRUTH for audio/feature parameters.
It is imported by both the training pipeline (training/) and the inference
backend (backend/app/ml) so that a model is always fed features that were
extracted the exact same way it was trained on.

All values can be overridden via environment variables so the same code
works in dev, training, and containerized deployment without edits.
"""
import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class AudioConfig:
    # --- Raw audio handling ---
    sample_rate: int = _env_int("SAMPLE_RATE", 16000)      # matches source dataset (verified from files)
    mono: bool = True

    # --- Segmentation ---
    # The source recordings are full missions (39s-90s). We cut them into
    # fixed-length segments so the model sees a consistent input size and so
    # we get enough training examples out of a modest number of recordings.
    segment_duration: float = _env_float("SEGMENT_DURATION", 2.0)   # seconds
    segment_hop: float = _env_float("SEGMENT_HOP", 1.0)             # seconds (50% overlap when training)

    # --- STFT / Mel spectrogram ---
    n_fft: int = _env_int("N_FFT", 1024)
    hop_length: int = _env_int("HOP_LENGTH", 256)
    win_length: int = _env_int("WIN_LENGTH", 1024)
    n_mels: int = _env_int("N_MELS", 64)
    fmin: int = _env_int("FMIN", 20)
    fmax: int = _env_int("FMAX", 8000)   # Nyquist for 16kHz audio

    # --- Normalization ---
    # Computed from the TRAINING split only (see training/prepare_dataset.py)
    # and stored in models/feature_stats.json, then reused at inference time.
    top_db: float = _env_float("TOP_DB", 80.0)  # dynamic range clip for log-mel

    @property
    def segment_samples(self) -> int:
        return int(self.segment_duration * self.sample_rate)

    @property
    def hop_samples(self) -> int:
        return int(self.segment_hop * self.sample_rate)

    @property
    def n_frames(self) -> int:
        """Expected number of STFT frames for one segment (mel-spectrogram time axis)."""
        return 1 + self.segment_samples // self.hop_length


AUDIO_CONFIG = AudioConfig()

# Class taxonomy is dataset-defined, NOT invented. This backend serves the
# real classes present in the TII UAV Propeller Anomaly Audio Dataset:
# https://github.com/tiiuae/UAV-Propeller-Anomaly-Audio-Dataset (ICASSP 2022)
CLASS_NAMES = ["normal", "1_broken", "2_broken", "3_4_broken"]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
DATA_RAW_DIR = os.environ.get("DATA_RAW_DIR", "./data/raw")
DATA_PROCESSED_DIR = os.environ.get("DATA_PROCESSED_DIR", "./data/processed")

RANDOM_SEED = _env_int("RANDOM_SEED", 42)

# Anomaly threshold is NOT hard-coded arbitrarily: it is computed by
# evaluation/evaluate.py from validation-set reconstruction errors (see
# models/registry.json -> autoencoder.anomaly_threshold) and only defaulted
# here in case that file is missing.
DEFAULT_ANOMALY_THRESHOLD = _env_float("ANOMALY_THRESHOLD", 0.5)
