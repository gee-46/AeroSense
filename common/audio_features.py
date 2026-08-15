"""
Shared audio preprocessing / feature extraction.

Used identically by:
  - training/prepare_dataset.py   (offline dataset preparation)
  - backend/app/ml/inference.py   (online API inference)

Keeping this in one place is what guarantees "no train/inference skew":
whatever transform a model was trained on is exactly what it receives
in production.
"""
from __future__ import annotations

import numpy as np
import librosa

from common.config import AudioConfig, AUDIO_CONFIG


def load_audio(path: str, cfg: AudioConfig = AUDIO_CONFIG) -> np.ndarray:
    """Load an audio file, convert to mono float32, resample to cfg.sample_rate."""
    audio, _sr = librosa.load(path, sr=cfg.sample_rate, mono=cfg.mono)
    return audio.astype(np.float32)


def normalize_waveform(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]. No-op on silence to avoid div-by-zero."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak


def segment_waveform(
    audio: np.ndarray,
    cfg: AudioConfig = AUDIO_CONFIG,
    min_last_segment_ratio: float = 0.5,
) -> list[np.ndarray]:
    """
    Cut a (possibly long) waveform into fixed-length, possibly-overlapping
    segments of cfg.segment_duration seconds (hop = cfg.segment_hop).

    The final partial segment is zero-padded and kept only if it covers at
    least `min_last_segment_ratio` of the target length (avoids emitting
    near-empty segments at the tail of a recording).
    """
    seg_len = cfg.segment_samples
    hop = cfg.hop_samples
    segments = []

    if len(audio) < seg_len:
        pad = np.zeros(seg_len, dtype=np.float32)
        pad[: len(audio)] = audio
        return [pad] if len(audio) >= seg_len * min_last_segment_ratio else []

    start = 0
    while start < len(audio):
        end = start + seg_len
        if end <= len(audio):
            segments.append(audio[start:end])
        else:
            remainder = audio[start:]
            if len(remainder) >= seg_len * min_last_segment_ratio:
                pad = np.zeros(seg_len, dtype=np.float32)
                pad[: len(remainder)] = remainder
                segments.append(pad)
            break
        start += hop
    return segments


def generate_mel_spectrogram(audio: np.ndarray, cfg: AudioConfig = AUDIO_CONFIG) -> np.ndarray:
    """Log-scaled Mel spectrogram. Shape: (n_mels, n_frames)."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=cfg.top_db)
    return log_mel.astype(np.float32)


def extract_scalar_features(audio: np.ndarray, cfg: AudioConfig = AUDIO_CONFIG) -> dict:
    """Auxiliary scalar features (not fed to the CNN, but exposed via the API /
    used for exploratory analysis and sanity-checking)."""
    stft = np.abs(librosa.stft(audio, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    centroid = librosa.feature.spectral_centroid(S=stft, sr=cfg.sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=cfg.sample_rate)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=cfg.sample_rate, n_mfcc=13)
    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "rms_energy_mean": float(np.mean(rms)),
        "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
    }


def prepare_model_input(
    audio_segment: np.ndarray,
    mean: float,
    std: float,
    cfg: AudioConfig = AUDIO_CONFIG,
) -> np.ndarray:
    """
    Full segment -> model-ready tensor pipeline:
    normalize waveform -> log-mel spectrogram -> standardize with
    TRAIN-SET statistics (mean/std loaded from models/feature_stats.json).

    Returns shape (1, n_mels, n_frames) ready to be stacked into a batch
    and converted to a torch tensor.
    """
    audio_segment = normalize_waveform(audio_segment)
    mel = generate_mel_spectrogram(audio_segment, cfg)
    mel = (mel - mean) / (std + 1e-8)
    return mel[np.newaxis, :, :]
