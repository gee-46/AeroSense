import numpy as np

from common.audio_features import load_audio, segment_waveform, normalize_waveform, generate_mel_spectrogram
from common.config import AUDIO_CONFIG
from app.core.exceptions import InvalidAudioError


def load_and_segment(path: str) -> list[np.ndarray]:
    """Load -> mono/resample -> segment. Reuses the EXACT same functions
    used during training (common/audio_features.py) so there is no
    train/inference skew."""
    audio = load_audio(path, AUDIO_CONFIG)
    if len(audio) == 0:
        raise InvalidAudioError("Decoded audio has zero length")
    segments = segment_waveform(audio, AUDIO_CONFIG)
    if not segments:
        raise InvalidAudioError(
            f"Audio is too short to analyze (need at least "
            f"{AUDIO_CONFIG.segment_duration * 0.5:.1f}s, got {len(audio)/AUDIO_CONFIG.sample_rate:.2f}s)"
        )
    return segments


def segments_to_model_batch(segments: list[np.ndarray], mean: float, std: float) -> np.ndarray:
    """Normalize + mel-spectrogram + standardize each segment, stack into a batch."""
    mels = []
    for seg in segments:
        seg = normalize_waveform(seg)
        mel = generate_mel_spectrogram(seg, AUDIO_CONFIG)
        mel = (mel - mean) / (std + 1e-8)
        mels.append(mel[np.newaxis, :, :])
    return np.stack(mels, axis=0)
