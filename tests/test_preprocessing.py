import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import AUDIO_CONFIG
from common.audio_features import (
    normalize_waveform, segment_waveform, generate_mel_spectrogram, prepare_model_input,
)


def test_normalize_waveform_peak_is_one():
    audio = np.array([0.1, -0.5, 0.25, -0.05], dtype=np.float32)
    normed = normalize_waveform(audio)
    assert np.isclose(np.max(np.abs(normed)), 1.0, atol=1e-5)


def test_normalize_waveform_handles_silence():
    audio = np.zeros(1000, dtype=np.float32)
    normed = normalize_waveform(audio)
    assert np.all(normed == 0)


def test_segment_waveform_produces_correct_length_segments():
    sr = AUDIO_CONFIG.sample_rate
    audio = np.random.randn(sr * 5).astype(np.float32)  # 5 seconds
    segments = segment_waveform(audio, AUDIO_CONFIG)
    assert len(segments) > 0
    for seg in segments:
        assert len(seg) == AUDIO_CONFIG.segment_samples


def test_segment_waveform_drops_too_short_tail():
    sr = AUDIO_CONFIG.sample_rate
    audio = np.random.randn(int(sr * 0.1)).astype(np.float32)  # 0.1s, way under min
    segments = segment_waveform(audio, AUDIO_CONFIG)
    assert segments == []


def test_mel_spectrogram_shape_matches_config():
    sr = AUDIO_CONFIG.sample_rate
    audio = np.random.randn(AUDIO_CONFIG.segment_samples).astype(np.float32)
    mel = generate_mel_spectrogram(audio, AUDIO_CONFIG)
    assert mel.shape[0] == AUDIO_CONFIG.n_mels
    assert mel.ndim == 2


def test_prepare_model_input_shape_and_standardization():
    audio = np.random.randn(AUDIO_CONFIG.segment_samples).astype(np.float32)
    x = prepare_model_input(audio, mean=0.0, std=1.0, cfg=AUDIO_CONFIG)
    assert x.shape[0] == 1
    assert x.shape[1] == AUDIO_CONFIG.n_mels
