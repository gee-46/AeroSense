"""PyTorch Dataset over the segment manifest produced by prepare_dataset.py."""
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import AUDIO_CONFIG, CLASS_NAMES, MODEL_DIR
from common.audio_features import load_audio, normalize_waveform, generate_mel_spectrogram

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_feature_stats():
    with open(os.path.join(MODEL_DIR, "feature_stats.json")) as f:
        stats = json.load(f)
    return stats["log_mel_mean"], stats["log_mel_std"]


class AeroSenseSegmentDataset(Dataset):
    """
    Loads raw audio lazily per-file (small LRU cache) rather than reading
    each segment as a separate file, since segments are offsets into a
    handful of source recordings.
    """

    def __init__(self, manifest_path: str, split: str, augment: bool = False, cache_size: int = 6):
        self.rows = []
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    row["start_sample"] = int(row["start_sample"])
                    self.rows.append(row)
        self.split = split
        self.augment = augment
        self.mean, self.std = load_feature_stats()
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size

    def __len__(self):
        return len(self.rows)

    def _get_audio(self, path: str) -> np.ndarray:
        if path not in self._cache:
            self._cache[path] = load_audio(path)
            self._cache_order.append(path)
            if len(self._cache_order) > self._cache_size:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        return self._cache[path]

    def _augment_waveform(self, seg: np.ndarray) -> np.ndarray:
        # Light, label-preserving augmentations: gaussian noise + amplitude
        # scaling + time shift. Applied only when self.augment=True (train split).
        if np.random.rand() < 0.5:
            noise_level = np.random.uniform(0.001, 0.01)
            seg = seg + np.random.randn(*seg.shape).astype(np.float32) * noise_level
        if np.random.rand() < 0.5:
            gain = np.random.uniform(0.7, 1.3)
            seg = seg * gain
        if np.random.rand() < 0.5:
            shift = np.random.randint(-len(seg) // 10, len(seg) // 10 + 1)
            seg = np.roll(seg, shift)
        return seg.astype(np.float32)

    def _spec_augment(self, mel: np.ndarray) -> np.ndarray:
        # Frequency + time masking (SpecAugment-style), train split only.
        mel = mel.copy()
        n_mels, n_frames = mel.shape
        if np.random.rand() < 0.5:
            f = np.random.randint(1, max(2, n_mels // 8))
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel[f0:f0 + f, :] = mel.min()
        if np.random.rand() < 0.5:
            t = np.random.randint(1, max(2, n_frames // 8))
            t0 = np.random.randint(0, max(1, n_frames - t))
            mel[:, t0:t0 + t] = mel.min()
        return mel

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio = self._get_audio(row["path"])
        seg_len = AUDIO_CONFIG.segment_samples
        start = row["start_sample"]
        seg = audio[start:start + seg_len]
        if len(seg) < seg_len:
            pad = np.zeros(seg_len, dtype=np.float32)
            pad[: len(seg)] = seg
            seg = pad

        seg = normalize_waveform(seg)
        if self.augment:
            seg = self._augment_waveform(seg)

        mel = generate_mel_spectrogram(seg)
        mel = (mel - self.mean) / (self.std + 1e-8)
        if self.augment:
            mel = self._spec_augment(mel)

        label = CLASS_TO_IDX[row["label"]]
        x = torch.from_numpy(mel[np.newaxis, :, :].astype(np.float32))
        return x, label
